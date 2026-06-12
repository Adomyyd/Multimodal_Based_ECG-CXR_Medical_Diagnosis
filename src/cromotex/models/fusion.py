import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import hydra
import numpy as np
from .timeseries_encoder import ECGPatchTransformer
from .image_encoder import get_image_encoder
from .cromotex import MLPClassifHead, get_cromotex, CroMoTEXFinetune
from src.cromotex.utils.utils import load_train_checkpoint
from src.cromotex.utils.ts_augmentations import ECGAugmentor, VCG_ECGAugmentor

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class MultiHeadCrossAttention1D(nn.Module):
    """
    多头交叉注意力层，用于处理1D特征向量（无序列维度）
    适用于CXR图像特征和ECG信号特征的融合
    """

    def __init__(self, input_dim = 256, output_dim = 256, num_heads = 8, dropout=0.1):
        super(MultiHeadCrossAttention1D, self).__init__()

        assert output_dim % num_heads == 0, "output_dim必须能被num_heads整除"

        self.output_dim = output_dim
        self.num_heads = num_heads
        self.d_k = output_dim // num_heads

        # 线性变换层
        self.w_q_1 = nn.Linear(input_dim, output_dim)  # Query来自第一个特征
        self.w_k_2 = nn.Linear(input_dim, output_dim)  # Key来自第二个特征
        self.w_v_2 = nn.Linear(input_dim, output_dim)  # Value来自第二个特征

        # 输出层
        self.w_o = nn.Linear(output_dim, output_dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.w_q_1.weight)
        nn.init.xavier_uniform_(self.w_k_2.weight)
        nn.init.xavier_uniform_(self.w_v_2.weight)
        nn.init.xavier_uniform_(self.w_o.weight)

        # 偏置初始化为0
        if self.w_q_1.bias is not None:
            nn.init.constant_(self.w_q_1.bias, 0)
        if self.w_k_2.bias is not None:
            nn.init.constant_(self.w_k_2.bias, 0)
        if self.w_v_2.bias is not None:
            nn.init.constant_(self.w_v_2.bias, 0)
        if self.w_o.bias is not None:
            nn.init.constant_(self.w_o.bias, 0)

    def forward(self, features1, features2):
        """
        前向传播

        Args:
            features1: 第一个特征 [batch_size, input_dim]
            features2: 第二个特征 [batch_size, input_dim]

        Returns:
            fused_features: 融合后的特征 [batch_size, output_dim]
            attention_weights: 注意力权重 [batch_size, num_heads]
        """
        batch_size = features1.size(0)

        # 线性变换并扩展维度（添加序列长度维度1）
        Q = self.w_q_1(features1).unsqueeze(1)  # [batch_size, 1, output_dim]
        K = self.w_k_2(features2).unsqueeze(1)  # [batch_size, 1, output_dim]
        V = self.w_v_2(features2).unsqueeze(1)  # [batch_size, 1, output_dim]

        # 分割成多个头
        Q = Q.view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, 1, d_k]
        K = K.view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, 1, d_k]
        V = V.view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)  # [batch_size, num_heads, 1, d_k]

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, 1, 1]
        attention_weights = self.dropout(attention_weights)

        # 加权求和：注意力权重 × V
        context = torch.matmul(attention_weights, V)  # [batch_size, num_heads, 1, d_k]

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, 1, self.output_dim
        ).squeeze(1)  # [batch_size, output_dim]

        # 输出层
        fused_features = self.w_o(context)  # [batch_size, output_dim]
        fused_features = self.dropout(fused_features)

        # 添加残差连接和层归一化
        # 注意：这里我们假设features1可以映射到output_dim维度
        if features1.shape[-1] == self.output_dim:
            fused_features = self.layer_norm(fused_features + features1)
        else:
            # 如果维度不匹配，先投影
            residual = self.w_q_1(features1)
            fused_features = self.layer_norm(fused_features + residual)

        # 去掉序列维度，返回 [batch_size, output_dim]
        attention_weights = attention_weights.squeeze(2).squeeze(2)  # [batch_size, num_heads]

        return fused_features, attention_weights
    
class BiDirectionalMultiHeadCrossAttention1D(nn.Module):
    """
    双向多头交叉注意力层（处理1D特征向量）
    同时考虑两个方向的融合：A→B 和 B→A
    """

    def __init__(self, input_dim, output_dim, num_heads, dropout=0.1):
        super(BiDirectionalMultiHeadCrossAttention1D, self).__init__()

        # 第一个方向：A作为query，B作为key/value
        self.attn_a_to_b = MultiHeadCrossAttention1D(
            input_dim, output_dim, num_heads, dropout
        )

        # 第二个方向：B作为query，A作为key/value
        self.attn_b_to_a = MultiHeadCrossAttention1D(
            input_dim, output_dim, num_heads, dropout
        )

        # 融合层：将两个方向的增强特征融合
        self.fusion_linear = nn.Linear(output_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, features1, features2):
        """
        双向融合

        Args:
            features1: 第一个特征 [batch_size, input_dim]
            features2: 第二个特征 [batch_size, input_dim]

        Returns:
            fused_features: 融合后的特征 [batch_size, output_dim]
            attn_weights_1: A→B注意力权重 [batch_size, num_heads]
            attn_weights_2: B→A注意力权重 [batch_size, num_heads]
        """
        # A → B
        enhanced_a, attn_weights_1 = self.attn_a_to_b(features1, features2)

        # B → A
        enhanced_b, attn_weights_2 = self.attn_b_to_a(features2, features1)

        # 拼接双向融合的特征
        combined = torch.cat([enhanced_a, enhanced_b], dim=-1)  # [batch_size, output_dim*2]

        # 融合
        fused_features = self.fusion_linear(combined)
        fused_features = self.dropout(fused_features)
        fused_features = self.layer_norm(fused_features)

        return fused_features, attn_weights_1, attn_weights_2
    
class CrossModalFusion(nn.Module):
    """
    医疗特征融合模块
    用于CXR图像特征和ECG信号特征的融合
    """

    def __init__(self, cfg, logger, feature_dim = 256, output_dim=256, num_heads=8, dropout=0.1):
        super(CrossModalFusion, self).__init__()

        cromotex = get_cromotex(cfg)

        ckpt_filename = cfg.finetune.ckpt_filename

        assert len(ckpt_filename) > 0, "必须提供预训练模型的ckpt_filename"

        ckpt_filename = f'cromotex_last_{cfg.pathology}_{ckpt_filename}'
        self.cromotex, _, _, _ = load_train_checkpoint(
            ckpt_filename, cromotex
        )
    
        logger.info(
            f"Loaded trained checkpoint {ckpt_filename}"
        )

        ckpt_file = os.path.join(
            hydra.utils.to_absolute_path('checkpoints'),
            ckpt_filename
        )
        ckpt_time = time.ctime(os.path.getmtime(ckpt_file))
        logger.info(f"Checkpoint was saved at: {ckpt_time}")

        # 双向交叉注意力融合
        self.cross_attention = BiDirectionalMultiHeadCrossAttention1D(
            feature_dim, output_dim, num_heads, dropout
        )

        # 分类头（用于下游任务）
        self.classifier = MLPClassifHead(cfg)

    def forward(self, cxr, ecg, return_attention=False):
        """
        多模态融合前向传播

        Args:
            cxr: CXR图像
            ecg: ECG信号
            return_attention: 是否返回注意力权重

        Returns:
            class_logits: 分类结果 [batch_size, class_num]
            fused_features: 融合后的特征 [batch_size, output_dim]
            attention_weights: 注意力权重（如果return_attention=True）
        """
        cxr_features, ecg_features, _ = self.cromotex(cxr, ecg)

        # 双向交叉注意力融合
        fused_features, attn_weights_1, attn_weights_2 = self.cross_attention(
            cxr_features, ecg_features
        )

        # 分类
        class_logits = self.classifier(fused_features)

        if return_attention:
            return class_logits, fused_features, (attn_weights_1, attn_weights_2)
        else:
            return class_logits, fused_features

class ECGCXRDynMM(nn.Module):
    def __init__(self, cfg, logger, infer_mode=0):
        super(ECGCXRDynMM, self).__init__()
        self.cfg = cfg
        # 计算分支数量
        self.branch_num = 2 + (1 if cfg.dynmm.expert_fusion else 0)
        self.expert_fusion = cfg.dynmm.expert_fusion

        # cromotex
        cromotex_finetune = CroMoTEXFinetune(cfg, logger)
        ckpt_filename = cfg.dynmm.finetune_ckpt_filename
        self.cromotex_finetune, _, _, _ = load_train_checkpoint(
                ckpt_filename, cromotex_finetune
            )
        logger.info(
            f"Loaded cromotex_finetune checkpoint {ckpt_filename}"
        )

        ckpt_file = os.path.join(
            hydra.utils.to_absolute_path('checkpoints'),
            ckpt_filename
        )
        ckpt_time = time.ctime(os.path.getmtime(ckpt_file))
        logger.info(f"Checkpoint was saved at: {ckpt_time}")

        self.ts_augmentor = ECGAugmentor()
        img_augs = get_image_encoder(cfg).get_augmentations(cfg)
        self.img_augs_train, self.img_augs_val = img_augs

        # CXR_classifier
        self.image_classifier = get_image_encoder(cfg)

        filepath = os.path.join(
            hydra.utils.to_absolute_path('checkpoints'),
            f'pretrain_img_{cfg.img_pth_name[0]}_{cfg.pathology}_{cfg.img_pth_name[1]}.pth'
        )

        checkpoint = torch.load(filepath, map_location='cpu')
        logger.info(f"Loaded pretrained image classifier from {filepath}")

        if isinstance(self.image_classifier, torch.nn.DataParallel):
            self.image_classifier.module.load_state_dict(
                checkpoint['model_state_dict'], strict=True
            )
        else:
            self.image_classifier.load_state_dict(
                checkpoint['model_state_dict'], strict=True
            )
        
        # ECG + CXR fusion
        if self.expert_fusion:
            if cfg.dynmm.fusion_pretrain:
                # 假设已经训练好的融合模型
                self.fusion_model = torch.load('./log/ecg_cxr/best_fusion.pt')
            else:
                # 从头训练的融合模型
                self.fusion_model = CrossModalFusion(cfg)
        
        # Freeze branches if needed
        if cfg.dynmm.freeze:
            self.freeze_branch(self.cromotex_finetune)
            self.freeze_branch(self.image_classifier)
            if self.expert_fusion:
                self.freeze_branch(self.fusion_model)

        feature_dim = cfg.cromotex.proj_dim
        
        # Gating network
        # 输入是ECG特征和CXR特征的拼接
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.branch_num)
        )
        
        self.temp = 1.0
        self.hard_gate = True
        self.weight_list = torch.Tensor()
        self.store_weight = cfg.dynmm.store_weight
        self.infer_mode = infer_mode

    def get_augmentations(self):
        return self.img_augs_train, self.img_augs_val, self.ts_augmentor
    
    def freeze_branch(self, m):
        for param in m.parameters():
            param.requires_grad = False
    
    def forward(self, ecg, cxr):
        # 提取特征
        _, cxr_embed, _= self.cromotex_finetune.cromotex(cxr, ecg)
        pred_ecg, ecg_embed = self.cromotex_finetune(ecg)
        
        # 拼接特征作为门控网络的输入
        gate_input = torch.cat([ecg_embed, cxr_embed], dim=1)
        weight = DiffSoftmax(self.gate(gate_input), tau=self.temp, hard=self.hard_gate)
        
        if self.store_weight:
            self.weight_list = torch.cat((self.weight_list, weight.cpu()))
        
        # 计算各个分支的预测
        pred_cxr = self.image_classifier(cxr)
        
        if self.expert_fusion:
            pred_fusion = self.fusion_model((ecg_embed, cxr_embed))
        
        # 推理模式：直接返回指定分支的结果
        if self.infer_mode > 0:
            if self.infer_mode == 1:
                return pred_ecg
            elif self.infer_mode == 2:
                return pred_cxr
            elif self.infer_mode == 3 and self.expert_fusion:
                return pred_fusion
        
        # 加权融合
        if self.expert_fusion:
            output = weight[:, 0:1] * pred_ecg + weight[:, 1:2] * pred_cxr + weight[:, 2:3] * pred_fusion
        else:
            output = weight[:, 0:1] * pred_ecg + weight[:, 1:2] * pred_cxr
        
        
        return output
    
    def forward_separate_branch(self, inputs, path):
        """用于单独测试各个分支的性能"""
        ecg_features, cxr_features = inputs
        
        if path == 1:
            output = self.ecg_classif(self.ecg_encoder(ecg_features))
        elif path == 2:
            output = self.cxr_classif(self.cxr_encoder(cxr_features))
        elif path == 3 and self.expert_fusion:
            output = self.fusion_model(inputs)
        else:
            output = None
        
        return output