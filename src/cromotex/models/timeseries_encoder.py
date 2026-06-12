import torch
import torch.nn as nn
import torch.nn.functional as F
import hydra
import os

class MLPClassifHead(nn.Module):
    def __init__(self, cfg):
        super(MLPClassifHead, self).__init__()
        embed_dim = cfg.cromotex.embed_dim
        hidden_dim = cfg.cromotex.classif_head_hid_dim
        num_layers = cfg.cromotex.classif_head_num_layers
        dropout = cfg.cromotex.classif_head_dropout

        layers = []
        for i in range(num_layers):
            in_dim = embed_dim if i == 0 else hidden_dim
            out_dim = len(cfg.pathology) if i == num_layers - 1 else hidden_dim
            
            layers.append(nn.Linear(in_dim, out_dim))
            
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim)) 
                layers.append(nn.GELU()) 
                layers.append(nn.Dropout(dropout))
                
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class PatchEmbed(nn.Module):
    '''
    Patch Embedding for ECG data to be input to ECGTransformer.
    Converts (12, 1000) to 196 patches of dim patch_dim.
    Source: https://github.com/svthapa/MoRE/blob/main/utils/build_model.py
    '''
    def __init__(
        self, in_channels=12, patch_dim=256,
        intermediate_dim=128, kernel_size=5, stride1=5, stride2=1
    ):
        super(PatchEmbed, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels, out_channels=intermediate_dim,
                kernel_size=kernel_size, stride=stride1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_dim),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=intermediate_dim, out_channels=patch_dim,
                kernel_size=kernel_size, stride=stride2
            ),
            nn.ReLU(),
            nn.BatchNorm1d(patch_dim),
        )
        self.num_patches = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, patch_dim]
        self.num_patches = x.size(1)  # Save the number of patches
        return x

class ECGPatchTransformer(nn.Module):
    '''
    Transformer for ECG data to generate global
    embeddings for contrastive learning.
    '''
    def __init__(
        self, 
        cfg,
        in_channels=12, 
        seq_len=1000, 
        intermediate_dim=128, 
        kernel_size=5, 
        stride1=5, 
        stride2=1,
        embed_dim=256, 
        num_heads=4, 
        depth=4, 
        num_classes=1
        ):
        super(ECGPatchTransformer, self).__init__()

        if cfg is not None:
            kernel_size = cfg.cromotex.kernel_size
            stride1 = cfg.cromotex.stride1
            stride2 = cfg.cromotex.stride2
            intermediate_dim = cfg.cromotex.intermediate_dim
            embed_dim = cfg.cromotex.embed_dim
            
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            patch_dim=embed_dim,
            intermediate_dim=intermediate_dim,
            kernel_size=kernel_size,
            stride1=stride1,
            stride2=stride2
        )
        self.cfg = cfg
        max_patches = 196
        self.positional_embedding = nn.Parameter(
            torch.randn(1, max_patches + 1, embed_dim)  # +1 for CLS token
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))


        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=depth
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else None
        )

    def forward(self, x):
        # Generate patches and embeddings
        x = self.patch_embed(x)  # [batch_size, num_patches, patch_dim]

        if self.training and self.cfg.cromotex.patch_drop > 0:
            drop_mask = (
                torch.rand(x.size(0), x.size(1), device=x.device)
                < self.cfg.cromotex.patch_drop
            )
            drop_mask = drop_mask.unsqueeze(-1)
            x = x * (~drop_mask)
        
        num_patches = x.size(1)
        pos_embed = self.positional_embedding[:, :num_patches + 1, :]

        cls_token = self.cls_token.expand(x.size(0), -1, -1)  
        # [batch_size, 1, embed_dim]

        x = torch.cat((cls_token, x), dim=1)

        x = x + pos_embed  
        # Ensure positional embedding matches num_patches

        # x = self.transformer_encoder(x)  # [batch_size, num_patches, embed_dim]

        # x = x.permute(0, 2, 1)  # [batch_size, embed_dim, num_patches]
        # global_embedding = self.global_pool(x).squeeze(-1)  
        # # [batch_size, embed_dim]

        # Transformer 输出：所有 token（cls + patches）
        tokens_all = self.transformer_encoder(x)  
        # [B, 1 + num_patches, embed_dim]
        
        # 拆分：cls token + 纯 patch 特征
        cls_token_out = tokens_all[:, 0:1, :]    # [B, 1, embed_dim]
        patches_out  = tokens_all[:, 1:, :]      # [B, num_patches, embed_dim]
        # ===================================================

        # 原来的全局 embedding 逻辑不变
        x_perm = tokens_all.permute(0, 2, 1)     # [B, embed_dim, num_patches+1]
        global_embedding = self.global_pool(x_perm).squeeze(-1)  # [B, embed_dim]

        # Optional classification
        if self.classifier is not None:
            logits = self.classifier(global_embedding)  
            # [batch_size, num_classes]
            return global_embedding, logits, tokens_all, patches_out

        return global_embedding, tokens_all, patches_out

class ECGTimeseriesEncoder(nn.Module):
    def __init__(self, cfg):
        super(ECGTimeseriesEncoder, self).__init__()
        self.cfg = cfg
        self.timeseries_encoder = ECGPatchTransformer(cfg)
        self.classif_head = MLPClassifHead(cfg)
        filepath = os.path.join(
            hydra.utils.to_absolute_path('checkpoints'),
            f'biot_pretrain_ecg_{cfg.ecg_pth_name[0]}_{cfg.ecg_pth_name[1]}.pth'
        )

        checkpoint = torch.load(filepath, map_location='cpu')
        print(f"Loaded pretrained timeseries encoder from {filepath}")

        ts_encoder_state_dict = {
            k.replace('timeseries_encoder.', ''): v
            for k, v in checkpoint['model_state_dict'].items()
            if k.startswith('timeseries_encoder')
        }

        self.timeseries_encoder.load_state_dict(ts_encoder_state_dict)

    def forward(self, x):
        embeds, _, _ , patches_out = self.timeseries_encoder(x)
        embeds = F.normalize(embeds, dim=-1)
        logits = self.classif_head(embeds)
        return embeds, logits, patches_out