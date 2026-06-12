import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import kendalltau

# class SMLILoss(nn.Module):
#     def __init__(self):
#         super(SMLILoss, self).__init__()
#         self.tau1 = 0.1
#         self.tau2 = 0.1
#         self.reg_weight = 0.1
        
#     def _soft_matching_late_interaction(self, ecg_patches, xray_patches, return_weights=False):
#         """
#         多对多软性后期交互
#         ecg_patches: [B, N_e, D]  ECG局部patch
#         xray_patches: [B, N_x, D] X光局部patch
#         return: 双向对比局部logits
#         """
#         B, N_e, D = ecg_patches.shape
#         N_x = xray_patches.size(1)

#         # 1. L2归一化
#         feat_e = F.normalize(ecg_patches, dim=-1)
#         feat_x = F.normalize(xray_patches, dim=-1)

#         # 2. 全局部两两相似度 [B, N_e, N_x]
#         sim = torch.bmm(feat_e, feat_x.transpose(1, 2))
#         sim_batch = torch.einsum('bnd,cjd->bcnj', feat_e, feat_x) # [B,B,N_e,N_x]

#         # ===================== 多对多软匹配 =====================
#         # ECG → X光：每个心电片段软匹配多个病灶区域
#         weight_e2x = (sim / self.tau1).softmax(dim=-1)  # [B,N_e,N_x]
#         # xray_aggregate = torch.bmm(weight_e2x, feat_x)  # [B,N_e,D]
#         # xray_aggregate = F.normalize(xray_aggregate, dim=-1)
#         # feat_e_expand = feat_e.unsqueeze(1)      # [B,1,N_e,D]
#         # xray_agg_expand = xray_aggregate.unsqueeze(0)  # [1,B,N_e,D]

#         # sim_cross = torch.einsum('bind,cjnd->bjn', feat_e_expand, xray_agg_expand)  # [B,B,N_e]

#         # # log-sum-exp pooling
#         # sim_matrix_e2x = torch.logsumexp(sim_cross / self.tau2, dim=-1)  # [B,B]

#         # X光 → ECG：每个病灶区域软匹配多段心电
#         weight_x2e = (sim / self.tau1).softmax(dim=1)  # [B,N_e,N_x]
#         # ecg_aggregate = torch.bmm(weight_x2e.transpose(1,2), feat_e)  # [B,N_x,D]
#         # ecg_aggregate = F.normalize(ecg_aggregate, dim=-1)
#         # feat_x_expand = feat_x.unsqueeze(1)      # [B,1,N_x,D]
#         # ecg_agg_expand = ecg_aggregate.unsqueeze(0)  # [1,B,N_x,D]

#         # sim_cross = torch.einsum('bind,cjnd->bjn', feat_x_expand, ecg_agg_expand)  # [B,B,N_x]

#         # # log-sum-exp pooling
#         # sim_matrix_x2e = torch.logsumexp(sim_cross / self.tau2, dim=-1)  # [B,B]
#         sim_matrix_e2x = torch.logsumexp(sim_batch / self.tau2, dim=-1)
#         sim_matrix_e2x = sim_matrix_e2x.mean(-1)
#         sim_matrix_x2e = torch.logsumexp(sim_batch / self.tau2, dim=2)
#         sim_matrix_x2e = sim_matrix_x2e.mean(-1)

#         reg_loss = (
#             F.kl_div(weight_e2x.log(), weight_x2e.detach(), reduction='batchmean') +
#             F.kl_div(weight_x2e.log(), weight_e2x.detach(), reduction='batchmean')
#         ) / 2

#         if return_weights:
#             return sim_matrix_e2x, sim_matrix_x2e, weight_e2x, weight_x2e, reg_loss

#         return sim_matrix_e2x, sim_matrix_x2e, reg_loss

#     def forward(self, xray_patches, ecg_patches, return_weights=False):
#         """多对多细粒度对齐损失"""
#         logits_e2x, logits_x2e, weight_e2x, weight_x2e, reg_loss = self._soft_matching_late_interaction(
#             ecg_patches, xray_patches, return_weights=True
#         )
#         labels = torch.arange(logits_e2x.size(0)).to(ecg_patches.device)
#         loss_e = F.cross_entropy(logits_e2x, labels)
#         loss_x = F.cross_entropy(logits_x2e, labels)

#         if return_weights:
#             return (loss_e + loss_x) / 2 + self.reg_weight * reg_loss, weight_e2x, weight_x2e

#         return (loss_e + loss_x) / 2 + self.reg_weight * reg_loss\
        
class SMLILoss(nn.Module):
    def __init__(self, cfg):
        super(SMLILoss, self).__init__()
        self.cfg = cfg
        self.tau1 = 0.1
        self.tau2 = 0.1
        self.reg_weight = 0.1
        self.krc_threshold = getattr(cfg.CLKD_train, 'krc_threshold', 0)  # 肯德尔阈值
        
    def _soft_matching_late_interaction(self, ecg_patches, xray_patches, return_weights=False):
        """
        多对多软性后期交互
        ecg_patches: [B, N_e, D]  ECG局部patch
        xray_patches: [B, N_x, D] X光局部patch
        return: 双向对比局部logits
        """
        B, N_e, D = ecg_patches.shape
        N_x = xray_patches.size(1)

        # 1. L2归一化
        feat_e = F.normalize(ecg_patches, dim=-1)
        feat_x = F.normalize(xray_patches, dim=-1)

        # 2. 全局部两两相似度 [B, N_e, N_x]
        sim = torch.bmm(feat_e, feat_x.transpose(1, 2))

        # ===================== 多对多软匹配 =====================
        # ECG → X光：每个心电片段软匹配多个病灶区域
        weight_e2x = (sim / self.tau1).softmax(dim=-1)  # [B,N_e,N_x]
        xray_aggregate = torch.bmm(weight_e2x, feat_x)  # [B,N_e,D]
        xray_aggregate = F.normalize(xray_aggregate, dim=-1)
        # feat_e_expand = feat_e.unsqueeze(1)      # [B,1,N_e,D]
        # xray_agg_expand = xray_aggregate.unsqueeze(0)  # [1,B,N_e,D]

        sim_cross = torch.einsum('bnd,cnd->bcn', feat_e, xray_aggregate)  # [B,B,N_e]

        # log-sum-exp pooling
        sim_matrix_e2x = torch.logsumexp(sim_cross / self.tau2, dim=-1)  # [B,B]

        # X光 → ECG：每个病灶区域软匹配多段心电
        weight_x2e = (sim / self.tau1).softmax(dim=1)  # [B,N_e,N_x]
        ecg_aggregate = torch.bmm(weight_x2e.transpose(1,2), feat_e)  # [B,N_x,D]
        ecg_aggregate = F.normalize(ecg_aggregate, dim=-1)
        # feat_x_expand = feat_x.unsqueeze(1)      # [B,1,N_x,D]
        # ecg_agg_expand = ecg_aggregate.unsqueeze(0)  # [1,B,N_x,D]

        sim_cross = torch.einsum('bnd,cnd->bcn', feat_x, ecg_aggregate)  # [B,B,N_x]

        # log-sum-exp pooling
        sim_matrix_x2e = torch.logsumexp(sim_cross / self.tau2, dim=-1)  # [B,B]

        reg_loss = (
            F.kl_div(weight_e2x.log(), weight_x2e.detach(), reduction='batchmean') +
            F.kl_div(weight_x2e.log(), weight_e2x.detach(), reduction='batchmean')
        ) / 2

        if return_weights:
            return sim_matrix_e2x, sim_matrix_x2e, weight_e2x, weight_x2e, reg_loss

        return sim_matrix_e2x, sim_matrix_x2e, reg_loss

    def forward(self, xray_patches, ecg_patches, img_logits, ts_logits, return_weights=False):
        if self.cfg.CLKD_train.use_krc:
            krc_mask, _, _ = self.kendall_mask(img_logits, ts_logits)
            keep = krc_mask.bool()  # 要保留的样本

            # 真正丢弃 mask=0 的样本
            xray_patches = xray_patches[keep]
            ecg_patches = ecg_patches[keep]
        """多对多细粒度对齐损失"""
        logits_e2x, logits_x2e, weight_e2x, weight_x2e, reg_loss = self._soft_matching_late_interaction(
            ecg_patches, xray_patches, return_weights=True
        )
        labels = torch.arange(logits_e2x.size(0)).to(ecg_patches.device)
        loss_e = F.cross_entropy(logits_e2x, labels)
        loss_x = F.cross_entropy(logits_x2e, labels)

        if return_weights:
            return (loss_e + loss_x) / 2 + self.reg_weight * reg_loss, weight_e2x, weight_x2e

        return (loss_e + loss_x) / 2 + self.reg_weight * reg_loss
    
    def kendall_mask(self, tea_logits, stu_logits):
        """计算肯德尔相关系数掩码：只保留高一致性样本"""
        B = tea_logits.size(0)
        kendall_scores = np.zeros(B, dtype=np.float32)
        
        tea_np = tea_logits.detach().cpu().numpy()
        stu_np = stu_logits.detach().cpu().numpy()
        
        for i in range(B):
            kendall_scores[i] = kendalltau(tea_np[i], stu_np[i])[0]
        
        kendall = torch.from_numpy(kendall_scores).to(tea_logits.device)
        mask = (kendall > self.krc_threshold).float()
        mask_sum = mask.sum()
        return mask, mask_sum, kendall.mean()