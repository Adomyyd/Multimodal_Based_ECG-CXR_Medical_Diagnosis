import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import kendalltau

class KDLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 配置参数
        self.temperature = getattr(cfg.CLKD_train, 'kd_temperature', 2.0)
        self.alpha = getattr(cfg.CLKD_train, 'kd_alpha', 1.0)
        self.bidirectional = getattr(cfg.CLKD_train, 'kd_bidirectional', False)
        self.krc_threshold = getattr(cfg.CLKD_train, 'krc_threshold', 0)  # 肯德尔阈值
        
        # 损失函数
        self.kd_loss = nn.BCELoss()


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

    def ntkl(self, logits_student, logits_teacher, labels, mask=None, temperature=1):
        # 将mask变为与logits相同的形状
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(logits_teacher)
        else:
            mask = torch.ones_like(logits_teacher)
        # print(f"Mask Shape: {mask.shape}, Mask:\n{mask}\n")

        if self.cfg.CLKD_train.use_NCKD:
            pred_teacher_part2 = mask * torch.sigmoid(logits_teacher / temperature) * (~(labels > 0).bool())
            pred_student_part2 = mask * torch.sigmoid(logits_student / temperature) * (~(labels > 0).bool())
        else:
            pred_teacher_part2 = mask * torch.sigmoid(logits_teacher / temperature)
            pred_student_part2 = mask * torch.sigmoid(logits_student / temperature)

        # print(f"Teacher Predictions (masked):\n{pred_teacher_part2}\n")
        # print(f"Student Predictions (masked):\n{pred_student_part2}\n")
        
        if mask.sum() == 0:
            temp = torch.tensor(0.0, device=logits_student.device)
        else:
            temp = self.kd_loss(pred_student_part2, pred_teacher_part2.detach())
        return temp

    def self_distill(self, logits1, logits2):
        """自蒸馏：模型自身输出对齐"""
        log_p = torch.log(torch.sigmoid(logits1 / self.temperature))
        p = torch.sigmoid(logits2.detach() / self.temperature)
        return self.kl_batch(log_p, p)

    def forward(self, img_logits, ts_logits, labels=None):
        """
        img_logits: 教师模态（图像/CXR）
        ts_logits:  学生模态（时序/ECG）
        """
        T = self.temperature
        tea = img_logits
        stu = ts_logits

        # ===================== 1. 肯德尔掩码 =====================
        if self.cfg.CLKD_train.use_krc:
            mask, mask_sum, corr_mean = self.kendall_mask(tea, stu)
        else:
            mask = torch.ones(tea.size(0), device=tea.device)
            mask_sum = tea.size(0)
            corr_mean = 0.0

        # ===================== 2. 噪声容忍双向蒸馏 =====================
        if mask_sum > 0:
            loss_t2s = self.ntkl(stu, tea, labels, mask, temperature=T)
            loss_s2t = self.ntkl(tea, stu, labels, mask, temperature=T) if self.bidirectional else 0.0
        else:
            loss_t2s = 0.0
            loss_s2t = 0.0

        # ===================== 3. 自蒸馏损失 =====================
        # self_tea = self.self_distill(tea, tea)
        # self_stu = self.self_distill(stu, stu)

        # ===================== 4. 总蒸馏损失 =====================
        if self.bidirectional:
            distill_loss = (loss_t2s + loss_s2t) / 2
        else:
            distill_loss = loss_t2s

        # 温度缩放
        loss = self.alpha * (T ** 2) * distill_loss
        return loss#, mask, mask_sum, corr_mean
    
if __name__ == "__main__":
    # 简单测试_多标签分类
    cfg = type('cfg', (object,), {})()
    cfg.CLKD_train = type('CLKD_train', (object,), {})()
    cfg.CLKD_train.kd_temperature = 2.0
    cfg.CLKD_train.kd_alpha = 1.0
    cfg.CLKD_train.kd_bidirectional = True
    cfg.CLKD_train.krc_threshold = 0.5

    kd_loss = KDLoss(cfg)

    batch_size = 4
    num_classes = 3
    img_logits = torch.randn(batch_size, num_classes)
    ts_logits = torch.randn(batch_size, num_classes)
    print(f"Image Logits:\n{img_logits}\n\nTime Series Logits:\n{ts_logits}\n")
    # 多标签二分类标签
    labels = torch.randint(0, 2, (batch_size, num_classes)).float()
    print(f"Labels:\n{labels}\n")

    loss, mask, mask_sum, corr_mean = kd_loss(img_logits, ts_logits, labels)
    print(f"KD Loss: {loss:.4f}")
    print(f"Mask: {mask}")
    print(f"Mask Sum: {mask_sum}")
    print(f"Correlation Mean: {corr_mean}")