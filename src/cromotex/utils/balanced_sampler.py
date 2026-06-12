import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler

def create_balanced_sampler(cfg=None, labels=None, pos_neg_ratio=1.0, num_classes=None):
    """
    创建多标签分类的平衡采样器
    
    解决MIMIC-IV等医疗数据集的类别极度不平衡问题，实现：
    1. 类别内正负样本比例平衡（通过pos_neg_ratio控制）
    2. 类别间样本数量平衡
    3. 多标签情况下的权重计算修正
    
    Args:
        cfg: 配置对象，包含随机种子等信息
        labels: 标签矩阵，形状为 (N, C)，其中 N 是样本数，C 是类别数
                1表示正例，0表示负例
        pos_neg_ratio: 类别内正负样本比例，默认1.0（1:1）
        num_classes: 类别数量，如果为None则从labels.shape[1]自动获取
    
    Returns:
        WeightedRandomSampler: 平衡采样器对象
    """
    # 自动获取类别数量
    if num_classes is None:
        num_classes = labels.shape[1]
    
    num_samples = len(labels)
    
    # 统计每个类别的正负样本数
    cls_pos_counts = []
    cls_neg_counts = []
    
    for c in range(num_classes):
        pos_num = (labels[:, c] == 1).sum()
        neg_num = (labels[:, c] == 0).sum()
        cls_pos_counts.append(pos_num)
        cls_neg_counts.append(neg_num)
    
    # 计算每个样本的权重
    sample_weights = np.zeros(num_samples, dtype=np.float32)
    
    for c in range(num_classes):
        pos_num = cls_pos_counts[c]
        neg_num = cls_neg_counts[c]
        
        # 类别内：让正负样本采样概率达到目标比例
        weight_pos = (1.0 / pos_num) * pos_neg_ratio if pos_num > 0 else 0
        weight_neg = 1.0 / neg_num if neg_num > 0 else 0
        
        # 为当前类别，给所有样本打上权重
        cls_weight = np.zeros(num_samples, dtype=np.float32)
        cls_weight[labels[:, c] == 1] = weight_pos
        cls_weight[labels[:, c] == 0] = weight_neg
        
        # 关键修正：取每个样本在所有类别中的最大权重（保证少数类不被淹没）
        sample_weights = np.maximum(sample_weights, cls_weight)
    
    # 构造采样器
    if cfg is not None:
        generator = torch.Generator().manual_seed(cfg.seed)
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights),
            num_samples=num_samples,
            replacement=True,
            generator=generator
        )
    else:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights),
            num_samples=num_samples,
            replacement=True
        )
    
    return sampler

def check_sampler_balance(sampler, labels, num_classes=None):
    """
    验证采样器的平衡效果
    
    Args:
        sampler: WeightedRandomSampler对象
        labels: 原始标签矩阵
        num_classes: 类别数量，如果为None则从labels.shape[1]自动获取
    
    Returns:
        dict: 包含每个类别的正负样本数和比例的字典
    """
    if num_classes is None:
        num_classes = labels.shape[1]
    
    # 模拟采样
    sample_indices = list(sampler)
    sampled_labels = labels[sample_indices]
    
    # 统计结果
    result = {}
    total_pos = 0
    total_neg = 0
    
    print("===== 采样后结果 ======")
    for c in range(num_classes):
        pos = (sampled_labels[:, c] == 1).sum()
        neg = (sampled_labels[:, c] == 0).sum()
        ratio = pos / neg if neg > 0 else 0
        total_pos += pos
        total_neg += neg
        
        print(f"类别{c}: 正={pos}, 负={neg}, 正负比={ratio:.2f}")
        result[f'class_{c}'] = {'pos': int(pos), 'neg': int(neg), 'ratio': float(ratio)}
    
    overall_ratio = total_pos / total_neg if total_neg > 0 else 0
    print(f"\n整体均衡度: {overall_ratio:.2f}")
    result['overall'] = {'total_pos': int(total_pos), 'total_neg': int(total_neg), 'ratio': float(overall_ratio)}
    
    return result

# 示例用法
if __name__ == "__main__":
    # 模拟MIMIC-IV极度不平衡数据
    np.random.seed(42)
    N = 10000
    NUM_CLASSES = 3
    
    # 生成9%正样本的极度不平衡数据
    train_labels = np.random.rand(N, NUM_CLASSES) < 0.09
    train_labels = train_labels.astype(np.int32)
    
    # 统计原始数据分布
    print("===== 原始数据分布 ======")
    for c in range(NUM_CLASSES):
        pos = (train_labels[:, c] == 1).sum()
        neg = (train_labels[:, c] == 0).sum()
        ratio = pos / neg
        print(f"类别{c}: 正={pos}, 负={neg}, 正负比={ratio:.2f}")
    
    # 创建平衡采样器
    sampler = create_balanced_sampler(
        cfg=None,
        labels=train_labels,
        pos_neg_ratio=1.0,  # 1:1比例
        num_classes=NUM_CLASSES
    )
    
    # 验证采样效果
    check_sampler_balance(sampler, train_labels)
