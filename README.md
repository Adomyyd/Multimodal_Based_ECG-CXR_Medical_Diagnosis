# CroMoTEX：跨模态知识注入的心电图病理检测

> Contrastive Cross-Modal Learning for Infusing Chest X-ray Knowledge into ECGs

本项目基于跨模态对比学习，将**胸片（CXR）所蕴含的病理知识**在训练阶段“注入”到**心电图（ECG）**表征中；推理阶段**只需 ECG 单模态**即可完成心胸相关病理的多标签检测，从而避免了部署时对昂贵胸片的依赖。

本仓库在官方 CroMoTEX 方法（[arXiv:2506.19329](https://arxiv.org/abs/2506.19329)）基础上，额外实现并扩展了 **CLKD（Contrastive Learning + Knowledge Distillation）** 模块，引入了基于 logits 的跨模态知识蒸馏、NCKD（非目标类蒸馏）、KRC（Kendall 秩相关样本筛选）等机制。

---

## 目录

- [核心思想](#核心思想)
- [模型架构](#模型架构)
- [项目结构](#项目结构)
- [环境安装](#环境安装)
- [数据准备](#数据准备)
- [训练流程](#训练流程)
- [评估](#评估)
- [配置说明](#配置说明)
- [引用](#引用)

---

## 核心思想

传统多模态模型在推理时要求所有模态同时存在，但胸片采集成本高、并非所有就诊场景都可获取。CroMoTEX 的关键洞见是：

- **胸片是“教师/辅助模态”**：富含心脏扩大、肺水肿、胸腔积液等病理的直接视觉证据。
- **心电图是“学生/主模态”**：采集便捷、成本低，但对上述病理的信号较为微弱。

训练时通过**跨模态监督对比学习**把配对的 CXR-ECG 拉近、不同病理的样本推远，使 ECG 编码器学会“胸片视角”的判别特征；推理时丢弃图像分支，**仅用 ECG 编码器 + 分类头**输出病理预测。

检测的三种目标病理（多标签）：

| 病理 | 配置标识 | 说明 |
|------|----------|------|
| 心脏扩大 | `cardiomegaly` | Cardiomegaly |
| 肺水肿 | `edema` | Edema |
| 胸腔积液 | `pleural_effusion` | Pleural Effusion（在 xrv 中映射为 `Effusion`） |

---

## 模型架构

### 编码器

| 模态 | 编码器 | 实现 | 说明 |
|------|--------|------|------|
| 胸片 CXR | DenseNet-121 | `models/image_encoder.py` | 基于 [torchxrayvision](https://github.com/mlmed/torchxrayvision)，加载 `densenet121-res224-mimic_nb` 预训练权重 |
| 心电图 ECG | Patch Transformer | `models/timeseries_encoder.py` | 12 导联 × 1000 采样点，先经两层 Conv1D 切分为 patch，再过 Transformer Encoder + CLS token |

ECG 输入形状为 `[batch, 12, 1000]`（12 导联，10 秒 @ 100Hz），经 `PatchEmbed`（Conv1D，kernel=5）转为 patch 序列，加入位置编码与 CLS token 后送入 4 层 `TransformerEncoder`，全局池化得到 256 维嵌入。

### 损失函数

| 损失 | 文件 | 作用 |
|------|------|------|
| **AHNP Loss** | `models/ahnp_loss.py` | 跨模态监督对比损失（Adaptive Hard-Negative Pairing），对难负样本加权（`topk`/`linear`/`exp` 三种策略） |
| **KD Loss** | `models/kd_loss.py` | 跨模态知识蒸馏，支持 NCKD（非目标类蒸馏）与双向蒸馏 |
| **SMLI Loss** | `models/smli_loss.py` | 软匹配后期交互损失（patch 级多对多对齐） |
| **BCEWithLogits** | — | 多标签病理分类的基础监督损失 |

其中 AHNP 的难负样本加权与 KRC（Kendall 秩相关）样本筛选是本方法增强判别力、抑制噪声配对的关键设计。

### 训练范式

```
                      ┌─────────────────┐
   CXR ─► DenseNet ──►│  跨模态对比学习  │  AHNP + KD
                      │   (训练时对齐)   │
   ECG ─► Transformer►└────────┬────────┘
                               │
                               ▼  推理时仅保留 ECG 分支
                       ECG Embedding ─► MLP 分类头 ─► 病理预测
```

---

## 项目结构

```
my_cromotex_copy/
├── config/
│   ├── config.yaml                       # 主配置（Hydra）：各阶段超参数
│   └── cromotex/
│       └── cromotex_patch_transformer.yaml  # 模型结构超参数
├── src/cromotex/
│   ├── models/
│   │   ├── cromotex.py                    # CroMoTEX 主模型 / 预训练 / 微调封装
│   │   ├── CLKD.py                        # 对比学习 + 知识蒸馏模型
│   │   ├── image_encoder.py              # DenseNet-121 CXR 编码器
│   │   ├── timeseries_encoder.py        # ECG Patch Transformer
│   │   ├── ahnp_loss.py / kd_loss.py / smli_loss.py  # 损失函数
│   │   └── fusion.py
│   └── utils/                             # 指标、采样器、数据增强、预处理
├── data_provider/
│   └── data_loader.py                     # CXR_ECG_MatchedDataset（HDF5 读取）
├── preprocess/
│   └── prepare_dataset.py                 # MIMIC 数据匹配、清洗、HDF5 生成
├── train/                                 # 各阶段训练入口
│   ├── pretrain_img_classif.py           # ① 预训练 CXR 编码器
│   ├── pretrain_ecg_encoder.py           # ② 预训练 ECG 编码器（无监督对比）
│   ├── train_multimodal.py               # ③ 跨模态对比训练（CroMoTEX）
│   ├── train_CLKD.py                      # ③' 对比学习 + 知识蒸馏训练
│   └── finetune.py                        # ④ 仅 ECG 微调 + 分类头
├── test/
│   └── evaluate.py                        # 统一评估入口
└── environment.yaml                       # Conda 环境
```

---

## 环境安装

### 1. 创建 Conda 环境

```bash
# 基础依赖（来自 environment.yaml）
conda create -n cromotex python=3.12 -y
conda activate cromotex
conda install -y ipykernel tqdm pandas imageio matplotlib h5py
```

### 2. 安装深度学习与医学信号依赖

```bash
# PyTorch（请按你的 CUDA 版本从官网选择对应命令）
pip install torch torchvision

# 核心依赖
pip install hydra-core omegaconf mlflow rich scikit-learn scipy
pip install torchxrayvision wfdb h5py opencv-python pillow
```

### 3. 以可编辑模式安装本项目

项目源码位于 `src/cromotex`，训练脚本通过 `sys.path` 注入根目录导入。如需以包形式使用，可执行：

```bash
pip install -e .
```

> 说明：`environment.yaml` 中的 `name` 字段指向原作者路径，导入前可忽略或改为本地环境名。

---

## 数据准备

本项目基于 **MIMIC** 系列公开数据集，需自行申请 [PhysioNet](https://physionet.org/) 权限后下载：

| 数据集 | 用途 |
|--------|------|
| MIMIC-CXR-JPG | 胸片图像与 CheXpert 病理标签 |
| MIMIC-IV-ECG | 12 导联心电图原始信号 |
| MIMIC-IV-ECG-Ext-ICD | ECG 的 ICD-10 诊断标签 |
| MIMIC-IV-ED / Note | 急诊就诊时间窗、临床文本（用于配对与可选文本模态） |

### 预处理流程

`preprocess/prepare_dataset.py` 依次完成：CXR 与 ECG 按 `subject_id` + 就诊时间窗匹配 → 病理标签清洗 → 按患者划分 train/val/test（70%/10%/20%）→ 生成 HDF5。

```bash
python preprocess/prepare_dataset.py
```

执行后将在 `datasets/processed/` 下生成：

- `train_matched.h5` / `val_matched.h5` / `test_matched.h5`：配对的 CXR + ECG + 标签
- `pretrain_train.h5` / `pretrain_val.h5`：CXR 单模态预训练数据
- `pretrain_ecg_train.h5` / `pretrain_ecg_val.h5`：ECG 单模态预训练数据

**ECG 预处理步骤**：NaN 置零 → 导联顺序对齐 → 重采样至 100Hz → 基线漂移去除 → 逐导联归一化至 [-1, 1] → 固定长度 1000（10 秒）。
**CXR 预处理步骤**：转灰度 → 中心裁剪 → resize 至 224×224 → xrv 归一化。

> ⚠️ **路径配置**：`data_provider/data_loader.py` 中的 `DATASET_BASE_DIR` 为硬编码本地路径，请按实际存放位置修改

---

## 训练流程

完整训练分为四个阶段，**前两个单模态预训练阶段产出的权重是后续跨模态训练的输入**。所有脚本均基于 [Hydra](https://hydra.cc/) 管理配置，可在命令行直接覆盖任意超参数。

### 阶段 ① 预训练 CXR 编码器

```bash
python train/pretrain_img_classif.py
```

产出权重命名形如 `pretrain_img_last_['cardiomegaly', 'edema', 'pleural_effusion']_14.pth`，保存于 `checkpoints/`。

### 阶段 ② 预训练 ECG 编码器（无监督对比）

```bash
python train/pretrain_ecg_encoder.py
```

产出权重形如 `biot_pretrain_ecg_last_40.pth`。

### 阶段 ③ 跨模态对比训练

两种可选范式，二选一：

```bash
# 范式 A：CroMoTEX 原始跨模态对比（AHNP Loss）
python train/train_multimodal.py

# 范式 B：对比学习 + 知识蒸馏（本仓库扩展）
python train/train_CLKD.py
```

---


## 配置说明

主配置文件 `config/config.yaml` 按训练阶段分节，关键字段如下：

| 配置节 | 关键参数 | 含义 |
|--------|----------|------|
| 顶层 | `pathology` | 目标病理列表（决定分类头输出维度） |
| 顶层 | `img_pth_name` / `ecg_pth_name` | 加载的预训练权重标识 `[阶段, epoch]` |
| `cromotex_train` | `temperature` | 对比损失温度系数（默认 0.01） |
| `cromotex_train` | `hard_neg_weights` | 难负样本加权策略：`topk`/`linear`/`exp`/`none` |
| `cromotex_train` | `lambda_cross_contrast` | 跨模态对比损失权重 |
| `CLKD_train` | `loss_BCE` / `loss_KD` / `loss_AHNP` / `loss_SMLI` | 各损失项权重 |
| `CLKD_train` | `use_NCKD` / `use_krc` / `use_DGSF` | 蒸馏机制开关 |
| `CLKD_train` | `img_encoder_freeze` | 是否冻结教师（CXR）编码器 |
| `finetune` | `freeze_backbone` / `weighted_sampling` | 微调策略 |
| `*.optim` | `lr_peak` / `warmup_epochs` / `grad_clip` | 优化器与学习率调度（线性升 + 余弦降） |

模型结构超参数见 `config/cromotex/cromotex_patch_transformer.yaml`（patch 卷积核、embed 维度、注意力头数、投影维度等）。

---

## 引用

本项目复现并扩展自以下工作，如使用请引用原论文：

```bibtex
@article{cromotex2025,
  title   = {Contrastive Cross-Modal Learning for Infusing Chest X-ray Knowledge into ECGs},
  journal = {arXiv preprint arXiv:2506.19329},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.19329}
}
```

**相关资源：**

- 论文：[arXiv:2506.19329](https://arxiv.org/abs/2506.19329)
- 官方代码：[vineetpmoorty/CroMoTEX](https://github.com/vineetpmoorty/CroMoTEX)
- ECG 编码器参考：[svthapa/MoRE](https://github.com/svthapa/MoRE)、[torchxrayvision](https://github.com/mlmed/torchxrayvision)
- 有关 CLKD 的理论可参考：CLKD.pdf

> 本仓库为研究复现版本，CLKD（对比学习 + 知识蒸馏）相关模块为在原方法基础上的扩展实现，仅供学术研究使用，不构成任何临床诊断依据。
