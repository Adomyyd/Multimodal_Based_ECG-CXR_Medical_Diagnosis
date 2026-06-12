import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

# Path setup
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import src.cromotex.utils.metrics as metrics
import src.cromotex.utils.utils as utils
from src.cromotex.utils.datasets import CXR_ECG_MatchedDataset
from src.cromotex.utils.cm_utils import find_optimal_threshold_and_plot_cm

def evaluate_model(cfg, model, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = []
    all_preds = []
    all_pred_probs = []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
            if cfg.model_type == 'ecg_only' or cfg.model_type == 'cromotexfinetune':
                _, ecg, labels = batch
                inputs = ecg.to(device)
            elif cfg.model_type == 'cxr_classif':
                cxr, _, labels = batch
                inputs = cxr.to(device)
            elif cfg.model_type == 'CLKD':
                _, ecg, labels = batch
                inputs = ecg.to(device)

            labels = labels.to(device).float()
            
            if cfg.model_type == 'ecg_only' or cfg.model_type == 'cromotexfinetune':
                ts_logits, _ = model(inputs)
            elif cfg.model_type == 'cxr_classif':
                ts_logits = model(inputs)
            elif cfg.model_type == 'CLKD':
                ts_logits = model(None, inputs, True)

            preds = (torch.sigmoid(ts_logits) > 0.5).float()
            probs = torch.sigmoid(ts_logits)
    
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.numel()

            all_pred_probs.append(probs)
            all_labels.append(labels)
            all_preds.append(preds)

    accuracy = correct_predictions / total_samples

    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_pred_probs = torch.cat(all_pred_probs, dim=0)

    auroc_scores = metrics.auroc(all_labels, all_pred_probs)
    auprc_scores = metrics.auprc(all_labels, all_pred_probs)
    
    # 自动寻找最优阈值并绘制混淆矩阵热力图
    labels_np = all_labels.cpu().numpy()
    probs_np = all_pred_probs.cpu().numpy()
    num_classes = labels_np.shape[1]
    
    # Try getting class names from configs
    if getattr(cfg, 'pathology', None):
        class_names = cfg.pathology
        if isinstance(class_names, str):
            class_names = [class_names]
    else:
        class_names = [f"Class_{i}" for i in range(num_classes)]
        
    cm = find_optimal_threshold_and_plot_cm(labels_np, probs_np, num_classes, class_names, model_name=cfg.model_name)
    print(cm)

    # 根据cm计算每个类别的F1-score和Recall
    f1_scores, recall_scores = [], []
    for i in range(num_classes):
        tp = cm[i][1, 1]
        fp = cm[i][0, 1]
        fn = cm[i][1, 0]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        f1_scores.append(f1)
        recall_scores.append(recall)

    val_info = {
        'accuracy': accuracy,
        'auroc': auroc_scores,
        'auprc': auprc_scores,
        'f1': f1_scores,
        'recall': recall_scores
    }

    print(f"Test Accuracy: {val_info['accuracy']:.4f}")
    for i in range(num_classes):
        class_name = class_names[i] if (class_names and i < len(class_names)) else f"Class_{i}"
        print(f"Class {class_name} AUROC: {val_info['auroc'][i]:.4f}")
        print(f"Class {class_name} AUPRC: {val_info['auprc'][i]:.4f}")
        # print(f"Class {class_name} F1-score: {val_info['f1'][i]:.4f}")
        # print(f"Class {class_name} Recall: {val_info['recall'][i]:.4f}")

    return val_info

def main():
    parser = argparse.ArgumentParser(description="Evaluate capability of different models")
    parser.add_argument("--model", type=str, required=True, choices=["ecg_only", "cromotexfinetune", "cxr_classif", "CLKD"], 
                        help="Choose the model to evaluate (ecg_only, cromotexfinetune, cxr_classif)")
    parser.add_argument("--isWeightedSample", type=int, required=True, choices=[1, 0],
                        help="Whether to use weighted sampling for training")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to the model checkpoint for evaluation")
    # Use parse_known_args to split standard args from hydra args
    args, unknown_args = parser.parse_known_args()

    # Pass remaining overrides directly to Hydras
    hydra.initialize(version_base=None, config_path="../config")
    cfg = hydra.compose(config_name="config", overrides=unknown_args)
    OmegaConf.set_struct(cfg, False)
    cfg.model_type = args.model
    cfg.model_name = args.model + ("_weightedsp" if args.isWeightedSample else "")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "cromotexfinetune":
        from src.cromotex.models.cromotex import CroMoTEXFinetune
        logger = logging.getLogger(f"eval for {args.model}")
        print(f"eval for {cfg.model_name}")
        model = CroMoTEXFinetune(cfg, logger)
        val_data = CXR_ECG_MatchedDataset(cfg, 'processed/test_matched.h5', None, None)
        batch_size = cfg.finetune.batch_size
    
    elif args.model == "ecg_only":
        from src.cromotex.models.cromotex import CroMoTEXFinetune
        logger = logging.getLogger(f"eval for {args.model}")
        print(f"eval for {cfg.model_name}")
        model = CroMoTEXFinetune(cfg, logger)
        val_data = CXR_ECG_MatchedDataset(cfg, 'processed/test_matched.h5', None, None)
        batch_size = cfg.finetune.batch_size

    elif args.model == "cxr_classif":
        from src.cromotex.models.image_encoder import get_image_encoder
        print(f"eval for {cfg.model_name}")
        model = get_image_encoder(cfg) # Adjust as per exact instantiation
        _, val_augmentations = model.get_augmentations(cfg)
        val_data = CXR_ECG_MatchedDataset(cfg, 'processed/test_matched.h5', val_augmentations, None)
        batch_size = cfg.pretrain_img.batch_size

    elif args.model == "CLKD":
        from src.cromotex.models.CLKD import CLKD
        logger = logging.getLogger(f"eval for {args.model}")
        print(f"eval for {cfg.model_name}")
        model = CLKD(cfg)
        val_data = CXR_ECG_MatchedDataset(cfg, 'processed/test_matched.h5', None, None)
        batch_size = cfg.CLKD_train.batch_size

    model.to(device)
    # 载入模型权重
    if args.checkpoint:
        checkpoint_path = os.path.join(
            hydra.utils.to_absolute_path('checkpoints'),
            args.checkpoint
        )
        if not os.path.exists(checkpoint_path):
            checkpoint_path = args.checkpoint

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
        else:
            print(f"Checkpoint not found at {checkpoint_path}. Evaluating with uninitialized model.")

    model.eval()

    # val_data = Subset(val_data, range(100))

    val_dataloader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    evaluate_model(cfg, model, val_dataloader, device)

if __name__ == "__main__":
    main()
