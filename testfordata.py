import torch
import os
import numpy as np
import torch.optim as optim
from omegaconf import DictConfig, ListConfig
from src.cromotex.utils.datasets import CXRPretrainDataset
from src.cromotex.models.image_encoder import get_image_encoder
from torch.utils.data import DataLoader, Subset
import hydra
import src.cromotex.utils.utils as utils
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import src.cromotex.utils.metrics as metrics
import src.cromotex.utils.utils as utils
from src.cromotex.utils.utils import lr_linear_rise_cosine_decay as lr_sched

def evaluate(cfg, model, dataloader, criterion, epoch, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_labels = [] #batch ground truths
    all_preds = [] #batch predictions
    all_pred_probs = [] #batch prediction probabilities

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).float()
            probs = torch.sigmoid(outputs)
    
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.numel()

            all_pred_probs.append(probs)
            all_labels.append(labels)
            all_preds.append(preds)
    
    accuracy = correct_predictions / total_samples
    loss_epoch = running_loss / len(dataloader)

    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_pred_probs = torch.cat(all_pred_probs, dim=0)

    auroc_scores = metrics.auroc(all_labels, all_pred_probs)
    auprc_scores = metrics.auprc(all_labels, all_pred_probs)

    val_info = {}
    val_info['loss'] = loss_epoch
    val_info['accuracy'] = accuracy
    val_info['auroc'] = auroc_scores
    val_info['auprc'] = auprc_scores
    
    print("loss_val", loss_epoch)
    print("accuracy_val", accuracy)
    for label_idx in range(all_labels.shape[1]):
        print(
            f"auroc_val_{label_idx}: {auroc_scores[label_idx]}"
        )
        print(
            f"auprc_val_{label_idx}: {auprc_scores[label_idx]}"
        )
    return val_info

@hydra.main(version_base=None, config_path="./config", config_name="config")
def main(cfg: DictConfig):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    model = get_image_encoder(cfg)
    model.to(device)

    # =========================
    # 数据集加载
    # =========================
    print('Loading data... \n')


    train_augmentations, val_augmentations = model.get_augmentations(cfg)

    train_data = CXRPretrainDataset(
        cfg,
        'processed/train_matched.h5',
        augmentations=train_augmentations
    )

    val_data = CXRPretrainDataset(
        cfg,
        'processed/test_matched.h5',
        augmentations=val_augmentations
    )

    if cfg.pretrain_img.weighted_sampling:
        sampler = utils.get_weighted_sampler(train_data.get_labels())
        shuffle = False
    else:
        shuffle = True
        sampler = None

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.pretrain_img.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.pretrain_img.num_dataloader_workers
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=cfg.pretrain_img.batch_size,
        shuffle=False,
        num_workers=cfg.pretrain_img.num_dataloader_workers
    )

    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.Adam([
        {
            'params': model.densenet.features.parameters(),
            'lr': (
                cfg.pretrain_img.optim.lr_peak * 
                cfg.pretrain_img.optim.backbone_lr_scaler
            ),
            'weight_decay': cfg.pretrain_img.optim.weight_decay,
            'name': 'backbone'
        },
        {
            'params': model.densenet.classifier.parameters(),
            'lr': cfg.pretrain_img.optim.lr_peak,
            'weight_decay': cfg.pretrain_img.optim.weight_decay,
            'name': 'classifier'
        }
    ])

    start_epoch = 0
    if True:
        checkpoint_data = utils.load_train_checkpoint(
            f'pretrain_img_last_{cfg.pathology}_14.pth', model, optimizer
        )
        model, optimizer, last_epoch, _ = checkpoint_data
        start_epoch = last_epoch + 1
        print(f"Loaded checkpoint @ epoch 14")
    else:
        pass

    total_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Model size: {(total_params/1e6):.2f}M parameters")

    print(f"Starting from epoch {start_epoch}")
    train_infos = []
    val_infos = []

    val_info = evaluate(
            cfg,
            model,
            val_dataloader,
            criterion,
            start_epoch,
            device
        )

    print(f"val_info: {val_info}")
        


if __name__ == '__main__':
    main()