# code layout
# import and configure model
# make the dataloader class
# perform data augmentations for cxr data
# define the training loop
# define the evaluation loop
# define the main function
# add logging and save checkpoints

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import numpy as np
import logging
from rich.logging import RichHandler
from rich.progress import track
import hydra
from omegaconf import DictConfig
import mlflow
import time
from tqdm import tqdm
import sys
# 1. 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 2. 获取当前文件所在目录（dir_A）
current_dir = os.path.dirname(current_file_path)
# 3. 获取上级目录（project）
parent_dir = os.path.dirname(current_dir)
# 4. 将上级目录添加到Python的搜索路径中
sys.path.append(parent_dir)
from src.cromotex.models.timeseries_encoder import ECGTimeseriesEncoder
import src.cromotex.utils.metrics as metrics
import src.cromotex.utils.utils as utils
from src.cromotex.utils.utils import lr_linear_rise_cosine_decay as lr_sched
from src.cromotex.utils.datasets import CXR_ECG_MatchedDataset
from src.cromotex.utils.balanced_sampler import create_balanced_sampler

import warnings

# 过滤掉特定的 UserWarning
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

def train_one_epoch(
    cfg,
    model,
    dataloader,
    criterion,
    optimizer,
    epoch,
    device
):
    model.train()
    running_loss = 0.0

    lr = lr_sched(cfg.pretrain_ecg_classif.optim, epoch)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.zero_grad()

    # for images, labels in track(dataloader, description=f"Epoch {epoch}"):
    for idx, (_, ecg, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        # if idx > 5:
        #     break #testing
        ecg, labels = ecg.to(device), labels.to(device)
        labels = labels.float()
        # embeds, logits = model(ecg)
        _, ts_logits = model(ecg)
        loss = criterion(ts_logits, labels)
        loss = loss / cfg.pretrain_ecg_classif.optim.grad_accum_steps

        loss.backward()

        if cfg.pretrain_ecg_classif.optim.grad_clip > 0.0:
            grad_clip = cfg.pretrain_ecg_classif.optim.grad_clip
            if cfg.pretrain_ecg_classif.data_parallel:
                torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), grad_clip
                )
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        running_loss += loss.item() * cfg.pretrain_ecg_classif.optim.grad_accum_steps

        if (idx + 1) % cfg.pretrain_ecg_classif.optim.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            mlflow.log_metric(
                "loss_batch", loss.item(), step=idx + len(dataloader) * epoch
            )
    
    loss_epoch = running_loss / len(dataloader)
    train_info = {}
    train_info['loss'] = loss_epoch
    train_info['epoch'] = epoch

    for param_group in optimizer.param_groups:
        train_info[f"lr_{param_group['name']}"] = param_group['lr']

    utils.log_train_info_to_mlflow(train_info)
    return train_info

def evaluate(cfg, model, dataloader, criterion, epoch, device):
    model.eval()
    running_loss = 0.0

    correct_predictions = 0
    total_samples = 0
    all_labels = [] #batch ground truths
    all_preds = [] #batch predictions
    all_pred_probs = [] #batch prediction probabilities

    with torch.no_grad():
        for idx, (_, ecg, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
            # if idx > 5:
            #     break #testing
            ecg = ecg.to(device)
            labels = labels.to(device)
            labels = labels.float()

            # embeds, logits = model(ecg)
            _, ts_logits = model(ecg)

            loss = criterion(ts_logits, labels)
            running_loss += loss.item()

            preds = (torch.sigmoid(ts_logits) > 0.5).float()
            probs = torch.sigmoid(ts_logits)
    
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
    # class_report = classification_report(all_labels.cpu(), all_preds.cpu(), output_dict=True)
    # f1_score = [
    #     class_report[f'{label_idx}']['f1-score']
    #     for label_idx in range(all_labels.shape[1])
    # ]

    val_info = {}
    val_info['loss'] = loss_epoch
    val_info['accuracy'] = accuracy
    val_info['auroc'] = auroc_scores
    val_info['auprc'] = auprc_scores
    # val_info['f1'] = f1_score

    metrics.log_precision_recall_curves_to_mlflow(
        all_labels, all_pred_probs, epoch
    )
    
    mlflow.log_metric("loss_val", loss_epoch, step=epoch)
    mlflow.log_metric(f"accuracy_val", accuracy, step=epoch)

    for label_idx in range(all_labels.shape[1]):
        mlflow.log_metric(
            f"auroc_val_{label_idx}", auroc_scores[label_idx], step=epoch
        )
        mlflow.log_metric(
            f"auprc_val_{label_idx}", auprc_scores[label_idx], step=epoch
        )
        # mlflow.log_metric(
        #     f"f1_val_{label_idx}", f1_score[label_idx], step=epoch
        # )
    return val_info

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config"
)
def main(cfg: DictConfig) -> None:

    np.random.seed(cfg.pretrain_ecg_classif.seed)
    torch.manual_seed(cfg.pretrain_ecg_classif.seed)
    torch.cuda.manual_seed(cfg.pretrain_ecg_classif.seed)
    torch.cuda.manual_seed_all(cfg.pretrain_ecg_classif.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if cfg.pretrain_ecg_classif.data_parallel:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        device = torch.device("cuda")
    else:
        device = torch.device("cuda", cfg.pretrain_ecg_classif.gpu_id)

    mlflow.set_tracking_uri(hydra.utils.to_absolute_path('mlruns'))
    mlflow.set_experiment(cfg.pretrain_ecg_classif.mlflow_expt_name)
    experiment = mlflow.get_experiment_by_name(
        cfg.pretrain_ecg_classif.mlflow_expt_name
    )

    logger = logging.getLogger("mlflow")
    logger.handlers = []
    logger.setLevel(logging.INFO)
    rich_handler = RichHandler(
        show_level=False,
        show_time=True,
        show_path=False,
        markup=True
    )
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%H:%M:%S]"
    )
    rich_handler.setFormatter(formatter)
    logger.addHandler(rich_handler)
    
    model = ECGTimeseriesEncoder(cfg)

    if cfg.pretrain_ecg_classif.data_parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    m = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    from src.cromotex.utils.ts_augmentations import ECGAugmentor
    ts_augmentor = ECGAugmentor()
    
    for param in m.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {'params': m.timeseries_encoder.parameters(), 'lr': cfg.pretrain_ecg_classif.optim.lr_peak, 'name': 'timeseries_encoder'},
        {'params': m.classif_head.parameters(), 'lr': cfg.pretrain_ecg_classif.optim.lr_peak, 'name': 'classif_head'}
    ], weight_decay=cfg.pretrain_ecg_classif.optim.weight_decay)

    train_data = CXR_ECG_MatchedDataset(
        cfg,
        'processed/train_matched.h5',
        None, ts_augmentor
    )

    val_data = CXR_ECG_MatchedDataset(
        cfg,
        'processed/test_matched.h5',
        None, None
    )

    train_labels = train_data.get_labels()

    if cfg.pretrain_ecg_classif.weighted_sampling is True:
        train_sampler = create_balanced_sampler(
            cfg.pretrain_ecg_classif, train_labels, pos_neg_ratio=cfg.pretrain_ecg_classif.pos_neg_ratio
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.pretrain_ecg_classif.batch_size,
        sampler=train_sampler,
        shuffle=shuffle,
        num_workers=cfg.pretrain_ecg_classif.num_dataloader_workers,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=cfg.pretrain_ecg_classif.batch_size,
        shuffle=False,
        num_workers=cfg.pretrain_ecg_classif.num_dataloader_workers,
        pin_memory=True
    )

    criterion = nn.BCEWithLogitsLoss()

    start_epoch = 0
    if cfg.pretrain_ecg_classif.resume_from_last_ckpt:
        checkpoint_data = utils.load_train_checkpoint(
            cfg.pretrain_ecg_classif.ckpt_filename, model, optimizer
        )
        model, optimizer, last_epoch, mlflow_run_id = checkpoint_data
        start_epoch = last_epoch + 1
        mlflow.start_run(
            run_id=mlflow_run_id, experiment_id=experiment.experiment_id
        )
        logger.info(f"Loaded checkpoint @ epoch {start_epoch}")
    else:
        mlflow.start_run(experiment_id=experiment.experiment_id)

    if isinstance(model, torch.nn.DataParallel):
        total_params = sum(
            p.numel() for p in model.module.parameters() if p.requires_grad
        )
    else:
        total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
    logger.info(f"Model size: {(total_params/1e6):.2f}M parameters")

    current_run_id = mlflow.active_run().info.run_id
    current_run_name = mlflow.active_run().info.run_name
    logger.info(f"Starting from epoch {start_epoch}")
    logger.info(f"mlflow run name: [bold red]{current_run_name}")
    mlflow.log_params(utils.format_cfg(cfg))

    # best_val_loss = float('inf')
    # best_auroc = 0.0
    # best_prauc = 0.0
    train_infos = []
    val_infos = []
    for epoch in range(start_epoch, start_epoch + cfg.pretrain_ecg_classif.num_epochs):
        start_time = time.time()

        train_info = train_one_epoch(
            cfg,
            model,
            train_dataloader,
            criterion,
            optimizer,
            epoch,
            device
        )
        # train_info = {'epoch': epoch, 'loss': 0} #Eval only
        val_info = evaluate(
            cfg,
            model,
            val_dataloader,
            criterion,
            epoch,
            device
        )

        train_infos.append(train_info)
        val_infos.append(val_info)

        epoch_time = time.time() - start_time
        logger.info(utils.log_epoch_metrics(train_info, val_info, epoch_time))
        
        # # Save best checkpoint
        # if val_info['loss'] < best_val_loss:
        #     best_val_loss = val_info['loss']
        #     if cfg.pretrain_ecg_classif.save_ckpt:
        #         fname = f'cromotex_finetuned_originECG_best_loss_{cfg.pathology}_weighted_{cfg.pretrain_ecg_classif.pos_neg_ratio}.pth'
        #         utils.save_checkpoint(
        #             fname,
        #             model,
        #             optimizer,
        #             epoch,
        #             current_run_id
        #         )
        # if val_info['auroc'][list(val_info['auroc'].keys())[0]] > best_auroc:
        #     best_auroc = val_info['auroc'][list(val_info['auroc'].keys())[0]]
        #     if cfg.pretrain_ecg_classif.save_ckpt:
        #         fname = f'cromotex_finetuned_best_auroc_{cfg.pathology}'
        #         fname += f'_{ckpt_run_name}.pth'
        #         utils.save_checkpoint(
        #             fname,
        #             model,
        #             optimizer,
        #             epoch,
        #             current_run_id
        #         )
        # if val_info['auprc'][list(val_info['auprc'].keys())[0]] > best_prauc:
        #     best_prauc = val_info['auprc'][list(val_info['auprc'].keys())[0]]
        #     if cfg.pretrain_ecg_classif.save_ckpt:
        #         fname = f'cromolts_finetuned_best_prauc_{cfg.pathology}'
        #         fname += f'_{ckpt_run_name}.pth'
        #         utils.save_checkpoint(
        #             fname,
        #             model,
        #             optimizer,
        #             epoch,
        #             current_run_id
        #         )
        # if val_info['f1'] > best_f1:
        #     best_f1 = val_info['f1']
        #     if cfg.pretrain_ecg_classif.save_ckpt:
        #         fname = f'cromolts_finetuned_best_f1_{cfg.pathology}'
        #         fname += f'_{ckpt_run_name}.pth'
        #         utils.save_checkpoint(
        #             fname,
        #             model,
        #             optimizer,
        #             epoch,
        #             current_run_id
        #         )
        # Save last checkpoint    
        if cfg.pretrain_ecg_classif.save_ckpt:
            fname = f'pretrain_ecg_classif_classif_{cfg.pathology}_{epoch}.pth'
            utils.save_checkpoint(
                fname,
                model,
                optimizer,
                epoch,
                current_run_id
            )
        if cfg.pretrain_ecg_classif.early_stop:
            if utils.early_stop(
                train_infos, val_infos, patience=3
            ):
                logger.info("Early stopping critera met")
                break

    mlflow.end_run()

if __name__ == '__main__':
    main()