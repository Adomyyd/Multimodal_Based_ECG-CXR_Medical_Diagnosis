import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
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
from src.cromotex.models.CLKD import CLKD
from src.cromotex.models.ahnp_loss import AHNPLoss, AHNPLosswithKRC
from src.cromotex.models.kd_loss import KDLoss
from src.cromotex.models.smli_loss import SMLILoss
import src.cromotex.utils.metrics as metrics
import src.cromotex.utils.utils as utils
from src.cromotex.utils.utils import lr_linear_rise_cosine_decay as lr_sched
# from src.cromotex.utils.datasets import CXR_ECG_MatchedDataset
from data_provider.data_loader import CXR_ECG_MatchedDataset

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

MODEL_NAME = "CLKD"

def train_one_epoch(
    cfg,
    model,
    dataloader,
    criterion1,
    criterion2,
    criterion3,
    optimizer,
    epoch,
    device,
    **kwargs
):
    model.train()
    running_loss = 0.0

    lr = lr_sched(cfg.CLKD_train.optim, epoch)
    
    optimizer = model.set_lr(cfg, optimizer, lr)

    optimizer.zero_grad()

    for idx, (img, ecg, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Training"):
        img, ecg, labels = img.to(device), ecg.to(device), labels.to(device)
        
        img_proj, ts_proj, img_logits, ts_logits, img_patches, ts_patches = model(img, ecg)
        labels = labels.float()

        lambda1 = cfg.CLKD_train.loss_AHNP
        if criterion1 is not None:
            loss1 = criterion1(epoch, img_proj, ts_proj, ts_logits, labels, True) if not cfg.CLKD_train.use_krc else criterion1(img_proj, ts_proj, img_logits, ts_logits, labels, True)
        else:
            loss1 = 0
        lambda2 = cfg.CLKD_train.loss_BCE
        loss2 = criterion2(ts_logits, labels)
        if criterion3 is not None:
            if cfg.CLKD_train.loss_AHNP > 0. and cfg.CLKD_train.loss_SMLI > 0.:
                lambda3 = cfg.CLKD_train.loss_SMLI
                loss3 = criterion3(img_patches, ts_patches, img_logits, ts_logits)
            elif cfg.CLKD_train.loss_KD > 0.:
                lambda3 = cfg.CLKD_train.loss_KD
                loss3 = criterion3(img_logits, ts_logits, labels)
        else:
            lambda3 = 0
            loss3 = 0

        if cfg.CLKD_train.img_encoder_freeze:
            loss = lambda1 *loss1 + lambda2 * loss2 + lambda3 * loss3
        else:
            loss4 = criterion2(img_logits, labels)
            loss = lambda1 *loss1 + lambda2 * (loss2 + loss4) + lambda3 * loss3

        loss = loss / cfg.CLKD_train.optim.grad_accum_steps
        # print(f"loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}, loss3: {loss3.item():.4f}, total_loss: {loss.item():.4f}")
        loss.backward()
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"Gradients for {name}: {param.grad.norm().item():.4f}")
        #     else:
        #         print(f"No gradients for {name}")

        if cfg.CLKD_train.optim.grad_clip > 0.0:
            grad_clip = cfg.CLKD_train.optim.grad_clip
            if cfg.CLKD_train.data_parallel:
                torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), grad_clip
                )
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        running_loss += loss.item() * cfg.CLKD_train.optim.grad_accum_steps
        
        if (idx + 1) % cfg.CLKD_train.optim.grad_accum_steps == 0:
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

def evaluate(cfg, model, dataloader, criterion1, criterion2, criterion3, epoch, device):
    model.eval()
    running_loss = 0.0
    
    correct_predictions = 0
    total_samples = 0
    all_labels = [] #batch ground truths
    all_preds = [] #batch predictions
    all_pred_probs = [] #batch prediction probabilities

    with torch.no_grad():
        for idx, (img, ecg, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing"):
            img, ecg = img.to(device), ecg.to(device)
            labels = labels.to(device)

            img_proj, ts_proj, img_logits, ts_logits, img_patches, ts_patches = model(img, ecg)
            labels = labels.float()

            lambda1 = cfg.CLKD_train.loss_AHNP
            if criterion1 is not None:
                loss1 = criterion1(epoch, img_proj, ts_proj, ts_logits, labels, True) if not cfg.CLKD_train.use_krc else criterion1(img_proj, ts_proj, img_logits, ts_logits, labels, True)
            else:
                loss1 = 0
            lambda2 = cfg.CLKD_train.loss_BCE
            loss2 = criterion2(ts_logits, labels)
            if criterion3 is not None:
                if cfg.CLKD_train.loss_AHNP > 0. and cfg.CLKD_train.loss_SMLI > 0.:
                    lambda3 = cfg.CLKD_train.loss_SMLI
                    loss3 = criterion3(img_patches, ts_patches, img_logits, ts_logits)
                elif cfg.CLKD_train.loss_KD > 0.:
                    lambda3 = cfg.CLKD_train.loss_KD
                    loss3 = criterion3(img_logits, ts_logits, labels)
            else:
                lambda3 = 0
                loss3 = 0

            if cfg.CLKD_train.img_encoder_freeze:
                loss = lambda1 *loss1 + lambda2 * loss2 + lambda3 * loss3
            else:
                loss4 = criterion2(img_logits, labels)
                loss = lambda1 *loss1 + lambda2 * (loss2 + loss4) + lambda3 * loss3

            running_loss += loss.item()
            preds = (torch.sigmoid(ts_logits) > 0.5).float()
            probs = torch.sigmoid(ts_logits)
    
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.numel()

            all_pred_probs.append(probs)
            all_labels.append(labels)
            all_preds.append(preds)
    
    # accuracy = correct_predictions / total_samples
    loss_epoch = running_loss / len(dataloader)

    all_labels = torch.cat(all_labels, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    all_pred_probs = torch.cat(all_pred_probs, dim=0)

    auroc_scores = metrics.auroc(all_labels, all_pred_probs)
    auprc_scores = metrics.auprc(all_labels, all_pred_probs)

    val_info = {}
    val_info['loss'] = loss_epoch
    # val_info['accuracy'] = accuracy
    val_info['auroc'] = auroc_scores
    val_info['auprc'] = auprc_scores

    mlflow.log_metric("loss_val", loss_epoch, step=epoch)
    # mlflow.log_metric("accuracy_val", accuracy, step=epoch)    
    for label_idx in range(all_labels.shape[1]):
        mlflow.log_metric(
            f"auroc_val_{label_idx}", auroc_scores[label_idx], step=epoch
        )
        mlflow.log_metric(
            f"auprc_val_{label_idx}", auprc_scores[label_idx], step=epoch
        )
    return val_info

@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="config"
)
def main(cfg: DictConfig) -> None:

    set_seed(cfg.CLKD_train.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if cfg.CLKD_train.data_parallel:
        os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
        device = torch.device("cuda")
    else:
        device = torch.device("cuda", cfg.CLKD_train.gpu_id)

    mlflow.set_tracking_uri(hydra.utils.to_absolute_path('mlruns'))
    mlflow.set_experiment(cfg.CLKD_train.mlflow_expt_name)
    experiment = mlflow.get_experiment_by_name(
        cfg.CLKD_train.mlflow_expt_name
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
    
    model = CLKD(cfg)
    if cfg.CLKD_train.data_parallel:
        model = torch.nn.DataParallel(model)
    model.to(device)

    if isinstance(model, torch.nn.DataParallel):
        img_augs_train, img_augs_val, ts_augmentor = (
            model.module.get_augmentations()
        )
    else:
        img_augs_train, img_augs_val, ts_augmentor = model.get_augmentations()

    train_data = CXR_ECG_MatchedDataset(
        cfg,
        'train_matched.h5',
        img_augs_train, ts_augmentor
    )

    val_data = CXR_ECG_MatchedDataset(
        cfg,
        'test_matched.h5',
        img_augs_val, None
    )

    subnum = None
    if subnum is not None:
        train_data = Subset(train_data, range(subnum))
        val_data = Subset(val_data, range(subnum))

    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg.CLKD_train.batch_size,
        sampler=None,
        shuffle=True,
        num_workers=cfg.CLKD_train.num_dataloader_workers,
        #prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=cfg.CLKD_train.batch_size,
        shuffle=False,
        num_workers=cfg.CLKD_train.num_dataloader_workers,
        #prefetch_factor=2,
        pin_memory=True,
        persistent_workers=True
    )
    logger.info(f"Train dataset size: {len(train_data)}")
    logger.info(f"Test dataset size: {len(val_data)}")

    if cfg.CLKD_train.loss_AHNP > 0.:
        criterion1 = AHNPLoss(cfg) if not cfg.CLKD_train.use_krc else AHNPLosswithKRC(cfg)
    else:
        criterion1 = None
    criterion2 = nn.BCEWithLogitsLoss()
    if cfg.CLKD_train.loss_AHNP > 0. and cfg.CLKD_train.loss_SMLI > 0.:
        criterion3 = SMLILoss(cfg)
    elif cfg.CLKD_train.loss_KD > 0.:
        criterion3 = KDLoss(cfg)
    else:
        criterion3 = None

    if cfg.CLKD_train.img_encoder_freeze:
        if isinstance(model, torch.nn.DataParallel):
            for param in model.module.image_encoder.parameters():
                param.requires_grad = False
        else:
            for param in model.image_encoder.parameters():
                param.requires_grad = False
        print("Freezing image encoder")

    log_module_info(cfg) # 输出改动模块的信息

    if isinstance(model, torch.nn.DataParallel):
        optimizer = model.module.get_optimizer(cfg, model)
    else:
        optimizer = model.get_optimizer(cfg, model)

    start_epoch = 0
    if cfg.CLKD_train.resume_from_last_ckpt:
        filepath = os.path.join(
            hydra.utils.to_absolute_path('checkpoints'),
            f'CLKD_KDlogits_NCKD_KRC_last_1.pth'
        )
        checkpoint = torch.load(filepath, map_location='cpu')
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        # if optimizer is not None:
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        # checkpoint_data = utils.load_train_checkpoint(
        #     f'{MODEL_NAME}_last_{cfg.pathology}_0.1,1,0_8.pth', model, optimizer
        # )
        # model, optimizer, last_epoch, mlflow_run_id = checkpoint_data
        start_epoch = last_epoch + 1
        # mlflow.start_run(
        #    run_id=mlflow_run_id, experiment_id=experiment.experiment_id
        # )
        tags = {'mlflow.note.content': cfg.CLKD_train.mlflow_run_notes}
        mlflow.start_run(experiment_id=experiment.experiment_id, tags=tags)
        logger.info(f"Loaded checkpoint @ epoch {start_epoch}")
    else:
        tags = {'mlflow.note.content': cfg.CLKD_train.mlflow_run_notes}
        mlflow.start_run(experiment_id=experiment.experiment_id, tags=tags)
    

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

    train_infos = []
    val_infos = []
    for epoch in range(start_epoch, start_epoch + cfg.CLKD_train.num_epochs):
        start_time = time.time()

        train_info = train_one_epoch(
            cfg,
            model,
            train_dataloader,
            criterion1,
            criterion2,
            criterion3,
            optimizer,
            epoch,
            device
        )
        # train_info = {'epoch': epoch, 'loss': 0} #Eval only
        val_info = evaluate(
            cfg,
            model,
            val_dataloader,
            criterion1,
            criterion2,
            criterion3,
            epoch,
            device
        )
        # exit(0)

        train_infos.append(train_info)
        val_infos.append(val_info)

        epoch_time = time.time() - start_time
        logger.info(utils.log_epoch_metrics(train_info, val_info, epoch_time))
        utils.save_log(utils.log_epoch_metrics(train_info, val_info, epoch_time), f'{MODEL_NAME}.txt')

        run_name = mlflow.active_run().data.tags.get('mlflow.runName')

        # Save last checkpoint    
        if cfg.CLKD_train.save_ckpt:
            utils.save_checkpoint(
                f'{MODEL_NAME}_last_{epoch}.pth',
                model,
                optimizer,
                epoch,
                current_run_id
            )
        if cfg.CLKD_train.early_stop:
            if utils.early_stop(
                train_infos, val_infos, patience=3
            ):
                logger.info("Early stopping critera met")
                break

    mlflow.end_run()

def log_module_info(cfg):
    global MODEL_NAME
    if cfg.CLKD_train.loss_AHNP > 0.:
        MODEL_NAME = f"{MODEL_NAME}_AHNP"
        print("Using CMKD_FCL")
        if cfg.CLKD_train.use_krc:
            MODEL_NAME = f"{MODEL_NAME}_KRC"
            print("Using KRC")
        if cfg.CLKD_train.loss_SMLI > 0.:
            MODEL_NAME = f"{MODEL_NAME}_SMLI"
            print("Using SMLI")
        if cfg.CLKD_train.use_DGSF:
            MODEL_NAME = f"{MODEL_NAME}_DGSF"
            print("Using DGSF")

    elif cfg.CLKD_train.loss_KD > 0.:
        MODEL_NAME = f"{MODEL_NAME}_KDlogits"
        print("Using CMKD_ML")
        if cfg.CLKD_train.use_NCKD:
            MODEL_NAME = f"{MODEL_NAME}_NCKD"
            print("Using NCKD")
        if cfg.CLKD_train.use_krc:
            MODEL_NAME = f"{MODEL_NAME}_KRC"
            print("Using KRC")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    main()