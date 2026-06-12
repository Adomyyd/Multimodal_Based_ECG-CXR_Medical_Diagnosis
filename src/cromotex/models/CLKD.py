import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import hydra
from src.cromotex.models.image_encoder import CXRImageEncoder
from src.cromotex.models.timeseries_encoder import ECGPatchTransformer
from src.cromotex.utils.ts_augmentations import ECGAugmentor, VCG_ECGAugmentor
from src.cromotex.models.cromotex import MLPClassifHead

class CLKD(nn.Module):
    """
    CLKD: Contrastive Learning and Knowledge Distillation based Modality Enhancement
    - Computes contrastive features exactly like CroMoTEXPatchTransformer.
    - Also outputs modality specific classification logits (img_logits, ts_logits) for KD loss.
    """
    def __init__(self, cfg):
        super(CLKD, self).__init__()
        self.cfg = cfg
        
        # Load the complete image encoder with its classifier
        checkpoint_filename = f"pretrain_img_{cfg.img_pth_name[0]}_{cfg.pathology}_{cfg.img_pth_name[1]}.pth"
        
        self.image_encoder = CXRImageEncoder(cfg, densenetCheckpoint=checkpoint_filename)
        
        self.timeseries_encoder = ECGPatchTransformer(cfg)

        # Load pretrained ECG weights
        filepath = os.path.join(
            hydra.utils.to_absolute_path('checkpoints'),
            f'biot_pretrain_ecg_{cfg.ecg_pth_name[0]}_{cfg.ecg_pth_name[1]}.pth'
        )
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location='cpu')
            print(f"Loaded pretrained timeseries encoder from {filepath}")
            ts_encoder_state_dict = {
                k.replace('timeseries_encoder.', ''): v
                for k, v in checkpoint['model_state_dict'].items()
                if k.startswith('timeseries_encoder')
            }
            self.timeseries_encoder.load_state_dict(ts_encoder_state_dict, strict=True)
        else:
            print(f"Warning: Pretrained timeseries encoder not found at {filepath}")
        
        self.img_feature_dim = self.image_encoder.feature_dim

        self.img_proj_linear = nn.Linear(
            self.img_feature_dim, cfg.cromotex.proj_dim
        )
        self.ts_proj_linear = nn.Linear(
            self.timeseries_encoder.classifier.in_features,
            cfg.cromotex.proj_dim
        )
        self.img_patches_proj_linear = nn.Linear(
            self.img_feature_dim, cfg.cromotex.proj_dim
        )
        self.ts_patches_proj_linear = nn.Linear(
            self.timeseries_encoder.classifier.in_features,
            cfg.cromotex.proj_dim
        )

        self.ts_augmentor = ECGAugmentor()

        img_augs = self.image_encoder.get_augmentations(cfg)
        self.img_augs_train, self.img_augs_val = img_augs

        self.ts_classif_head = MLPClassifHead(cfg)

    def forward(self, images, ts, evaluate=False):
        ts_embeds, _ , _, ts_patches = self.timeseries_encoder(ts)
        ts_embeds = F.normalize(ts_embeds, dim=-1)
        ts_logits = self.ts_classif_head(ts_embeds)
        
        # For KD loss, we keep unnormalized embeddings for the projection later if finetune is false

        if evaluate:
            return ts_logits

        ts_proj = self.ts_proj_linear(ts_embeds)
        ts_proj = F.normalize(ts_proj, dim=-1)

        batch_size = images.size(0)

        # Get img embeddings and logits
        images = images.mean(1).unsqueeze(1)
        img_feats = self.image_encoder.image_encoder(images)
        B, C, H, W = img_feats.shape
        img_patches = img_feats.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, num_patches, feature_dim]
        
        # pooling
        img_embeds = F.relu(img_feats, inplace=True)
        img_embeds = F.adaptive_avg_pool2d(
            img_embeds, (1, 1)
        ).view(batch_size, -1)
        
        # classification
        img_logits = self.image_encoder.classifier(img_embeds)
        img_logits = img_logits[:, self.image_encoder.pathology_indices]
        
        # contrastive projection
        img_proj = self.img_proj_linear(img_embeds)
        img_proj = F.normalize(img_proj, dim=-1)

        # patches projection
        img_patches_proj = self.img_patches_proj_linear(img_patches)
        ts_patches_proj = self.ts_patches_proj_linear(ts_patches)

        return img_proj, ts_proj, img_logits, ts_logits, img_patches_proj, ts_patches_proj

    def get_augmentations(self):
        return self.img_augs_train, self.img_augs_val, self.ts_augmentor

    def get_optimizer(self, cfg, model, loss=None):

        m = model.module if isinstance(model, torch.nn.DataParallel) else model
        
        param_groups = [
            {
                'params': m.img_proj_linear.parameters(),
                'lr': (
                    cfg.CLKD_train.optim.lr_peak * cfg.CLKD_train.optim.proj_linear_lr_scaler
                ),
                'weight_decay': cfg.CLKD_train.optim.weight_decay,
                'name': 'img_proj_linear'
            },
            {
                'params': m.timeseries_encoder.parameters(),
                'lr': cfg.CLKD_train.optim.lr_peak,
                'weight_decay': cfg.CLKD_train.optim.weight_decay,
                'name': 'timeseries_encoder'
            },
            {
                'params': m.ts_proj_linear.parameters(),
                'lr': (
                    cfg.CLKD_train.optim.lr_peak * cfg.CLKD_train.optim.proj_linear_lr_scaler
                ),
                'weight_decay': cfg.CLKD_train.optim.weight_decay,
                'name': 'ts_proj_linear'
            },
            {
                'params': m.ts_classif_head.parameters(),
                'lr': cfg.CLKD_train.optim.lr_peak,
                'weight_decay': cfg.CLKD_train.optim.weight_decay,
                'name': 'ts_classif_head'
            }
        ]
        if not cfg.CLKD_train.img_encoder_freeze:
            param_groups.append({
                'params': m.image_encoder.parameters(),
                'lr': cfg.CLKD_train.optim.lr_peak,
                'weight_decay': cfg.CLKD_train.optim.weight_decay,
                'name': 'image_encoder'
            })

        optimizer = torch.optim.AdamW(param_groups)
        return optimizer

    def set_lr(self, cfg, optimizer, lr):
        for param_group in optimizer.param_groups:
            if param_group['name'] == 'img_proj_linear':
                param_group['lr'] = (
                    lr * cfg.CLKD_train.optim.proj_linear_lr_scaler
                )
            elif param_group['name'] == 'ts_proj_linear':
                param_group['lr'] = (
                    lr * cfg.CLKD_train.optim.proj_linear_lr_scaler
                )
            elif param_group['name'] == 'timeseries_encoder': 
                param_group['lr'] = lr
            elif param_group['name'] == 'ts_classif_head':
                param_group['lr'] = (
                    lr
                )
            elif param_group['name'] == 'image_encoder':
                param_group['lr'] = lr

        return optimizer