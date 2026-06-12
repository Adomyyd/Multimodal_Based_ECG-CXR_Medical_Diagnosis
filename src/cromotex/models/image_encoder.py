import os
import hydra
import torch
import torch.nn as nn
from torchvision import models, transforms
import torchxrayvision as xrv
from omegaconf import ListConfig

class DenseNet121XRV(nn.Module):
    def __init__(self, cfg):
        super(DenseNet121XRV, self).__init__()

        if cfg.pretrain_img.use_pretrained:
            weights = "densenet121-res224-mimic_nb"
            #"densenet121-res224-mimic_ch": not doing very well
        else:
            weights = None

        self.densenet = xrv.models.DenseNet(weights=weights)
        self.densenet.op_threshs = None #remove calibration for training, imp!
        if isinstance(cfg.pathology, ListConfig):
            pathology = []
            for p in cfg.pathology:
                if p == 'pleural_effusion':
                    pathology.append('Effusion')
                else:
                    pathology.append(p.capitalize())
        else:
            if cfg.pathology == 'pleural_effusion':
                pathology = ['Effusion']
            else:
                pathology = [cfg.pathology.capitalize()]
        self.pathology_indices = [
            self.densenet.pathologies.index(p) for p in pathology
        ]
        self.feature_dim = self.densenet.classifier.in_features

    def forward(self, x):
        x = x.mean(1).unsqueeze(1) #mean over channels (RGB)
        y = self.densenet(x)
        y = y[:, self.pathology_indices]
        return y

    def get_augmentations(self, cfg):

        train_augmentations = [
            transforms.ToTensor(),
            RenormalizeToCustomRange(
                old_min=0,
                old_max=1,
                new_min=-1024,
                new_max=1024
            )
        ]

        val_augmentations = [
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            RenormalizeToCustomRange(
                old_min=0,
                old_max=1,
                new_min=-1024,
                new_max=1024
            )
        ]

        train_augmentations = transforms.Compose(train_augmentations)
        val_augmentations = transforms.Compose(val_augmentations)
        return train_augmentations, val_augmentations

class RenormalizeToCustomRange(torch.nn.Module):
    def __init__(self, old_min=0, old_max=1, new_min=-1024, new_max=1024):
        super().__init__()
        self.old_min = old_min
        self.old_max = old_max
        self.new_min = new_min
        self.new_max = new_max

    def forward(self, tensor):
        # Scale from [old_min, old_max] to [new_min, new_max]
        tensor = (tensor - self.old_min) / (self.old_max - self.old_min)  
        tensor = tensor * (self.new_max - self.new_min) + self.new_min
        return tensor

def get_image_encoder(cfg):
    if cfg.cromotex.img_model == 'densenet121_xrv':
        return DenseNet121XRV(cfg)
    else:
        raise ValueError(
            f'Invalid img_model in config.yaml: {cfg.cromotex.img_model}'
        )

class CXRImageEncoder(DenseNet121XRV):
    def __init__(self, cfg, densenetCheckpoint=None):
        super(CXRImageEncoder, self).__init__(cfg)

        self.image_encoder = self.densenet.features
        self.classifier = self.densenet.classifier

        if densenetCheckpoint is not None:
            filepath = os.path.join(hydra.utils.to_absolute_path('checkpoints'),densenetCheckpoint)
            checkpoint = torch.load(filepath, map_location='cpu')
            encoder_state_dict = {
                k[18:]: v for k, v in checkpoint['model_state_dict'].items()
                if k.startswith('densenet.features')
            }
            classifier_state_dict = {
                k[20:]: v for k, v in checkpoint['model_state_dict'].items()
                if k.startswith('densenet.classifier')
            }
            self.image_encoder.load_state_dict(encoder_state_dict, strict=True)
            self.classifier.load_state_dict(classifier_state_dict, strict=True)
            print(f"Loaded checkpoint from {filepath}")
    def forward(self, x):
        x = x.mean(1).unsqueeze(1) #mean over channels (RGB)
        features = self.image_encoder(x)
        
        B, C, H, W = features.shape
        patch_features = features.permute(0, 2, 3, 1).reshape(B, H*W, C)  # [B, num_patches, feature_dim]
        # print(patch_features.shape)
        features = nn.functional.relu(features, inplace=True)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        y = self.classifier(features)
        y = y[:, self.pathology_indices]
        return features, patch_features, y