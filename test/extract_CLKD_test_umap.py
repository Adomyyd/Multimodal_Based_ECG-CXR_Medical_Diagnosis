import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from omegaconf import OmegaConf
import sys
# 1. 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 2. 获取当前文件所在目录（dir_A）
current_dir = os.path.dirname(current_file_path)
# 3. 获取上级目录（project）
parent_dir = os.path.dirname(current_dir)
# 4. 将上级目录添加到Python的搜索路径中
sys.path.append(parent_dir)
import hydra
from data_provider.data_loader import CXR_ECG_MatchedDataset
from src.cromotex.models.CLKD import CLKD


def load_config(config_root: str):
    config_path = os.path.join(config_root, 'config.yaml')
    cfg = OmegaConf.load(config_path)
    cromotex_path = os.path.join(config_root, 'cromotex', cfg.defaults[0]['cromotex']) if isinstance(cfg.defaults, list) else None
    if cromotex_path and os.path.exists(cromotex_path + '.yaml'):
        cfg.cromotex = OmegaConf.load(cromotex_path + '.yaml')
    elif os.path.exists(os.path.join(config_root, 'cromotex', 'cromotex_patch_transformer.yaml')):
        cfg.cromotex = OmegaConf.load(os.path.join(config_root, 'cromotex', 'cromotex_patch_transformer.yaml'))
    else:
        raise FileNotFoundError('Cannot find cromotex config file')
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Extract CLKD test features and visualize with UMAP')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/CLKD_origin_last_7.pth', help='Path to CLKD checkpoint')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for test dataloader')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of DataLoader workers')
    parser.add_argument('--output-dir', type=str, default='test_umap', help='Directory to save UMAP images')
    parser.add_argument('--sample-size', type=int, default=None, help='Optional number of test samples to use for UMAP')
    parser.add_argument('--config-root', type=str, default='config', help='Root directory for config files')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run model on')
    parser.add_argument('--hdf5-file', type=str, default='test_matched.h5', help='HDF5 test dataset filename')
    return parser.parse_args()


def build_model(cfg, device, checkpoint_path: str):
    # Ensure Hydra absolute path resolution is correct in this standalone script
    root = os.getcwd()
    hydra.utils.to_absolute_path = lambda x: os.path.join(root, x)

    model = CLKD(cfg)
    model = model.to(device)
    checkpoint_path = os.path.join(root, checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    # Handle DataParallel prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.') and not any(n.startswith('module.') for n in model.state_dict().keys()):
            new_state_dict[k[len('module.'):]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model


def extract_features(model, dataloader, device):
    feature_ts_list = []
    feature_img_list = []
    label_list = []

    with torch.no_grad():
        for img, ecg, labels in dataloader:
            img = img.to(device)
            ecg = ecg.to(device)
            labels = labels.cpu().numpy()

            img_proj, ts_proj, *_ = model(img, ecg)
            features_ts = ts_proj
            features_img = img_proj
            feature_ts_list.append(features_ts.cpu().numpy())
            feature_img_list.append(features_img.cpu().numpy())
            label_list.append(labels)

    features_ts = np.concatenate(feature_ts_list, axis=0)
    features_img = np.concatenate(feature_img_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    return features_ts, features_img, labels


def compute_umap(features, n_neighbors=15, min_dist=0.1, random_state=42):
    features = StandardScaler().fit_transform(features)
    umap = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return umap.fit_transform(features)


def plot_umap(projected, labels, pathology_names, output_dir, data_name):
    positive_color = 'red'
    negative_color = 'blue'

    os.makedirs(output_dir, exist_ok=True)

    for idx, pathology in enumerate(pathology_names):
        plt.figure(figsize=(10, 8))

        mask = labels[:, idx] == 1
        negative_mask = labels[:, idx] == 0
        if np.any(mask):
            plt.scatter(
                projected[mask, 0], projected[mask, 1],
                c=positive_color, label=pathology, alpha=0.9, s=30, marker='o', edgecolors='none'
            )
        if np.any(negative_mask):
            plt.scatter(
                projected[negative_mask, 0], projected[negative_mask, 1],
                c=negative_color, label='negative', alpha=0.5, s=20, marker='x', edgecolors='none'
            )
        plt.legend(loc='best', fontsize=12)
        plt.title(f'CLKD {data_name} UMAP: {pathology}', fontsize=16)
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'test_{data_name}_umap_{pathology}.png')
        plt.savefig(output_path, dpi=300)
        plt.close()


def main():
    args = parse_args()
    root_dir = os.getcwd()
    cfg = load_config(args.config_root)
    if torch.cuda.is_available() and args.device.startswith('cuda'):
        device = torch.device(args.device, 3)
    else:
        device = torch.device('cpu')
    model = build_model(cfg, device, args.checkpoint)

    _, _, ts_augmentor = model.get_augmentations()
    img_augmentations = model.get_augmentations()[1]
    test_dataset = CXR_ECG_MatchedDataset(cfg, args.hdf5_file, img_augmentations, None)
    if args.sample_size is not None and args.sample_size < len(test_dataset):
        indices = np.random.choice(len(test_dataset), size=args.sample_size, replace=False)
        from torch.utils.data import Subset
        test_dataset = Subset(test_dataset, indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )

    features_ts, features_img, labels = extract_features(model, test_loader, device)
    projected_ts = compute_umap(features_ts)
    projected_img = compute_umap(features_img)
    output_dir = os.path.join(root_dir, args.output_dir)
    output_dir = os.path.join(output_dir, f'{args.checkpoint.split("/")[-1].replace(".pth", "")}')
    plot_umap(projected_ts, labels, cfg.pathology, output_dir, 'ECG')
    plot_umap(projected_img, labels, cfg.pathology, output_dir, 'CXR')
    print(f'Saved UMAP visualizations to {output_dir}')


if __name__ == '__main__':
    main()
