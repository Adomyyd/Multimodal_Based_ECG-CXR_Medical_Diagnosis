import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from .uea import subsample, interpolate_missing, Normalizer
from construct_image.monash_cls import load_from_tsfile_to_dataframe
from PIL import Image
from torchvision import transforms
import h5py
import wfdb
import preprocess.preprocess as preprocess
import warnings
import torchxrayvision as xrv

warnings.filterwarnings('ignore')


def preprocess_ecg_signal(ecg_file):
    """封装 ECG 信号处理流程"""
    try:
        # Load raw ECG signal
        signal, fields = wfdb.rdsamp(ecg_file)
        signal[np.isnan(signal)] = 0.0
        
        # 1. Lead consistency
        signal = preprocess.ecg_consistency(signal, fields)
        
        # 2. Resample from fields['fs'] to 100Hz
        signal = preprocess.resample_signal_poly(signal, fields['fs'], 100)
        
        # 3. Baseline wander removal (expects [n_leads, n_samples])
        signal = preprocess.baseline_wander_removal(signal.T, sampling_frequency=100).T
        
        # 4. Normalize per-lead with min = -1.0 and max = 1.0
        signal = preprocess.normalize_per_lead(signal)
        
        # Ensure fixed length of 1000 samples (10s @ 100Hz)
        if signal.shape[0] < 1000:
            pad = np.zeros((1000 - signal.shape[0], signal.shape[1]))
            signal = np.vstack([signal, pad])
        elif signal.shape[0] > 1000:
            signal = signal[:1000, :]
            
        return signal
    except Exception as e:
        print(f"Error processing ECG {ecg_file}: {e}")
        return None


def preprocess_cxr_image(img_path, size=224):
    """封装 CXR 图像处理流程"""
    try:
        # 1. 转灰度图
        img = Image.open(img_path).convert('L')
        img_np = np.array(img)
        
        # 2. 使用 xrv.datasets.normalize 进行标准化
        # Note: xrv.datasets.normalize expects input in range [0, 255] or [0, 1]
        # It typically scales to [-1024, 1024]
        img_np = xrv.datasets.normalize(img_np, 255)
        
        # 3. 转 224 图像大小
        # 使用 PIL Resize 比较方便
        img_pil = Image.fromarray(img_np)
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        img_t = transform(img_pil)
        
        # 转换为 3 通道以保持与其他模型兼容 (3, 224, 224)
        if img_t.shape[0] == 1:
            img_t = img_t.repeat(3, 1, 1)
            
        return img_t
    except Exception as e:
        print(f"Error processing CXR {img_path}: {e}")
        return None


class Dataloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug(fast load dataset to verify code)
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, loader, dataset_name, file_list=None, limit_size=None, flag=None, data_path=None):
        self.dataset_name = dataset_name
        self.all_df, self.labels_df = self.load_all(loader, dataset_name, file_list=file_list, flag=flag, data_path=data_path)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)
        self.num_class = len(np.unique(self.labels_df))

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)

    def load_all(self, loader, dataset_name, file_list=None, flag=None, data_path=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        root_path = get_file_root(loader, dataset_name, data_path)
        self.root_path = root_path

        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern = '*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0], dataset_name)  # a single file contains dataset

        return all_df, labels_df

    # 使用npy缓存文件进行加速
    def load_single(self, filepath, dataset_name):        
        # Create cache directory
        cache_root = os.path.join(os.getcwd(), 'cache')
        if not os.path.exists(cache_root):
            os.makedirs(cache_root)
        
        dataset_cache_dir = os.path.join(cache_root, dataset_name)
        if not os.path.exists(dataset_cache_dir):
            os.makedirs(dataset_cache_dir)
        
        cache_data_path = os.path.join(dataset_cache_dir, f"{dataset_name}_data.npy")
        cache_labels_path = os.path.join(dataset_cache_dir, f"{dataset_name}_labels.npy")
        cache_meta_path = os.path.join(dataset_cache_dir, f"{dataset_name}_meta.npy")
        
        # Check if cache exists
        if os.path.exists(cache_data_path) and os.path.exists(cache_labels_path) and os.path.exists(cache_meta_path):
            print(f"Loading {dataset_name} from cache...")
            data_array = np.load(cache_data_path, allow_pickle=True)  # shape: (N, seq_len, feat_dim)
            labels_array = np.load(cache_labels_path, allow_pickle=True)
            meta_info = np.load(cache_meta_path, allow_pickle=True)
            self.max_seq_len = int(meta_info[0])
            
            # Reconstruct labels_df
            labels = pd.Series(labels_array, dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int32)
            
            # Reconstruct df as DataFrame of arrays (each cell is a (seq_len,) array)
            N, seq_len, feat_dim = data_array.shape
            data_flat = data_array.reshape(N * seq_len, feat_dim)
            sample_indices = np.repeat(np.arange(N), seq_len) # set_index
            df = pd.DataFrame(data_flat, columns=[f'dim_{j}' for j in range(feat_dim)], index=sample_indices)
    
        else:
            print(f"Loading {dataset_name} from .ts file and creating cache...")
            df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                    replace_missing_vals_with='NaN')
            labels = pd.Series(labels, dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes,
                                    dtype=np.int32) 

            lengths = df.applymap(
                lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

            horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

            if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
                df = df.applymap(subsample)

            lengths = df.applymap(lambda x: len(x)).values
            vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
            if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
                self.max_seq_len = int(np.max(lengths[:, 0]))
            else:
                self.max_seq_len = lengths[0, 0]

            padded_data = []
            for row in range(df.shape[0]):
                sample_padded = []
                for col in range(df.shape[1]):
                    series = df.iloc[row, col]
                    if len(series) < self.max_seq_len:
                        extended_series = np.pad(series, (0, self.max_seq_len - len(series)), 
                                            mode='constant', constant_values=series[-1] if len(series) > 0 else 0)
                    else:
                        extended_series = series[:self.max_seq_len]
                    # Apply interpolate_missing to the padded series
                    interpolated_series = interpolate_missing(extended_series)
                    sample_padded.append(interpolated_series)
                padded_data.append(sample_padded)
            
            # Build df with one row per sample
            df = pd.DataFrame(padded_data, columns=[f'dim_{j}' for j in range(df.shape[1])])

            # Save to cache
            # Convert to (N, seq_len, feat_dim) array for efficient storage
            N = len(df)
            feat_dim = len(df.columns)
            data_array = np.empty((N, self.max_seq_len, feat_dim), dtype=np.float32)
            for i in range(N):
                for j in range(feat_dim):
                    data_array[i, :, j] = df.iloc[i, j]
            
            np.save(cache_data_path, data_array, allow_pickle=False)
            np.save(cache_labels_path, labels.cat.codes.values, allow_pickle=False)
            np.save(cache_meta_path, np.array([self.max_seq_len]), allow_pickle=False)
            print(f"Cache files created for {dataset_name}")

        return df, labels_df

    # 之前作者提供的load_single函数，由于导入数据太慢，因此做了相关加速优化(参考load_single)
    def load_single_ago(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                   replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int32)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        # get image and to tensor
        img_path = os.path.join(self.root_path, 'processed_data/{}.png'.format(self.all_IDs[ind]))
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_transform = transforms.Compose([transforms.ToTensor()])
        img = img_transform(img)  # [C H W]
        
        img_freq_path = os.path.join(self.root_path, 'wavelet_processed_data/{}.png'.format(self.all_IDs[ind]))
        img_freq = Image.open(img_freq_path)
        img_freq = img_freq.convert('RGB')
        img_freq = img_transform(img_freq)  # [C H W]
        
        # return feature(numpy)、label(numpy)、img(tensor,折线图)、img_freq(tensor,频谱图)
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
           torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values), \
           img, \
           img_freq, \
           self.dataset_name  # 添加返回数据集名称

    def __len__(self):
        return len(self.all_IDs)


def get_file_root(loader, filename, data_path=None):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, data_path)
    file_path = os.path.join(file_path, loader)
    file_path = os.path.join(file_path, filename)
    return file_path


DATASET_BASE_DIR = '/media/omnisky/Disk8.0T/rj/data/MIMIC/processed/'
pathologies = [
    "cardiomegaly", "edema",
    "enlarged_cardiomediastinum", "pleural_effusion", "pneumonia"
]

class CXR_ECG_MatchedDataset(Dataset):
    def __init__(self, cfg, hdf5_file_path, cxr_augmentations=None, ecg_augmentor=None):
        self.mode = "hdf5"
        self.hdf5_path = os.path.join(DATASET_BASE_DIR, hdf5_file_path)
        
        # 不要在这里打开 hdf5_file，避免多进程读取冲突
        self.hdf5_file = None
        self.images = None
        self.ecg = None
        
        # 将标签一次性读入内存，减少后续 IO
        with h5py.File(self.hdf5_path, 'r') as f:
            self.total_len = len(f['images'])
            self.labels = f['labels'][:]
            
        self.cxr_augmentations = cxr_augmentations
        self.ecg_augmentor = ecg_augmentor

    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        # 延迟初始化，确保每个 DataLoader worker 拥有独立的文件句柄
        if self.hdf5_file is None:
            self.hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=True)
            self.images = self.hdf5_file['images']
            self.ecg = self.hdf5_file['ecg']

        # 1. 图像加载优化 (尽量保持在 numpy 直到必要时刻)
        img_data = self.images[idx]
        if self.cxr_augmentations:
            # transforms.ToPILImage 处理 numpy 数组效率更高
            # print(f"img_data.shape: {img_data.shape}, dtype: {img_data.dtype}")
            img_data = img_data.transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
            img = transforms.ToPILImage()(img_data)
            img = self.cxr_augmentations(img)
        else:
            img = torch.from_numpy(img_data).float()

        # 2. 标签加载 (从内存数组读取，极快)
        label = torch.tensor(self.labels[idx][[0,1,3],], dtype=torch.long)

        # 3. ECG 加载优化
        ecg_data = self.ecg[idx]
        # if ecg_data.shape[0] == 12: # 转置为 (T, C)
        #     ecg_data = ecg_data.T
        
        if self.ecg_augmentor:
            ecg_data = self.ecg_augmentor.augment(ecg_data)
        ecg = torch.from_numpy(ecg_data).float()

        # print(f"img.shape: {img.shape}, ecg.shape: {ecg.shape}")

        return img, ecg, label

    def get_labels(self):
        return self.labels


class ECG_Dataset(Dataset):
    """
    Dataset for loading and preprocessing ECG signals from raw files.
    """
    def __init__(self, csv_path, ecg_root_path='/media/omnisky/Disk8.0T/rj/data/MIMIC/mimic-iv-ecg/1.0/files', ecg_augmentor=None):
        self.df = pd.read_csv(csv_path)
        self.ecg_root_path = ecg_root_path
        self.ecg_augmentor = ecg_augmentor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ecg_filename = row['ecg_filename']
        ecg_file = os.path.join(self.ecg_root_path, ecg_filename)
        
        label = 0
        if 'label' in row:
            label = row['label']
        
        dummy_img = torch.zeros((3, 224, 224)).float()
        
        # 使用封装后的处理函数
        signal = preprocess_ecg_signal(ecg_file)
        
        if self.ecg_augmentor:
            signal = self.ecg_augmentor.augment(signal)
        
        batch = torch.from_numpy(signal.copy()).float()
        
        return batch, torch.tensor(label).long(), dummy_img
