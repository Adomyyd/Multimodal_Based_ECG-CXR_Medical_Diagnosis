import os
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pywt


def load_ucr_data(filepath):
    """
    加载UCR格式的csv数据
    UCR数据格式: 第一列是标签，后续列是时间序列数据
    """
    data = pd.read_csv(filepath, header=None, sep='\t')
    labels = data.iloc[:, 0].values
    time_series = data.iloc[:, 1:].values
    return time_series, labels


def normalize_ts(ts):
    """
    归一化时间序列到[0, 1]区间
    """
    ts_min = np.min(ts)
    ts_max = np.max(ts)
    if ts_max - ts_min == 0:
        return ts - ts_min
    return (ts - ts_min) / (ts_max - ts_min)


def interpolate_ts(ts, target_length):
    """
    使用插值将时间序列调整为固定长度
    """
    # 处理空序列的情况
    if len(ts) == 0:
        return np.zeros(target_length)
    
    if len(ts) == target_length:
        return ts
    
    # 处理只有一个点的情况
    if len(ts) == 1:
        return np.full(target_length, ts[0])
    
    x_old = np.linspace(0, 1, len(ts))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, ts, kind='linear')
    return f(x_new)


def compute_wavelet_transform(ts_data, wavelet='morl', scales=None):
    """
    计算时间序列的小波变换
    """
    if scales is None:
        scales = np.arange(1, min(128, len(ts_data) // 2))
    
    coefficients, frequencies = pywt.cwt(ts_data, scales, wavelet)
    return coefficients, frequencies


def draw_ts_as_image(ts_data, image_path, image_size=(224, 224), mode='line'):
    """
    将时间序列绘制为图像
    """
    # 创建一个新的图像
    fig = plt.figure(figsize=(image_size[0]/100, image_size[1]/100), dpi=100)
    ax = fig.add_subplot(111)
    
    # 隐藏坐标轴
    ax.axis('off')
    
    if mode == 'line':
        # 如果是多维时间序列，绘制所有维度
        if ts_data.ndim > 1:
            for dim in range(ts_data.shape[1]):
                normalized_ts = normalize_ts(ts_data[:, dim])
                x = np.linspace(0, 1, len(normalized_ts))
                ax.plot(x, normalized_ts, linewidth=1.0)
        else:
            # 单维时间序列
            normalized_ts = normalize_ts(ts_data)
            x = np.linspace(0, 1, len(normalized_ts))
            ax.plot(x, normalized_ts, linewidth=1.0)
    elif mode == 'wavelet':
        if ts_data.ndim > 1:
            # 对于多通道时间序列，为每个通道计算小波变换并显示为子图
            num_channels = ts_data.shape[1]
            # 计算子图的行列数，尽量使布局接近正方形
            subplot_cols = int(np.ceil(np.sqrt(num_channels)))
            subplot_rows = int(np.ceil(num_channels / subplot_cols))
            
            # 清除默认的axes
            fig.clear()
            
            # 为每个通道创建子图
            for dim in range(num_channels):
                normalized_ts = normalize_ts(ts_data[:, dim])
                coefficients, _ = compute_wavelet_transform(normalized_ts)
                
                # 添加子图
                ax = fig.add_subplot(subplot_rows, subplot_cols, dim + 1)
                ax.axis('off')
                ax.imshow(np.abs(coefficients), aspect='auto', cmap='jet')
        else:
            # 单维时间序列
            normalized_ts = normalize_ts(ts_data)
            coefficients, _ = compute_wavelet_transform(normalized_ts)
            ax.imshow(np.abs(coefficients), aspect='auto', cmap='jet')
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)


def process_ucr_dataset(data_dir, target_length=1000, image_size=(224, 224), mode='line'):
    """
    处理整个UCR数据集
    """
    # 查找训练和测试文件
    train_file = None
    test_file = None
    
    for file in os.listdir(data_dir):
        if file.endswith('_TRAIN.tsv') or file.endswith('_train.tsv'):
            train_file = os.path.join(data_dir, file)
        elif file.endswith('_TEST.tsv') or file.endswith('_test.tsv'):
            test_file = os.path.join(data_dir, file)
    
    # 创建处理后的数据目录
    if mode == 'line':
        processed_dir = os.path.join(data_dir, 'processed_data')
    else:
        processed_dir = os.path.join(data_dir, 'processed_wavelet_data')
    os.makedirs(processed_dir, exist_ok=True)
    
    # 处理训练数据
    if train_file and os.path.exists(train_file):
        print(f"Processing train data: {train_file}")
        ts_data, labels = load_ucr_data(train_file)
        
        for i in range(len(ts_data)):
            # 获取时间序列数据
            ts = ts_data[i]
            
            # 如果是多维，保持原样；如果是1维，转换为列向量
            if ts.ndim == 1:
                ts = ts.reshape(-1, 1)
            
            # 插值到目标长度
            if ts.shape[0] != target_length:
                # 对每一维度进行插值
                interpolated_ts = np.zeros((target_length, ts.shape[1]))
                for dim in range(ts.shape[1]):
                    interpolated_ts[:, dim] = interpolate_ts(ts[:, dim], target_length)
                ts = interpolated_ts
            
            # 保存图像
            image_path = os.path.join(processed_dir, f"{i}.png")
            draw_ts_as_image(ts, image_path, image_size, mode)
        
        # 保存标签
        np.save(os.path.join(processed_dir, 'train_labels.npy'), labels)
        print(f"Processed {len(ts_data)} train samples")
    
    # 处理测试数据
    if test_file and os.path.exists(test_file):
        print(f"Processing test data: {test_file}")
        ts_data, labels = load_ucr_data(test_file)
        
        for i in range(len(ts_data)):
            # 获取时间序列数据
            ts = ts_data[i]
            
            # 如果是多维，保持原样；如果是1维，转换为列向量
            if ts.ndim == 1:
                ts = ts.reshape(-1, 1)
            
            # 插值到目标长度
            if ts.shape[0] != target_length:
                # 对每一维度进行插值
                interpolated_ts = np.zeros((target_length, ts.shape[1]))
                for dim in range(ts.shape[1]):
                    interpolated_ts[:, dim] = interpolate_ts(ts[:, dim], target_length)
                ts = interpolated_ts
            
            # 保存图像
            image_path = os.path.join(processed_dir, f"{i+len(ts_data)}.png")
            draw_ts_as_image(ts, image_path, image_size, mode)
        
        # 保存标签
        np.save(os.path.join(processed_dir, 'test_labels.npy'), labels)
        print(f"Processed {len(ts_data)} test samples")


def main():
    parser = argparse.ArgumentParser(description='Construct images from UCR time series data')
    parser.add_argument('--data_dir', required=True, help='Path to UCR dataset directory')
    parser.add_argument('--target_length', type=int, default=1000, help='Target length for time series interpolation')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='Output image size (height, width)')
    parser.add_argument('--mode', type=str, choices=['line', 'wavelet'], default='line', help='Visualization mode: line (default) or wavelet')
    
    args = parser.parse_args()
    
    print(f"Processing UCR dataset in {args.data_dir}")
    print(f"Visualization mode: {args.mode}")
    process_ucr_dataset(args.data_dir, args.target_length, tuple(args.image_size), args.mode)
    print("Image construction completed!")


if __name__ == '__main__':
    main()