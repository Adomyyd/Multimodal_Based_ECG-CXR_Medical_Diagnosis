import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

# 定义路径
base_data_path = '/media/omnisky/Disk8.0T/rj/data/MIMIC/processed/'
output_path = '/media/omnisky/Disk8.0T/rj/data/MIMIC/processed_csv/'
failed_files_path = '/media/omnisky/Disk8.0T/rj/AimTS/preprocess/failed_files_pair.txt'

# 确保输出目录存在
os.makedirs(output_path, exist_ok=True)

# 1. 加载 failed_files_pair.txt 中的 ECG 文件名
failed_ecgs = set()
if os.path.exists(failed_files_path):
    print(f"Reading failed files from {failed_files_path}...")
    with open(failed_files_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Parsing failed files"):
            # 匹配 ECG: pXXXX/pXXXXXXXX/sXXXXXXXX/XXXXXXXX
            match = re.search(r'ECG: ([\w/]+)', line)
            if match:
                failed_ecgs.add(match.group(1).strip())
    print(f"Loaded {len(failed_ecgs)} unique failed ECG files.")
else:
    print(f"Warning: Failed files list not found at {failed_files_path}")

# 定义需要保存的列名顺序
columns_to_keep = [
    'subject_id', 
    'ecg_filename', 
    'cxr_filename', 
    'cardiomegaly', 
    'edema', 
    'enlarged_cardiomediastinum', 
    'pleural_effusion', 
    'pneumonia'
]

# 处理三个文件
for split in ['train', 'val', 'test']:
    pkl_file = os.path.join(base_data_path, f'df_{split}.pkl')
    if not os.path.exists(pkl_file):
        print(f"File {pkl_file} not found, skipping...")
        continue
    
    print(f"Processing {split} split...")
    
    # 1. 读取内容
    df = pd.read_pickle(pkl_file)
    initial_len = len(df)
    
    # 2. 筛选记录
    # 筛选掉 ecg_filename 为 NaN 的记录
    df = df.dropna(subset=['ecg_filename'])
    nan_filtered_len = len(df)
    
    # 筛选掉 ecg_filename 在 failed_files_pair.txt 中的记录
    df = df[~df['ecg_filename'].isin(failed_ecgs)]
    final_len = len(df)
    
    print(f"  Initial records: {initial_len}")
    print(f"  After removing NaN ECG: {nan_filtered_len} (Removed {initial_len - nan_filtered_len})")
    print(f"  After removing failed ECGs: {final_len} (Removed {nan_filtered_len - final_len})")
    
    # 3. 按照指定顺序保存列
    # 确保列名存在，如果不存在则填充 NaN (例如可能有些 pkl 里没选中的列)
    for col in columns_to_keep:
        if col not in df.columns:
            df[col] = np.nan
            
    df_output = df[columns_to_keep]
    
    # 保存为 CSV
    save_name = f'df_matched_pretrain_{split}.csv'
    save_path = os.path.join(output_path, save_name)
    df_output.to_csv(save_path, index=False)
    print(f"  Saved to {save_path}")

print("\nDone!")
