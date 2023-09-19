import os
import shutil
from tqdm import tqdm
import numpy as np


def get_file_paths(folder_path):
    file_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


def dataset_split(data_path, out_path, names, split_dict):
    np.random.seed(42)
    test = []
    for c in tqdm(names):
        path = f'{data_path}\\{c}'
        path_list = get_file_paths(path)
        np.random.shuffle(path_list)
        np.random.seed(None)
        train_ratio, val_ratio, test_ratio = split_dict['train'], split_dict['val'], split_dict['test']
        train_len = int(len(path_list) * train_ratio)
        train = path_list[:train_len]

        os.makedirs(os.path.join(out_path, 'train', c), exist_ok=True)
        for path in train:
            _, name = os.path.split(path)
            shutil.copy(path, os.path.join(out_path, 'train', c, name))

        if val_ratio != 0.0:
            val_len = int(len(path_list) * val_ratio)
            val = path_list[train_len:train_len + val_len]
            os.makedirs(os.path.join(out_path, 'val', c), exist_ok=True)
            for path in val:
                filepath, name = os.path.split(path)
                shutil.copy(path, os.path.join(out_path, 'val', c, name))
            if test_ratio != 0.0:
                test = path_list[train_len + val_len:]

        else:
            test = path_list[train_len:]

        if len(test) != 0:
            os.makedirs(os.path.join(out_path, 'test', c), exist_ok=True)
            for path in test:
                filepath, name = os.path.split(path)
                shutil.copy(path, f'{out_path}\\test\\{c}\\{name}')

