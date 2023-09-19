import random

import cv2
from PIL import Image, ImageEnhance
from astropy.visualization import make_lupton_rgb
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

from utlis.dataset_split import *


class MyDataset(Dataset):

    def __init__(self, files, names, hyp_dict=None):
        self.files, self.dataset_save_method = files
        self.hyp_dict = hyp_dict
        self.class_dict = dict(zip(names, range(len(names))))
        self.w, self.h = self.hyp_dict["size"][0], self.hyp_dict["size"][1]
        self.x = np.zeros((self.w, self.h, 6))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        path = self.files[item]

        data = None
        if path.lower().endswith((".jpg", ".png", ".jpeg", ".gif")):
            data = cv2.imread(path)
        elif path.lower().endswith(".npy"):
            data = np.load(path)  # ugriz

        gri = make_lupton_rgb(data[:, :, 1], data[:, :, 2], data[:, :, 3], Q=8, stretch=0.5)
        urz = make_lupton_rgb(data[:, :, 0], data[:, :, 2], data[:, :, 4], Q=8, stretch=0.5)

        # 根据路径获取的标签值
        if self.dataset_save_method == 'file':
            label = self.class_dict[path.split(' ')[0]]
        elif self.dataset_save_method == 'folder':
            label = self.class_dict[path.split("\\")[-2]]

        if self.hyp_dict["augment"]:
            if random.random() < self.hyp_dict["flipud"]:
                gri = np.flipud(gri)
                urz = np.flipud(urz)

            # flip left-right
            if random.random() < self.hyp_dict["fliplr"]:
                gri = np.fliplr(gri)
                urz = np.fliplr(urz)

            if random.random() < self.hyp_dict["rot90"]:
                for _ in range(0, np.random.randint(1, 3)):
                    gri = np.rot90(gri, 1, (0, 1))
                    urz = np.rot90(urz, 1, (0, 1))

            if random.random() < self.hyp_dict["brightness"]:
                if gri.shape[1] <= 20:
                    brightness = 1 + 0.2 * np.random.random()
                else:
                    brightness = 0.8 + 0.4 * np.random.random()
                gri = self.brightnessEnhancement(gri, brightness)
                urz = self.brightnessEnhancement(urz, brightness)

        gri = cv2.resize(gri, (self.w, self.h))
        urz = cv2.resize(urz, (self.w, self.h))

        self.x[:, :, 0:3] = gri
        self.x[:, :, 3:6] = urz

        x = np.transpose(self.x / 255, (2, 0, 1)).astype(np.float32)

        return x, label

    def brightnessEnhancement(self, image, brightness):  # 亮度增强
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        enh_bri = ImageEnhance.Brightness(image)
        image_brightened = enh_bri.enhance(brightness)
        image_brightened = cv2.cvtColor(np.array(image_brightened), cv2.COLOR_RGB2BGR)
        return image_brightened


def create_dataset(files, names, hyp_dict):
    dataset = MyDataset(files, names, hyp_dict)

    return dataset


def create_dataloader(dataset, hyp_dict):
    batch_size = min(hyp_dict['batch_size'], len(dataset))

    dataloader = DataLoader(dataset=dataset,
                            num_workers=hyp_dict['workers'],
                            shuffle=True,
                            batch_size=batch_size,
                            pin_memory=True)
    return dataloader


def get_train_dataloader(data_dict, names, hyp_dict):
    train_path = data_dict['train']
    val_path = data_dict['val']

    if os.path.isfile(train_path):
        train_files = open(train_path).readlines()  # class file_path
        dataset_save_method = 'file'
    else:
        train_files = get_file_paths(train_path)
        dataset_save_method = 'folder'

    train_dataset = create_dataset((train_files, dataset_save_method), names, hyp_dict)

    if val_path is None:
        # 划分数据集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        if os.path.isfile(train_path):
            val_files = open(train_path).readlines()  # class file_path
        else:
            val_files = get_file_paths(train_path)
        val_dataset = create_dataset((val_files, dataset_save_method), names, hyp_dict)

    val_dataset.augment = False
    train_loader = create_dataloader(train_dataset, hyp_dict)
    val_loader = create_dataloader(val_dataset, hyp_dict)

    return train_loader, val_loader
