# coding: utf-8

import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.utils import MyDataset

train_csv_path = os.path.join(".", "data", "input", "MushroomsRecognition", "train.csv")

train_bs = 8
valid_bs = 16
lr_init = 0.001
max_epoch = 1

# ------------ 数据加载------------#

image_mean = [0.38753143, 0.36847523, 0.27735737]
image_std = [0.25998375, 0.23844026, 0.2313706]

normTransform = transforms.Normalize(image_mean, image_std)
trainTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    normTransform
])

validTransform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset 实例
train_data = MyDataset(txt_path=train_csv_path, transform=trainTransform)
# valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
# valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)
