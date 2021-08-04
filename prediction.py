# -*- coding: utf-8 -*
from flyai.framework import FlyAI
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import cv2
import os

image_mean = [0.38753143, 0.36847523, 0.27735737]
image_std = [0.25998375, 0.23844026, 0.2313706]

test_train_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(image_mean, image_std)
])


class Prediction(FlyAI):
    def __init__(self):
        super(Prediction, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std)
        ])
        self.use_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        self.model_path = './trained/trained_model.pth'
        self.net = torch.load(self.model_path)
        self.net.eval()

    def predict(self, image_path):
        '''
        模型预测返回结果
        :param input: 评估传入样例 {"image_path": "./data/input/image/image/133_AZcM0VPHmxw.jpg"}
        :return: 模型预测成功后，直接返回预测的结果 {"label": 0}
        '''
        # return {"label": 0}
        img = Image.open(image_path).convert(
            'RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)

        if self.use_gpu:
            out = self.net(img.unsqueeze(dim=0).cuda())
            pred_label = np.argmax(out.cpu().detach().numpy())
        else:
            out = self.net(img.unsqueeze(dim=0))
            pred_label = np.argmax(out.detach().numpy())

        # print("label:", pred_label)
        return {"label": pred_label}
