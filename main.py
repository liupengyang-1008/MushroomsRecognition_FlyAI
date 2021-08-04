# -*- coding: utf-8 -*-
import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from flyai.data_helper import DataHelper
from flyai.framework import FlyAI
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from mymodel.vgg import vgg11
from path import MODEL_PATH
from utils.utils import MyDataset


'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=1, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=8, type=int, help="batch size")
parser.add_argument('-gpu', default=False, help='use gpu or not')
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate')
parser.add_argument('-net', type=str, default='lenet', help='Net of project')
parser.add_argument('-loss', type=str, default='CE', help='Net of project')

args = parser.parse_args()

image_mean = [0.38753143, 0.36847523, 0.27735737]
image_std = [0.25998375, 0.23844026, 0.2313706]
max_epochs = args.EPOCHS
train_bs = args.BATCH
train_csv_path = os.path.join(".", "data", "input", "MushroomsRecognition", "train.csv")


# 判断GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_gpu = torch.cuda.is_available()

# 创建保存模型的文件夹
save_dir = './trained'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def __init__(self):
        self.train_trans = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(image_mean, image_std)
        ])

    def init_net(self):
        self.net = vgg11()
        self.net.to(device)

        if args.loss == 'MSE':
            self.loss_function = nn.MSELoss(reduction='mean')
        if args.loss == 'CE':
            self.loss_function = nn.CrossEntropyLoss()

    def download_data(self):
        # 根据数据ID下载训练数据

        data_helper = DataHelper()
        data_helper.download_from_ids("MushroomsRecognition")

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        # 构建MyDataset 实例
        self.train_data = MyDataset(txt_path=train_csv_path, transform=self.train_trans)
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=train_bs, shuffle=True, drop_last=True)


    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        self.init_net()
        self.deal_with_data()
        optimizer = optim.Adam(self.net.parameters(), lr=0.001, betas=(0.9, 0.999))

        criterion = self.loss_function
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        for epoch in range(max_epochs):
            loss_sigma = 0.0
            correct = 0.0
            total = 0.0
            scheduler.step()

            for i, data in enumerate(self.train_loader):
                inputs, labels = data
                if use_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # 统计预测信息
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if use_gpu:
                    correct += (predicted == labels).cpu().squeeze().sum().numpy()
                else:
                    correct += (predicted == labels).squeeze().sum().numpy()
                loss_sigma += loss.item()

                if i%10 == 0:
                    loss_avg = loss_sigma / 10
                    loss_sigma = 0.0
                    print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                        epoch + 1, max_epochs, i + 1, len(self.train_loader), loss_avg, correct / total))

            torch.save(self.net, save_dir + '/trained_model.pth')


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.train()