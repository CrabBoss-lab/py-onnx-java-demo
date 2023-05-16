# -*- codeing = utf-8 -*-
# @Time :2023/5/8 8:15
# @Author :yujunyu
# @Site :
# @File :net.py
# @software: PyCharm

import torch
import torch.nn as nn


#####
# 搭建CNN：三层卷积、两层全连接
#####

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 26 * 26, 64)
        self.fc2 = nn.Linear(64, 3)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 26 * 26)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    model = Net()
    print(f"\033[34m{model}\033[0m")

    from torchsummary import summary

    if torch.cuda.is_available():
        model.cuda()
    summary(model, input_size=(3, 224, 224))
