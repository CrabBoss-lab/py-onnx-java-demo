# Author:yujunyu
# -*- codeing = utf-8 -*-
# @Time :2022/9/9 16:20
# @Author :yujunyu
# @Site :
# @File :test.py
# @software: PyCharm

"""
加载数据集、数据预处理
    数据集目录格式:
        |-dataset
            |-train
                |-类别1
                    |-img.png
                    |-img2.png
                    |-...
                |-类别2
                |-...
    注意:同一类别在同一文件夹下，不同类别文件夹是同级目录
"""

from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, RandomHorizontalFlip, RandomVerticalFlip, \
    RandomResizedCrop, ColorJitter, RandomGrayscale, RandomCrop
import torch.utils.data

from reset_transforms_resize import CV2_Resize
import cv2


def Load_data(trainDir: str, valDir: str, shape: tuple, batch_size: int, num_workers: int):
    '''
    # 加载指定目录下的图像，返回数据加载器
    :param trainDir:
    :param valDir:
    :param shape:
    :param batch_size:
    :return:train_loader, val_loader, train_set.class_to_idx
    '''
    # 图像变换
    transform_train = Compose([
        # Resize(shape)
        # Resize(shape,interpolation=InterpolationMode.BILINEAR),
        CV2_Resize(shape, interpolation=cv2.INTER_LINEAR),
        RandomCrop(224, padding=20),
        RandomHorizontalFlip(),  # 0.5的进行水平翻转
        RandomVerticalFlip(),  # 0.5的进行垂直翻转
        ToTensor(),  # PIL转tensor
        # mean = tensor([0.4740, 0.4948, 0.4338]),std = tensor([0.1920, 0.1591, 0.2184])
        # Normalize(mean=[0.4740, 0.4948, 0.4338], std=[0.1920, 0.1591, 0.2184])
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        # 归一化   # 输入必须是Tensor
    ])
    transform_val = Compose([
        # Resize(shape),
        # Resize(shape,interpolation=InterpolationMode.BILINEAR),
        CV2_Resize(shape, interpolation=cv2.INTER_LINEAR),
        ToTensor(),
        # Normalize(mean=[0.4740, 0.4948, 0.4338], std=[0.1920, 0.1591, 0.2184])
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 加载数据集
    train_set = ImageFolder(trainDir, transform=transform_train)
    val_set = ImageFolder(valDir, transform=transform_val)

    # 封装批处理的迭代器（加载器）
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, train_set.class_to_idx


def Get_datasets_info(train_loader, val_loader, class_to_idx):
    '''
    打印train_loader、val_loader的信息
    :param train_loader:
    :param val_loader:
    :return:
    '''
    img_num, lab_num = 0, 0
    for image, label in train_loader:
        img_num += len(image)
        lab_num += len(label)
        # print(image, label)
    print(f'\033[32mtrain_inputs:{img_num}\ttrain_labels:{lab_num}\033[0m')

    img_num, lab_num = 0, 0
    for image, label in val_loader:
        img_num += len(image)
        lab_num += len(label)
    print(f'\033[32mval_inputs:{img_num}\tval_labels:{lab_num}\033[0m')

    print(f'\033[32mlabel_map:{class_to_idx}\033[0m')


# 测试
if __name__ == "__main__":
    train_loader, val_loader, class_to_idx = Load_data(
        trainDir='./dataset/train',
        valDir='./dataset/val',
        shape=(224, 224),
        batch_size=128,
        num_workers=0
    )
    Get_datasets_info(train_loader, val_loader, class_to_idx)
