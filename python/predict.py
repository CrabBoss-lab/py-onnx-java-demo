# -*- codeing = utf-8 -*-
# @Time :2023/5/12 10:43
# @Author :yujunyu
# @Site :
# @File :predict2222.py
# @software: PyCharm

import torch
from torchvision import transforms
from PIL import Image
import time

import cv2
from reset_transforms_resize import CV2_Resize

from net import Net


# 定义预测类
class Predict:
    def __init__(self, model):
        # 加载整个模型（包括网络结构、参数）
        # self.model = model
        # self.net = torch.load(self.model, map_location='cpu')

        # 加载神经网络模型
        self.net = Net()
        # 加载预训练模型参数
        self.model = model
        state = torch.load(self.model, map_location='cpu')
        self.net.load_state_dict(state)

        print('模型加载完成！')
        # 将模型设置为评估模式
        self.net.eval()

    @torch.no_grad()
    def recognize(self, img, label_map=None):
        # 图像预处理
        img = self.preprocess(img)
        # 进行预测
        y = self.net(img)
        print(y)
        # 计算输出的概率分布
        p_y = torch.nn.functional.softmax(y, dim=1)
        print(p_y)
        # 获取预测的概率、类别
        p, cls_index = torch.max(p_y, dim=1)
        # print(p)
        # 获取预测类别对应的类别名称
        cls_name = label_map[cls_index]
        return cls_name, p.item()

    def preprocess(self, img):
        # 定义图像预处理
        transform = transforms.Compose([
            # transforms.Resize((224, 224)),  # resize
            # transforms.Resize((224, 224), interpolation=cv2.INTER_LINEAR),  # resize
            CV2_Resize((224, 224), interpolation=cv2.INTER_LINEAR),
            transforms.ToTensor(),  # 转为tensor
            # transforms.Normalize(mean=[0.4737, 0.4948, 0.4336], std=[0.1920, 0.1592, 0.2184])  # 归一化
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # 图像预处理并转换为形状为(1, C, H, W)的张量
        img_tensor = transform(img).unsqueeze(0)

        # 临时保存一下预处理后的图片
        # utils.save_image(img_tensor, 'img.png')
        # print(img_tensor[0][0][0])

        return img_tensor


if __name__ == '__main__':
    # # 模型路径
    model_path = 'weight/model.pth'
    # # 创建预测类
    recognizer = Predict(model_path)
    # # 标签映射表
    label_map = ['番茄叶斑病', '苹果黑星病', '葡萄黑腐病']

    # 测试图片路径
    img_path = './dataset/test/番茄叶斑病/番茄叶斑病 (25).JPG'
    print(f'预测单张图片:{img_path}')
    # 打开测试图片
    img = Image.open(img_path)
    # 进行预测
    st = time.time()
    cls_name, p = recognizer.recognize(img, label_map)
    # 输出预测结果
    print(f'推理时间:{time.time() - st}\t真实标签:{img_path}\t预测标签:{cls_name}\t预测概率:{p}')

    # 预测多张图片
    # folder_path = './dataset/test'
    # files = os.listdir(folder_path)
    # # 得到每个img文件地址
    # images_files = [os.path.join(folder_path, f) for f in files]
    # sum = 0
    # #
    # for img in images_files:
    #     true_label = img.split('\\')[-1].split('.')[0]
    #     # print(img)
    #     imgs = os.listdir(img)
    #     img_path = [os.path.join(img, f) for f in imgs]
    #     for img_path in img_path:
    #         # print(img_path)
    #         # print(img,true_label)
    #         image = Image.open(img_path)
    #         st = time.time()
    #         cls, p = recognizer.recognize(image, label_map=label_map)
    #         if true_label == cls:
    #             sum += 1
    #         print(f'推理时间:{time.time() - st}\t真实标签:{img_path}\t 预测标签:{cls}\t 预测概率:{p}')
    # print(f'正确数:{sum}')
    # print(f'准确率:{sum / 60}')
