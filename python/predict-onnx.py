import onnxruntime
import numpy as np
import cv2
from PIL import Image

from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode

from reset_transforms_resize import CV2_Resize


# 预处理图片
def preprocess(img):
    # img = img.resize((224, 224))
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
    # 0~255 ——》 0~1
    img = img.astype(np.float32)
    img = (img / 255. - 0.5) / 0.5  # torch style norm
    # c,h,w
    img = np.transpose(img, (2, 0, 1))
    # n,c,h,w
    img = np.expand_dims(img, axis=0)
    return img


def preprocessbytorch(img):
    # 定义图像预处理
    from torchvision import transforms
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # resize
        # Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
        CV2_Resize((224, 224), interpolation=cv2.INTER_LINEAR),
        transforms.ToTensor(),  # 转为tensor
        # transforms.Normalize(mean=[0.4737, 0.4948, 0.4336], std=[0.1920, 0.1592, 0.2184])  # 归一化
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    ])
    # 图像预处理并转换为形状为(1, C, H, W)的张量
    img_tensor = transform(img).unsqueeze(0)

    return img_tensor.numpy()


if __name__ == '__main__':
    # 加载模型
    model_path = 'weight/model.onnx'
    session = onnxruntime.InferenceSession(model_path)

    # 标签映射表
    label_map = ['番茄叶斑病', '苹果黑星病', '葡萄黑腐病']

    # 测试图片路径
    img_path = 'dataset/tomato25.JPG'
    print(img_path)
    # 打开测试图片 preprocess
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_tensor = preprocess(img)

    # 打开测试图片 preprocessbytorch
    img = Image.open(img_path)
    img_tensor=preprocessbytorch(img)

    # 进行预测
    outputs = session.run(None, input_feed={'input.1': img_tensor})  # 模型输出
    print(outputs)
    print(np.exp(outputs) / np.sum(np.exp(outputs)))
    output = outputs[0][0]
    pred_index = np.argmax(output)  # 最大值的索引
    pred_class = label_map[pred_index]
    pred_score = np.exp(output[pred_index]) / np.sum(np.exp(output))  # 转概率

    # 输出预测结果
    print(f'预测标签：{pred_class}')
    print(f'预测分数：{pred_score}')
