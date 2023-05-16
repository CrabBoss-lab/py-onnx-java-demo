import torch
from python.net import Net

# 加载PyTorch模型
model = Net()
weight = torch.load('./weight/model.pth')
model.load_state_dict(weight)

# 设置模型输入
dummy_input = torch.randn(1, 3, 224, 224)

# 导出ONNX模型
torch.onnx.export(model, dummy_input, 'weight/model.onnx', verbose=True)
