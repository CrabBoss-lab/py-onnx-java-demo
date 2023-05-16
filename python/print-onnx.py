import onnxruntime

# 加载模型
model_path = 'weight/model.onnx'
session = onnxruntime.InferenceSession(model_path)

# 获取输入名称和形状
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

# 获取输出名称和形状
output_name = session.get_outputs()[0].name
output_shape = session.get_outputs()[0].shape

# 输出模型的输入和输出信息
print(f'Model input name: {input_name}, shape: {input_shape}')
print(f'Model output name: {output_name}, shape: {output_shape}')