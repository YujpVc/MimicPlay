import torch

# 加载模型
model_path = '/home/yujp/MimicPlay/trained_models_lowlevel/test/lowlevel_model_epoch_950.pth'
output_txt_path = '/home/yujp/MimicPlay/trained_models_lowlevel/test/lowlevel_model_epoch_950.txt'
content = torch.load(model_path, map_location='cpu')

# 检查 keys
print("Keys in the model:", content.keys())

# 检查是否包含路径信息
search_key = 'checkpoints/highlevel_model_epoch_357_best_validation_-65.05948219299316.pth'
contains_path = search_key in str(content)
print(f"Does the model contain the path '{search_key}'? {contains_path}")

# 如果 'model' 键存在，检查其内容
if 'model' in content:
    # print("Content under 'model':", content['model'])
    pass
else:
    print("No 'model' key found in the checkpoint.")
    pass

# 将内容转换为字符串并保存为 .txt 文件
with open(output_txt_path, 'w') as f:
    for key, value in content.items():
        f.write(f"Key: {key}\n")
        f.write(f"Value: {str(value)}\n\n")

print(f"模型内容已保存为文本文件：{output_txt_path}")