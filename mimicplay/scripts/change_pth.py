import torch
import json

# 加载 .pth 文件
file_path = "/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/Trained_model_0610/lowlevel/test/20250609070322/models/model_epoch_1052_best_validation_-43.54394798278808.pth"
data = torch.load(file_path)

# 检查 'config' 的类型
if 'config' in data and isinstance(data['config'], str):
    # 将 JSON 字符串解析为字典
    config_dict = json.loads(data['config'])

    # 修改 `trained_highlevel_planner` 的值
    config_dict['algo']['lowlevel']['trained_highlevel_planner'] = "/home/yujp/Genesis/scripts/XBOX_control/dataCollect/demo/Trained_model_0610/highlevel/test/20250608200627/models/model_epoch_101_best_validation_-68.45099143981933.pth"

    # 将字典转换回 JSON 字符串
    data['config'] = json.dumps(config_dict, indent=4)

    # 保存修改后的文件
    new_file_path = file_path
    torch.save(data, new_file_path)
    print(f"文件已修改并保存到: {new_file_path}")
else:
    print("'config' 不存在或不是字符串，无法处理。")
