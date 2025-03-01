import json
import random

# 文件路径
input_file_path = '/root/autodl-tmp/fossv_test/MetaMathQA/MetaMathQA-395K.json'
output_file_path = '/root/autodl-tmp/fossv_test/sampled_data.json'  # 输出文件路径

# 加载JSON文件
try:
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Error: 文件未找到，请检查路径是否正确：{input_file_path}")
    data = []
except json.JSONDecodeError:
    print("Error: JSON文件格式错误，无法解析。")
    data = []

# 检查数据是否成功加载
if not data:
    print("数据加载失败，无法继续操作。")
else:
    # 确保数据是一个列表
    if isinstance(data, list):
        # 随机抽取10条数据
        sample_size = min(10, len(data))  # 确保不会超出数据长度
        sampled_data = random.sample(data, sample_size)

        # 将抽取的数据保存为新的JSON文件
        try:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                json.dump(sampled_data, output_file, ensure_ascii=False, indent=4)
            print(f"成功生成JSON文件：{output_file_path}")
        except IOError:
            print(f"Error: 无法写入文件 {output_file_path}")
    else:
        print("Error: 数据不是列表格式，无法进行抽样操作。")