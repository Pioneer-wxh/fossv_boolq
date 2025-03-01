import torch
from datasets import Dataset
from transformers import AutoTokenizer
import numpy as np

# 加载数据集
dataset_path = '/root/autodl-tmp/fossv_test/boolq/validation-00000-of-00001.parquet'
dataset = Dataset.from_parquet(dataset_path)

# 只选取前10个样本进行测试
dataset = dataset.select(range(10))

# 使用AutoTokenizer从本地路径加载
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/fossv_test/Llama-2-7b-hf")

# 如果分词器没有定义pad_token，使用eos_token作为pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 使用eos_token作为pad_token

# 定义数据处理函数
def preprocess_boolq_data(examples, tokenizer):
    # 提取问题和段落文本
    questions = examples["question"]
    passages = examples["passage"]
    labels = examples["answer"]  # BoolQ 数据集中的答案标签（0 或 1）

    # 构造输入文本
    input_texts = [f"Question: {q}\nPassage: {p}" for q, p in zip(questions, passages)]

    # 使用分词器对输入文本进行编码
    inputs = tokenizer(
        input_texts,
        truncation=True,
        padding="max_length",  # 确保填充到最大长度
        max_length=256,        # 确保最大长度为 256
        return_tensors="pt",
    )

    # 将标签转换为张量
    labels = torch.tensor(labels, dtype=torch.long)

    # 返回处理后的输入和标签
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels,
    }

# 检查数据集的标签分布
labels = [sample["answer"] for sample in dataset]
print("Label Distribution:", np.bincount(labels))

# 应用数据预处理函数
processed_dataset = dataset.map(lambda examples: preprocess_boolq_data(examples, tokenizer), batched=True)

# 从数据集中移除原始字段
processed_dataset = processed_dataset.remove_columns(['question', 'passage', 'answer'])

# 主程序中检查每个字段是否是tensor类型
for sample in processed_dataset:
    print("Input IDs:", sample['input_ids'])
    print("Attention Mask:", sample['attention_mask'])
    print("Labels:", sample['labels'])
    break