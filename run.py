import torch
import wandb


from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from config import FOSSVConfig, FOSSVTrainingArguments
from model import FOSSVModel




# 配置参数
MODEL_PATH = "/root/autodl-tmp/fossv_test/Llama-2-7b-hf"  # 本地模型路径
DATA_PATH = "/root/autodl-tmp/fossv_test/boolq"           # 数据集路径

# 数据集处理
print("Loading dataset...")
dataset = load_dataset("parquet", data_files={
    "train": f"{DATA_PATH}/train-00000-of-00001.parquet",
    "validation": f"{DATA_PATH}/validation-00000-of-00001.parquet"
})

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Validation dataset size: {len(dataset['validation'])}")

# 只选择前 500 条数据进行测试
train_dataset = dataset['train'] # 选取前500条训练数据
validation_dataset = dataset['validation']  # 选取前500条验证数据
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token# 将填充标记（pad_token）设置为结束标记（eos_token）
tokenizer.pad_token_id = tokenizer.eos_token_id# 这确保了在编码和解码过程中，填充标记和结束标记的ID是一致的

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

print("Processing train and validation datasets...")
train_dataset = train_dataset.map(lambda examples: preprocess_boolq_data(examples, tokenizer), batched=True, num_proc=1, load_from_cache_file=False, remove_columns=["question", "passage", "answer"])
validation_dataset = validation_dataset.map(lambda examples: preprocess_boolq_data(examples, tokenizer), batched=True, num_proc=1, load_from_cache_file=False, remove_columns=["question", "passage", "answer"])
print("Train labels:", np.unique(train_dataset["labels"], return_counts=True))
print("Validation labels:", np.unique(validation_dataset["labels"], return_counts=True))

# 添加调试代码，打印 logits 和 labels 的维度
def compute_metrics(p):
    logits, labels = p
    #print("Logits Shape:", logits.shape)  # 应为 (N, 2)
    #print("Labels Shape:", labels.shape)  # 应为 (N,)
    predictions = logits.argmax(dim=-1)
    accuracy = (predictions == labels).float().mean().item()
    return {"accuracy": accuracy}


print("Initializing model...")
fossv_config = FOSSVConfig(
    base_model_name_or_path=MODEL_PATH,
    rank=10,
    alpha=20,
    dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj"]
)

model = FOSSVModel(fossv_config)
model.model.config.pad_token_id = tokenizer.pad_token_id
model.config.keys_to_ignore_at_inference = ["past_key_values"]

model.fossv_init()

model.set_trainable(True)

# 修改 train.py 中的 training_args：
# 定义训练参数对象，设置各种训练相关配置
wandb.login(key="c1e2a295a7959d767565a41a2b31fb92ad713fc7")
run = wandb.init(
    project="test",
    job_type="training",
)
training_args = FOSSVTrainingArguments(
    output_dir="./results",  # 指定输出目录，存放训练结果和检查点
    evaluation_strategy="epoch",  # 设置评估策略为每个epoch结束时进行一次评估
    save_strategy="epoch",       # 每个epoch结束后保存模型检查点
    save_total_limit=1,          # 最多保留最近的一个检查点，节省存储空间
    learning_rate=3e-4,          # 学习率设为0.0003，控制权重更新幅度
    per_device_train_batch_size=4,  # 每个设备上的训练批次大小为2
    per_device_eval_batch_size=8,   # 每个设备上的评估批次大小为4
    num_train_epochs=3,         # 总共训练3个epoch，即整个训练集遍历3次
    weight_decay=0.01,          # 应用权重衰减正则化，系数为0.01，防止过拟合
    logging_steps=10,           # 每训练10步记录一次日志
    #logging_dir="./logs",       # 日志文件存放位置
    gradient_accumulation_steps=2,  # 设置梯度累积步数为2，模拟更大的批次大小
    remove_unused_columns=False,    # 不移除数据集中的未使用列
    report_to="wandb",           # 不向任何外部服务报告训练进度或结果
    fp16=False,                  # 启用FP16混合精度训练，减少内存占用并可能加速训练
    gamma=4e-4,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print("Creating Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

print("Starting training...")
# 修改 train.py 中的训练部分：
try:
    trainer.train()
    torch.cuda.empty_cache()  # 训练后清理显存
except Exception as e:
    print(f"Error during training: {e}")

model.merge(mode=True)  # 确保评估时权重已合并

# 保存模型
model.save_pretrained("./merged_model")


# 切换到评估模式
model.eval()

# 存储预测结果和真实标签
all_preds = []
all_labels = []

# 遍历验证数据集
with torch.no_grad():  # 禁用梯度计算
    for i in range(len(validation_dataset)):
        sample = validation_dataset[i]

        # 将输入数据移动到设备上
        input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)
        label = sample["labels"]

        # 进行前向传播
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 获取预测标签（通过 argmax 获取最大值索引）
        pred = torch.argmax(logits, dim=-1).item()

        # 保存预测和真实标签
        all_preds.append(pred)
        all_labels.append(label)

# 计算准确性
correct_predictions = sum([1 if p == l else 0 for p, l in zip(all_preds, all_labels)])
accuracy = correct_predictions / len(validation_dataset)

print(f"Validation Accuracy: {accuracy:.4f}")

