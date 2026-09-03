# ============================================================
# 03_train_lora.py —— 第一个最小 LoRA 实验
# 目标：GPT-2 + LoRA 微调，学会"情感标签格式"
# 运行：uv run python experiments/03_train_lora.py
#
# 学习方式：分步填代码（每步完成→运行→看结果→再下一步）
#   步骤 1：加载模型 + 打印微调前 baseline
#   步骤 2：LoraConfig + get_peft_model + 打印可训练参数
#   步骤 3：准备数据（tokenize + dataloader）
#   步骤 4：训练循环（前向 / 反向 / 更新）
#   步骤 5：微调后生成，和 baseline 对比
# ============================================================
# ↓↓↓ 每一步的代码，对照对话里给的"参考"，自己敲进来 ↓↓↓

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
sys.path.insert(0, "data")           # 让 Python 能找到 data/ 目录下的文件
from train_data import TEST_PROMPTS  # 从数据集文件导入测试句子

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
device = 'mps'
model = model.to(device)

# ③ 定义生成函数（和你昨天 01_baseline.py 里的一样）
def generate(prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ④ 打印微调前 baseline
print("【微调前 baseline】")
for p in TEST_PROMPTS:
    print(f"\n输入: {p}")
    print(f"输出: {generate(p)}")



from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
    bias="none",
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


from datasets import Dataset
from train_data import TRAIN_DATA

# 定义把文本变成张量的函数
def tokenize_fn(examples):
    enc = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=64)
    enc['labels'] = enc['input_ids'].copy()
    return enc

dataset = Dataset.from_dict({'text': TRAIN_DATA}).map(tokenize_fn, batched=True)
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)



optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)


model.train()

for epoch in range(5):
    total_loss = 0.0
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 前向：模型吃输入，预测下一个词
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,        # 告诉模型"答案是什么"，自动算 loss
        )
        loss = outputs.loss       # 误差（数字越小 = 预测越准）

        # 反向传播 + 更新参数（核心三连）
        loss.backward()           # ① 反向传播：算每个参数的梯度
        optimizer.step()          # ② 用梯度更新 LoRA 参数
        optimizer.zero_grad()     # ③ 清梯度，防止下一批累加

        total_loss += loss.item()
        print(f"[epoch {epoch+1}/5, step {step+1}/4] loss = {loss.item():.4f}")

    print(f"==> epoch {epoch+1} 平均 loss = {total_loss/4:.4f}")        