# ============================================================
# 00_full_ft.py —— 全参数微调（对照实验）
# 目标：先跑通「不外挂、所有参数都训」的流程，
#       再和 03_train_lora.py 对比，感受 LoRA 省在哪
#
# 运行：uv run python experiments/00_full_ft.py
#
# 流程和 03 完全同构：
#   加载 → baseline → 【差异：这里不挂 LoRA】→ 数据 → 训练 → 再生成对比
# ============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
import time
sys.path.insert(0, "data")
from train_data import TEST_PROMPTS, TRAIN_DATA

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "mps"
model = model.to(device)

def generate(prompt, max_new_tokens=20):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ① 微调前 baseline（和 03 一样）
print("【微调前 baseline】")
for p in TEST_PROMPTS:
    print(f"\n输入: {p}")
    print(f"输出: {generate(p)}")

# ② 全参数微调：这里没有 LoraConfig / get_peft_model
#    所有参数都可训练
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n【全参数微调】trainable: {trainable:,} / {total:,} = {trainable/total*100:.2f}%")
print("（对比 LoRA：大约 0.65%）")

# ③ 数据准备（和 03 相同）
from datasets import Dataset

# 每条样本末尾显式加 1 个 EOS：
# 教模型「Output 句子写完就停」，而不是学坏成「随时都想停」
TRAIN_DATA = [s + tokenizer.eos_token for s in TRAIN_DATA]

def tokenize_fn(examples):
    enc = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=64,
    )
    # 关键：填充位不参与 loss（labels=-100 表示忽略）
    # 否则模型会学一堆「预测 EOS」的假分，生成时容易说完就停
    labels = []
    for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
        labels.append([
            tok if m == 1 else -100
            for tok, m in zip(ids, mask)
        ])
    enc["labels"] = labels
    return enc

dataset = Dataset.from_dict({"text": TRAIN_DATA}).map(tokenize_fn, batched=True)
dataset.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

# ④ 优化器吃的是「全部参数」——这是全参贵的根源之一
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
# 注意：学习率比 LoRA 那版（5e-4）小 10 倍
# 全参动整盘，步子要更小，否则容易把预训练知识冲掉

model.train()

train_start = time.time()

for epoch in range(5):
    total_loss = 0.0
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        print(f"[epoch {epoch+1}/5, step {step+1}/4] loss = {loss.item():.4f}")

    print(f"==> epoch {epoch+1} 平均 loss = {total_loss/4:.4f}")

train_time = time.time() - train_start
print(f"\n⏱ 训练总耗时: {train_time:.1f}s")

# ⑤ 微调后再生成，和 baseline 对比
print("\n【微调后】")
model.eval()
for p in TEST_PROMPTS:
    print(f"\n输入: {p}")
    print(f"输出: {generate(p)}")
