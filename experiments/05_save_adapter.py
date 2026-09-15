# ============================================================
# 05_save_adapter.py —— LoRA 只存外挂 vs 全参整模
# 目标：亲眼看见「adapter 只有几 MB，还能重新挂回去用」
#
# 运行：uv run python experiments/05_save_adapter.py
# ============================================================

import os
import shutil
import sys
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.insert(0, "data")
from train_data import TEST_PROMPTS, TRAIN_DATA

MODEL_NAME = "gpt2"
DEVICE = "mps"
MAX_NEW_TOKENS = 20
SAVE_DIR = "results/adapter"
FULL_MODEL_DIR = "results/full_model"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model


def build_dataloader(tokenizer, batch_size=4):
    data = [s + tokenizer.eos_token for s in TRAIN_DATA]

    def tokenize_fn(examples):
        enc = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        labels = []
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
            labels.append([tok if m == 1 else -100 for tok, m in zip(ids, mask)])
        enc["labels"] = labels
        return enc

    dataset = Dataset.from_dict({"text": data}).map(tokenize_fn, batched=True)
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def train_lora(tokenizer, model, epochs=20, lr=5e-4):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataloader = build_dataloader(tokenizer)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    model.train()
    start = time.time()
    for epoch in range(epochs):
        total, n = 0.0, 0
        for batch in dataloader:
            out = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
            n += 1
        if epoch == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1}/{epochs}  avg loss = {total/n:.4f}")
    elapsed = time.time() - start
    print(f"⏱ 训练耗时: {elapsed:.1f}s")
    return model


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)


def clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)


# ============================================================
print("=" * 50)
print("1. 训练一个 LoRA（20 epoch）")
print("=" * 50)

tokenizer, model = load_model()
model = train_lora(tokenizer, model)

print("\n【微调后 · 训练中的模型】")
model.eval()
for p in TEST_PROMPTS:
    print(f"  {generate(model, tokenizer, p)}")

# ============================================================
print("\n" + "=" * 50)
print("2. 只保存 LoRA adapter（外挂）")
print("=" * 50)

clear_dir(SAVE_DIR)
model.save_pretrained(SAVE_DIR)   # peft：只存 lora_A / lora_B 等
tokenizer.save_pretrained(SAVE_DIR)

adapter_mb = dir_size_mb(SAVE_DIR)
print(f"adapter 目录: {SAVE_DIR}")
print(f"adapter 体积: {adapter_mb:.2f} MB")
print("文件列表:")
for f in sorted(os.listdir(SAVE_DIR)):
    fp = os.path.join(SAVE_DIR, f)
    print(f"  {f:30s} {os.path.getsize(fp)/1024:8.1f} KB")

# ============================================================
print("\n" + "=" * 50)
print("3. 对比：adapter 体积 vs 原版 GPT-2 权重体积")
print("=" * 50)
print("全参微调才需要保存整模；LoRA 只存 adapter。")
print("这里拿「原版 GPT-2 权重」当分母，看 adapter 占多大比例。")

import glob

cache_hits = glob.glob(
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--openai--gpt2/**/model.safetensors"
    ),
    recursive=True,
)
if cache_hits:
    base_mb = os.path.getsize(cache_hits[0]) / (1024 * 1024)
    print(f"原版 GPT-2 权重体积: {base_mb:.2f} MB")
    print(f"LoRA adapter 体积:   {adapter_mb:.2f} MB")
    print(f"比例: adapter ≈ 原模型的 {adapter_mb/base_mb*100:.2f}%")
else:
    print("（未在缓存中找到 model.safetensors，跳过体积对比）")
    base_mb = None

# ============================================================
print("\n" + "=" * 50)
print("4. 重新加载：原版 GPT-2 + 只有几 MB 的 adapter")
print("=" * 50)

# 重新加载干净的 GPT-2，再挂上刚才存的 adapter
base_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
reloaded = PeftModel.from_pretrained(base_model, SAVE_DIR)
reloaded.eval()

print("重新加载后的输出（应与训练结束时一致）:")
for p in TEST_PROMPTS:
    print(f"  {generate(reloaded, base_tokenizer, p)}")

print("\n" + "=" * 50)
print("结论")
print("=" * 50)
print("1. 训练完只需保存 adapter（几 MB），不必保存整模")
print("2. 用时：加载原版 GPT-2 + 挂 adapter → 效果恢复")
print("3. 这就是多任务/多LoRA 能共享一个底座的原因")
if base_mb:
    print(f"4. 体积上：{adapter_mb:.1f}MB vs {base_mb:.1f}MB，约 1/{int(base_mb/adapter_mb)}")
