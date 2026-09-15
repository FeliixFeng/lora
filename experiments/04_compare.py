# ============================================================
# 04_compare.py —— 全参 vs LoRA 对照实验
# 一次跑完两边：同样数据、同样任务，只改「谁被训练」
#
# 运行：uv run python experiments/04_compare.py
# 调参：改下面 CONFIG 即可
# ============================================================

import sys
import time

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "data")
from train_data import TEST_PROMPTS, TRAIN_DATA

MODEL_NAME = "gpt2"
DEVICE = "mps"
MAX_NEW_TOKENS = 20

# ---------- 调参区：改这里就行 ----------
CONFIG = {
    "full_epochs": 20,
    "full_lr": 1e-4,
    "lora_epochs": 50,
    "lora_lr": 5e-4,
    "lora_r": 8,
    "lora_alpha": 16,
    "batch_size": 4,
}
# ----------------------------------------


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return tokenizer, model


def build_dataset(tokenizer, train_data, batch_size):
    # 每条样本末尾加 1 个 EOS（终点线）
    data = [s + tokenizer.eos_token for s in train_data]

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


def generate(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def train(model, dataloader, epochs, lr):
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=lr
    )
    model.train()
    start = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
                labels=batch["labels"].to(DEVICE),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            n += 1
        avg = total_loss / n
        # 轮数多时只打首尾和每 5 轮，避免刷屏
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{epochs}  avg loss = {avg:.4f}")
    elapsed = time.time() - start
    return elapsed


def evaluate(model, tokenizer, title):
    print(f"\n【{title} · 微调后】")
    model.eval()
    for p in TEST_PROMPTS:
        print(f"\n输入: {p}")
        print(f"输出: {generate(model, tokenizer, p)}")


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def run_baseline(tokenizer, model):
    print("【微调前 baseline】（两边相同，只打印一次）")
    model.eval()
    for p in TEST_PROMPTS:
        print(f"\n输入: {p}")
        print(f"输出: {generate(model, tokenizer, p)}")


def clear_mps():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main():
    print("=" * 50)
    print("本次 CONFIG")
    print("=" * 50)
    for k, v in CONFIG.items():
        print(f"  {k:14s} = {v}")
    print()

    # ----------------------------------------------------------------
    # A. 全参微调
    # ----------------------------------------------------------------
    print("=" * 50)
    print(f"A. 全参数微调  ({CONFIG['full_epochs']} epoch, lr={CONFIG['full_lr']})")
    print("=" * 50)

    tokenizer, model = load_model()
    run_baseline(tokenizer, model)

    trainable, total = count_params(model)
    print(f"\ntrainable: {trainable:,} / {total:,} = {trainable/total*100:.2f}%")

    dataloader = build_dataset(tokenizer, TRAIN_DATA, CONFIG["batch_size"])
    full_time = train(
        model, dataloader, epochs=CONFIG["full_epochs"], lr=CONFIG["full_lr"]
    )
    print(f"\n⏱ 全参训练总耗时: {full_time:.1f}s")
    evaluate(model, tokenizer, "全参")

    model = None
    clear_mps()

    # ----------------------------------------------------------------
    # B. LoRA 微调
    # ----------------------------------------------------------------
    from peft import LoraConfig, get_peft_model

    print("\n" + "=" * 50)
    print(
        f"B. LoRA 微调  ({CONFIG['lora_epochs']} epoch, "
        f"lr={CONFIG['lora_lr']}, r={CONFIG['lora_r']}, "
        f"alpha={CONFIG['lora_alpha']})"
    )
    print("=" * 50)

    tokenizer, model = load_model()

    lora_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = count_params(model)
    print(f"trainable: {trainable:,} / {total:,} = {trainable/total*100:.2f}%")

    dataloader = build_dataset(tokenizer, TRAIN_DATA, CONFIG["batch_size"])
    lora_time = train(
        model, dataloader, epochs=CONFIG["lora_epochs"], lr=CONFIG["lora_lr"]
    )
    print(f"\n⏱ LoRA 训练总耗时: {lora_time:.1f}s")
    evaluate(model, tokenizer, "LoRA")

    model = None
    clear_mps()

    # ----------------------------------------------------------------
    # C. 汇总
    # ----------------------------------------------------------------
    print("\n" + "=" * 50)
    print("C. 汇总对比")
    print("=" * 50)
    print(
        f"全参: trainable 100%,  {CONFIG['full_epochs']} epoch, "
        f"lr={CONFIG['full_lr']}, {full_time:.1f}s"
    )
    print(
        f"LoRA: trainable 0.65%, {CONFIG['lora_epochs']} epoch, "
        f"lr={CONFIG['lora_lr']}, r={CONFIG['lora_r']}, {lora_time:.1f}s"
    )
    print(f"时间比: LoRA / 全参 = {lora_time/full_time:.2f}x")
    print(
        f"轮数比: LoRA/全参 = "
        f"{CONFIG['lora_epochs']/CONFIG['full_epochs']:.2f}x"
    )
    print("（小数据 + 小模型下固定开销大，真实差距会更大）")


if __name__ == "__main__":
    main()
