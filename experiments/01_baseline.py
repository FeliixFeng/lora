# ============================================================
# 01_baseline.py —— 第一步：微调前的"基准"输出
#
# 目标：
#   1. 加载 GPT-2（从本地缓存，不联网）
#   2. 打印模型总参数量
#   3. 对几个输入做生成，记录"微调前"模型的输出
#
# 作用：
#   这是后面所有对比的起点 —— 微调前模型长什么样，
#   后面 LoRA 微调完再跑一遍，就能看到行为差异。
#
# 运行：uv run python experiments/01_baseline.py
# ============================================================
# ↓↓↓ 代码留给你敲（对照对话里的带注释版）↓↓↓

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 1. 加载模型和分词器
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = 'mps'
model = model.to(device)

# 2. 打印参数总量
total = sum(p.numel() for p in model.parameters())
print(f'参数总量：{total:,} ({total/1e6:.1f}M)')


def generate(prompt, max_new_tokens=30):
  inputs = tokenizer(prompt, return_tensors='pt').to(device)

  with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

  return tokenizer.decode(outputs[0], skip_special_tokens=True)



test_prompts = [
    "I love this movie because",
    "The weather today is",
]
for p in test_prompts:
    print(f"\n[输入] {p}")
    print(f"[输出] {generate(p)}")

