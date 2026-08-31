# ============================================================
# 02_decode_compare.py —— 解码策略对比实验
# 目的：直观感受 temperature / top_k / top_p 对生成的影响
# 运行：uv run python experiments/02_decode_compare.py
# ============================================================

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ---------- 固定随机种子：保证采样结果可复现 ----------
torch.manual_seed(42)

# ---------- 加载模型（离线，命中本地缓存） ----------
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
device = "mps"
model = model.to(device)

def generate_with(prompt, max_new_tokens=40, **gen_kwargs):
    """按传入的生成参数生成一段文本"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, **gen_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def show(title, text):
    print(f"\n--- {title} ---")
    print(text)

prompt = "Once upon a time, there was"

# ---------- 1. 贪心解码（基线） ----------
show("① 贪心  do_sample=False（每步选概率最大的词）",
     generate_with(prompt))

# ---------- 2. temperature：控制随机度 ----------
show("② temperature=0.3（很低，接近贪心，很保守）",
     generate_with(prompt, do_sample=True, temperature=0.3))
show("③ temperature=1.0（默认，正常随机度）",
     generate_with(prompt, do_sample=True, temperature=1.0))
show("④ temperature=2.0（很高，非常发散）",
     generate_with(prompt, do_sample=True, temperature=2.0))

# ---------- 3. top_k：只从概率最高的 K 个词里选 ----------
show("⑤ top_k=5（只在前5个最可能的词里挑）",
     generate_with(prompt, do_sample=True, top_k=5))
show("⑥ top_k=50（前50个里挑，选择面更大）",
     generate_with(prompt, do_sample=True, top_k=50))

# ---------- 4. top_p（nucleus）：从累积概率达到 p 的词里选 ----------
show("⑦ top_p=0.5（只保留累积概率前50%的词）",
     generate_with(prompt, do_sample=True, top_p=0.5))
show("⑧ top_p=0.95（保留累积概率前95%，很宽松）",
     generate_with(prompt, do_sample=True, top_p=0.95))

# ---------- 5. 组合：temperature + top_p 一起用（生产环境常见） ----------
show("⑨ 组合  temperature=1.2 + top_p=0.9",
     generate_with(prompt, do_sample=True, temperature=1.2, top_p=0.9))
