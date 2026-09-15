# ============================================================
# 06_merge_unload.py —— 验证「推理无额外开销」
# 目标：把 BA 乘回 W0，拆掉外挂，再看输出是否不变
#
# 论文说法：部署时可把 LoRA 合并进原权重，
#           推理时和普通模型一样，没有旁路计算。
#
# 运行：uv run python experiments/06_merge_unload.py
# ============================================================

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "gpt2"
DEVICE = "mps"
ADAPTER_DIR = "results/adapter"
MAX_NEW_TOKENS = 20

TEST_PROMPTS = [
    "Input: The sunset is gorgeous.",
    "Input: I forgot my keys again.",
    "Input: This cake tastes amazing.",
]


def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ============================================================
print("=" * 50)
print("1. 加载：原版 GPT-2 + 已保存的 adapter")
print("=" * 50)
# 和上次第 4 步一样：干净底座 + 贴纸
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

print("merge 前（LoRA 旁路还在，两条路相加）:")
before_outputs = []
for p in TEST_PROMPTS:
    out = generate(model, tokenizer, p)
    before_outputs.append(out)
    print(f"  {out}")

# ============================================================
print("\n" + "=" * 50)
print("2. merge_and_unload：BA 乘回 W0，拆掉外挂")
print("=" * 50)
# 训练好的外挂：每层有 lora_A、lora_B、scaling(=α/r)
# merge 做的事（概念上，每层）：
#     W0 ← W0 + scaling * (B @ A)
# 然后把 A、B 从模型上拆掉
# unload 返回一个「普通」的 HuggingFace 模型（不再包着 LoRA）
merged_model = model.merge_and_unload()
merged_model.eval()

# 结构上应该已经不是 PeftModel 了
print(f"merge 后类型: {type(merged_model).__name__}")
print("（不再是 PeftModel → 没有 lora 旁路，就是一个普通 GPT2LMHeadModel）")

# ============================================================
print("\n" + "=" * 50)
print("3. merge 后再生成（应与 merge 前一致）")
print("=" * 50)

after_outputs = []
for p in TEST_PROMPTS:
    out = generate(merged_model, tokenizer, p)
    after_outputs.append(out)
    print(f"  {out}")

# ============================================================
print("\n" + "=" * 50)
print("4. 逐条对比")
print("=" * 50)
all_same = True
for i, (b, a) in enumerate(zip(before_outputs, after_outputs), 1):
    same = b == a
    all_same = all_same and same
    mark = "OK" if same else "DIFF"
    print(f"[{mark}] 第{i}条")
    if not same:
        print(f"  before: {b}")
        print(f"  after:  {a}")

print()
if all_same:
    print("结论：三条输出完全一致 → merge 无损，旁路已并进主干")
    print("      推理时只有一条矩阵乘路径，无额外开销")
else:
    print("结论：存在差异（贪心解码下少见，可能有数值误差）")
    print("      请检查 before/after 文本")

# ============================================================
print("\n" + "=" * 50)
print("原理小结")
print("=" * 50)
print("""
训练中：  h = W0·x + (α/r)·B·A·x     ← 两条路
merge 后： W' = W0 + (α/r)·B·A
          h = W'·x                    ← 一条路

因为矩阵乘对加法满足分配律：
  (W0 + ΔW)·x = W0·x + ΔW·x
所以「先加权重再乘」和「先乘再加」等价（理想浮点下）。
""")
