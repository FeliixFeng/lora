"""一次性脚本：下载 GPT-2 到本地 HF 缓存。
仅下载模型时需要联网，按需选一种方式：
  - 镜像（国内推荐，快且稳）：uv run --env-file mirror.env python scripts/download_model.py
  - 代理（走本地 Clash）：uv run --env-file proxy.env python scripts/download_model.py
只下载训练/推理必需文件（跳过 tflite 等无关文件）。
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
from huggingface_hub import snapshot_download

p = snapshot_download(
    "gpt2",
    allow_patterns=[
        "model.safetensors",
        "*.json",   # config / tokenizer 配置
        "*.txt",    # vocab / merges
    ],
)
print("GPT-2 下载完成，缓存路径:", p)
