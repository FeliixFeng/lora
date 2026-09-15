# LoRA 学习实验

从「看懂 LoRA」到「自己跑过一次 LoRA 实验」的完整学习记录。

基于论文 **《LoRA: Low-Rank Adaptation of Large Language Models》**（arXiv:2106.09685, Microsoft, 2021）。

## 学习目标

按照 **论文 → 问题 → 原理 → 代码 → 实验 → 观察结果 → 再理解** 的路径，真正理解并亲手验证 LoRA：

- LoRA 解决什么问题？（全量微调成本高、每任务一份完整模型）
- 普通 Fine-tuning 和 LoRA 有什么区别？（全量重训所有参数 vs 冻结权重 + 训练低秩旁路）
- `ΔW = BA` 是什么意思？（权重变化量 = 两个小矩阵的乘积）
- rank 是什么？（中间瓶颈维度，控制增量表达的信息量）
- 为什么训练参数大幅减少、原模型参数可以冻结？
- LoRA 加在哪些 Transformer 层？（GPT-2 是 c_attn / c_proj）
- 实际跑一次全参 vs LoRA 对照，看懂训练代码。

## 当前进度

- [x] 论文精读（Abstract / Introduction / 方法部分）
- [x] 环境搭建 + GPT-2 (124M) 离线加载
- [x] 微调前 baseline 输出（生成链路跑通）
- [x] 解码策略对比（temperature / top_k / top_p）
- [x] **全参数微调**（00_full_ft.py，5→20 epoch）
- [x] **LoRA 最小实验**（03_train_lora.py）
- [x] **全参 vs LoRA 对照**（04_compare.py，可调 CONFIG）
- [ ] PEFT 源码理解（y = Wx + BAx）
- [ ] 保存 adapter vs 全参权重，对比体积
- [ ] merge_and_unload 验证无推理开销
- [ ] 手写极简 LoRA Linear
- [ ] rank 对照实验（可选，小任务上收益有限）

## 项目结构

```
lora/
├── experiments/
│   ├── 00_full_ft.py           # 全参数微调（对照组）
│   ├── 01_baseline.py          # 微调前 baseline（加载 GPT-2 + 生成）
│   ├── 02_decode_compare.py    # 解码策略对比（temperature/top_k/top_p）
│   ├── 03_train_lora.py        # LoRA 微调（最小实验）
│   └── 04_compare.py           # 全参 vs LoRA 一次跑完，顶部 CONFIG 可调
├── data/
│   └── train_data.py           # 16 条情感标签格式样本 + 3 条测试句
├── scripts/
│   └── download_model.py       # 国内镜像下载 GPT-2
├── results/                    # 实验结果
├── pyproject.toml              # uv 依赖
└── README.md
```

## 环境

- Python 3.12（uv 虚拟环境）
- PyTorch 2.13 / transformers / peft / datasets / accelerate
- 硬件：Apple Silicon（MPS）
- 模型：GPT-2 (124M)，已缓存本地，完全离线运行

## 运行方法

```bash
uv run python experiments/01_baseline.py       # 微调前 baseline
uv run python experiments/02_decode_compare.py # 解码参数对比
uv run python experiments/00_full_ft.py        # 全参微调
uv run python experiments/03_train_lora.py     # LoRA 微调
uv run python experiments/04_compare.py        # 全参 vs LoRA 对照
```

## 任务设计

让 GPT-2 学会固定格式的情感标签：

```
Input: I love this movie.
Output: [POSITIVE] I love this movie.
```

- 训练：16 条英文样本（正/负各 8）
- 测试：3 条未见过的句子
- 每条训练样本末尾显式加 EOS，教模型「写完就停」

## 关键发现

### 解码与 baseline

- GPT-2 无任务引导时会机械复读（贪心解码死循环），这是微调前 baseline 的典型行为
- 解码参数中：temperature 控随机度、top_k/top_p 截断候选集

### 训练两个坑（踩过并修复）

1. **padding 不能进 labels**
   - 现象：loss 降得漂亮，但微调后在 Input 句末直接 EOS，格式学不会
   - 原因：样本 pad 到 64 后约一半位置是 EOS，模型被训练成「爱收工」
   - 修法：`attention_mask=0` 的位置 `labels=-100`，不参与 loss

2. **要在内容末尾显式加 1 个 EOS**
   - 只 mask 填充还不够：真内容里没有「终点线」，模型可能写完还继续甩 `Output:`
   - 修法：`TRAIN_DATA = [s + tokenizer.eos_token for s in TRAIN_DATA]`
   - 注意顺序：先加 EOS（算真内容），再 tokenize + padding

### 全参 vs LoRA 对照（2026-09）

| | 全参 | LoRA |
|--|------|------|
| 配置 | 20 epoch, lr=1e-4 | 50 epoch, lr=5e-4, r=8 |
| 可训练参数 | 100%（1.24 亿） | 0.65%（81 万） |
| 训练耗时 | 20.0s | 21.1s（多 2.5 倍轮数） |
| 等效 20 轮耗时 | 20.0s | ≈8.4s（约快 2.4 倍） |
| 测试句格式 | 3/3 正确 | 3/3 正确 |
| 测试句情感 | 3/3 正确 | 3/3 正确 |

结论：

- **0.65% 参数可以做到和全参一样的事**——LoRA 核心主张在小任务上成立
- **LoRA 不是免费的**：同样少轮数时更吃力（曾出现复读翻车），需要多训几轮
- **小模型 + 小数据上时间差不明显**：固定开销大；论文里的优势要在大模型上才爆出来
- **数据和训练量决定效果上限**：情感从全错到全对，靠的是 epoch，不是换方法

## 后续计划（下一阶段）

1. 读 PEFT 源码中 LoRA 的 forward（对应 `y = Wx + BAx`）
2. 训练后只保存 adapter，对比全参权重体积
3. `merge_and_unload()` 后验证输出不变（无推理延迟）
4. 手写极简 LoRA Linear，去掉框架黑盒
5. （可选）rank 对照；小任务上预期收益有限
