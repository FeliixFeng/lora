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
- LoRA 加在哪些 Transformer 层？（注意力投影层 q/k/v/o）
- 实际跑一次 LoRA 微调，比较不同 rank，看懂训练代码。

## 当前进度

- [x] 论文精读（Abstract / Introduction / 方法部分）
- [x] 环境搭建 + GPT-2 (124M) 离线加载
- [x] 微调前 baseline 输出（生成链路跑通）
- [x] 解码策略对比（temperature / top_k / top_p）
- [ ] **LoRA 微调**（最小实验）
- [ ] rank 对照实验（r = 4 / 8 / 16 / 32）
- [ ] PEFT 源码理解（y = Wx + BAx）
- [ ] 手写极简 LoRA Linear
- [ ] 完整实验记录

## 项目结构

```
lora/
├── experiments/
│   ├── 01_baseline.py          # 微调前 baseline（加载 GPT-2 + 生成）
│   └── 02_decode_compare.py    # 解码策略对比（temperature/top_k/top_p）
├── scripts/
│   └── download_model.py       # 国内镜像下载 GPT-2
├── data/                       # 数据集（LoRA 微调用）
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
```

## 关键发现（随实验持续更新）

- GPT-2 无任务引导时会机械复读（贪心解码的死循环），这是微调前 baseline 的典型行为
- 解码参数中：temperature 控随机度、top_k/top_p 截断候选集、生产常用组合是 temperature + top_p

## 后续计划

1. 造极小数据集，设计"固定输出格式"任务
2. 用 PEFT 挂 LoRA，观察 trainable parameters / loss / 微调前后输出
3. rank 对照实验，理解参数量与效果的权衡
4. 读 PEFT 源码中 LoRA 的 forward 实现
5. 手写极简 LoRA Linear，加深理解
6. 整理完整实验记录
