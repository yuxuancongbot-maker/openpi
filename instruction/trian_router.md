# Router 训练指令

Router 是一个 3 层 MLP（2048→256→256→1 + Sigmoid），用于动态调度 L1 Flow 的推理步数（NFE=1 或 NFE=2），在不破坏 `torch.compile` 效率的前提下节省计算。

## 工作流

### 1. 收集数据

```bash
uv run scripts/collect_router_data.py \
    --checkpoint_dir /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/checkpoints/pi05_libero_l1flow_pytorch \
    --num_samples 2000 \
    --output router_data.npz
```

参数说明：
- `--config`：训练配置名，默认 `pi05_libero_l1_flow`
- `--checkpoint_dir`：checkpoint 目录（含 `model.safetensors` + `assets/`）
- `--num_samples`：采集样本数，默认 2000
- `--output`：输出 `.npz` 文件路径

输出文件包含：
- `prefix_feats`：`(N, 2048)` float32 — PaliGemma prefix 最后一层 hidden states 的 mean pooling
- `diffs`：`(N,)` float32 — `L1(actions_1step, actions_2step)`，即两步之间的差异

### 2. 查看分布，确定阈值

```bash
uv run scripts/train_router.py router_data.npz --plot histogram.png
```

生成直方图 + CDF 图，标注 P25/P50/P75/P90 分位点。根据图选择 P 值：

| P 值 | 样本走 2 步比例 | 平均 NFE |
|------|----------------|---------|
| P50  | 50% | 1.5 |
| P60  | 40% | 1.4 |
| P75  | 25% | 1.25 |
| P90  | 10% | 1.1 |

### 3. 训练 Router

```bash
# 使用百分位阈值（推荐）
uv run scripts/train_router.py router_data.npz --percentile 60 --save router_weights.pt

# 或使用固定阈值
uv run scripts/train_router.py router_data.npz --threshold 0.05 --save router_weights.pt
```

参数说明：
- `--percentile`：阈值百分位（与 `--threshold` 互斥）
- `--threshold`：固定阈值（与 `--percentile` 互斥）
- `--save`：输出权重路径，默认 `router_weights.pt`
- `--epochs`：训练轮数，默认 50
- `--lr`：学习率，默认 1e-4
- `--batch_size`：默认 256

输出：`router_weights.pt` — 可直接加载到模型的 `router` 模块。

### 4. 推理时加载

把 `router_weights.pt` 放到 checkpoint 目录下：

```bash
cp router_weights.pt /path/to/pi05_libero_l1flow_pytorch/
```

`serve_policy.py` 会自动检测并加载：

```bash
uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config=pi05_libero_l1_flow \
    --policy.dir=/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/checkpoints/pi05_libero_l1flow_pytorch
```

手动加载（Python API）：

```python
model.router.load_state_dict(torch.load("router_weights.pt", map_location="cpu"))
```

## Router 架构

```python
nn.Sequential(
    nn.Linear(2048, 256),   # PaliGemma hidden_size → 256
    nn.SiLU(),
    nn.Linear(256, 256),
    nn.SiLU(),
    nn.Linear(256, 1),
    nn.Sigmoid(),            # 输出 difficulty ∈ (0, 1)
)
```

推理时 `difficulty > 0.3` 走 2 步，否则走 1 步（阈值在训练时由百分位决定，训练脚本打完标后用 BCE loss 优化）。

## 关键设计

- **Router 输入免费**：`prefix_hidden` 来自语言模型 KV cache 填充时的 `output_hidden_states=True`，复用已算好的中间结果，0 额外 FLOPs
- **Python if 不破坏 compile**：决策在 tensor 计算之前完成，`torch.compile` 为 1步/2步分别缓存特化 CUDA graph
- **两个子函数静态图**：`_l1_1step` 和 `_l1_2step` 内部都是固定计算路径，无动态控制流
