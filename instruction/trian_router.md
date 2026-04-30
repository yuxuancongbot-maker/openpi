# Router 训练指令

Router 是一个 3 层 MLP（2048→256→256→1 + Sigmoid），用于动态调度 L1 Flow 的推理步数（NFE=1 或 NFE=2），在不破坏 `torch.compile` 效率的前提下节省计算。

---

## LIBERO（原版）

### 1. 收集数据

```bash
uv run scripts/collect_router_data.py \
    --config pi05_libero_l1_flow \
    --checkpoint_dir /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/checkpoints/pi05_libero_l1flow_pytorch \
    --num_samples 2000 \
    --output router_data.npz
```

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

### 4. 推理时加载

```bash
cp router_weights.pt /path/to/pi05_libero_l1flow_pytorch/

uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config=pi05_libero_l1_flow \
    --policy.dir=/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/checkpoints/pi05_libero_l1flow_pytorch
```

---

## LIBERO-Plus

LIBERO-Plus 数据集格式与原版 LIBERO 有以下差异：

| 字段 | 原版 LIBERO | LIBERO-Plus |
|------|------------|-------------|
| 动作 key | `actions` | `action` |
| 前置图像 key | `image` | `observation.images.front` |
| 腕部图像 key | `wrist_image` | `observation.images.wrist` |
| 状态 key | `state` | `observation.state` |

已在 `config.py` 中新增 `LeRobotLiberoPlusDataConfig` 和 `pi05_libero_plus_l1_flow` 配置来处理这些差异。

### 前置步骤：JAX → PyTorch 转换

如果 checkpoint 是 JAX 格式（目录下有 `params/` 而非 `model.safetensors`），需要先转换：

```bash
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/checkpoints/pi05_libero_plus_l1_flow_from_ckpt/libero_plus_from_ckpt29999/15000 \
    --config_name pi05_libero_plus_l1_flow \
    --output_path /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/checkpoints/pi05_libero_plus_l1flow_pytorch
```

转换后需确保 `assets/` 目录下的 norm_stats 路径与 config 的 `asset_id` 匹配。如果不匹配（例如 `assets/data/libero_plus_lerobot/` vs `assets/physical-intelligence/libero/`），手动重命名：

```bash
cd /path/to/pytorch_checkpoint/assets/
mv data/libero_plus_lerobot physical-intelligence/libero
```

### 1. 收集数据

```bash
uv run scripts/collect_router_data.py \
    --config pi05_libero_plus_l1_flow \
    --checkpoint_dir /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/checkpoints/pi05_libero_plus_l1flow_pytorch \
    --data_dir /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/data/libero_plus_lerobot \
    --num_samples 2000 \
    --output router_data_liberoplus.npz
```

关键参数：
- `--config pi05_libero_plus_l1_flow`：使用 LIBERO-Plus 专用配置（处理 key 映射）
- `--data_dir`：指向本地 LIBERO-Plus LeRobot 数据集路径（覆盖 config 中的 `repo_id`）

### 2. 查看分布，确定阈值

```bash
uv run scripts/train_router.py router_data_liberoplus.npz --plot histogram_liberoplus.png
```

### 3. 训练 Router

```bash
uv run scripts/train_router.py router_data_liberoplus.npz --percentile 60 --save router_weights_liberoplus.pt
```

### 4. 推理时加载

```bash
cp router_weights_liberoplus.pt /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/checkpoints/pi05_libero_plus_l1flow_pytorch/router_weights.pt

uv run scripts/serve_policy.py --port 8000 \
    policy:checkpoint \
    --policy.config=pi05_libero_plus_l1_flow \
    --policy.dir=/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/checkpoints/pi05_libero_plus_l1flow_pytorch
```

---

## 参数说明

### collect_router_data.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `pi05_libero_l1_flow` | 训练配置名 |
| `--checkpoint_dir` | (必填) | checkpoint 目录（含 `model.safetensors` + `assets/`） |
| `--data_dir` | `None` | 本地 LeRobot 数据集路径（覆盖 config 中的 `repo_id`） |
| `--num_samples` | `2000` | 采集样本数 |
| `--output` | `router_data.npz` | 输出 `.npz` 文件路径 |

输出文件包含：
- `prefix_feats`：`(N, 2048)` float32 — PaliGemma prefix 最后一层 hidden states 的 mean pooling
- `diffs`：`(N,)` float32 — `L1(actions_1step, actions_2step)`，即两步之间的差异

### train_router.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--percentile` | - | 阈值百分位（与 `--threshold` 互斥） |
| `--threshold` | - | 固定阈值（与 `--percentile` 互斥） |
| `--save` | `router_weights.pt` | 输出权重路径 |
| `--epochs` | `50` | 训练轮数 |
| `--lr` | `1e-4` | 学习率 |
| `--batch_size` | `256` | 批大小 |

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
