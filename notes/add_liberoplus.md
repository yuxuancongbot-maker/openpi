# LIBERO-Plus 环境配置与评测

## 概述

LIBERO-Plus 是 LIBERO 的扩展基准，在原有任务基础上增加了 7 个扰动维度（Camera、Robot、Language、Light、Background、Noise、Layout），用于测试 VLA 模型的鲁棒性。

- GitHub: https://github.com/sylvestf/LIBERO-plus
- 论文: arXiv:2510.13626

## 安装

### 1. 添加 submodule

```bash
cd third_party
git submodule add https://github.com/sylvestf/LIBERO-plus.git
```

### 2. 下载 assets

assets.zip 约 6GB，解压后约 15GB，用 wget 直接下载 HuggingFace：

```bash
wget https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip \
    -O /tmp/assets.zip

cd third_party/LIBERO-plus/libero/libero

# 解压（注意：zip 内部路径很深）
unzip -o /tmp/assets.zip

# 从深层路径把 assets 移到正确位置
mv inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/release/dataset/LIBERO-plus-0/assets .
rm -rf inspire

rm /tmp/assets.zip
```

**最终 assets 位置**: `third_party/LIBERO-plus/libero/libero/assets/`

### 3. 安装系统依赖

```bash
apt install -y libexpat1 libfontconfig1-dev libpython3-stdlib libmagickwand-dev
```

**注意**：不需要 `pip install` LIBERO-Plus。通过 PYTHONPATH 切换加载，避免与已安装的原版 LIBERO 冲突。

## 环境配置

### 创建 LIBERO 配置文件

```bash
mkdir -p ~/.libero

cat > ~/.libero/config.yaml << 'EOF'
benchmark_root: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/openpi/third_party/LIBERO-plus/libero/libero
bddl_files: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/openpi/third_party/LIBERO-plus/libero/libero/bddl_files
init_states: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/openpi/third_party/LIBERO-plus/libero/libero/init_files
datasets: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/openpi/third_party/LIBERO-plus/libero/datasets
assets: /inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/openpi/third_party/LIBERO-plus/libero/libero/assets
EOF
```

## PYTHONPATH 切换方案（LIBERO ↔ LIBERO-Plus）

LIBERO 和 LIBERO-Plus 的 Python 包名都是 `libero`，不能同时存在。通过 PYTHONPATH 切换：

### 切到 LIBERO-Plus

```bash
export PYTHONPATH=$(pwd)/third_party/LIBERO-plus:$PYTHONPATH
export LIBERO_CONFIG_PATH=~/.libero
export PYOPENGL_PLATFORM=egl
```

### 切回 LIBERO

```bash
export PYTHONPATH=$(pwd)/third_party/libero:$PYTHONPATH
unset LIBERO_CONFIG_PATH
```

### 快速切换函数（可选，添加到 ~/.bashrc）

```bash
use_libero() {
    export PYTHONPATH=$(pwd)/third_party/libero:$PYTHONPATH
    unset LIBERO_CONFIG_PATH
}
use_libero_plus() {
    export PYTHONPATH=$(pwd)/third_party/LIBERO-plus:$PYTHONPATH
    export LIBERO_CONFIG_PATH=~/.libero
    export PYOPENGL_PLATFORM=egl
}
```

## LIBERO-Plus Benchmark

| Benchmark | 任务数 | 说明 |
|-----------|--------|------|
| `libero_spatial` | 2402 | 空间关系（含扰动变体） |
| `libero_object` | 2518 | 物体关系（含扰动变体） |
| `libero_goal` | 2591 | 目标任务（含扰动变体） |
| `libero_90` | 90 | 预训练任务（含扰动变体） |
| `libero_10` | 2519 | 测试任务（含扰动变体） |
| `libero_100` | 100 | 100 个独立任务 |
| `libero_mix` | - | 混合扰动任务 |

**注意**：LIBERO-Plus 的 `libero_spatial`/`libero_object`/`libero_goal` 和原版 LIBERO 不同——原版每个 30 个纯净任务，LIBERO-Plus 每个有数千个带扰动的变体。

## 评测

### 启动策略服务

```bash
uv run scripts/serve_policy.py --port 8002 \
    policy:checkpoint \
    --policy.config=pi05_libero_l1_flow \
    --policy.dir=/path/to/checkpoint
```

### 运行评测

```bash
# 设置环境
export PYTHONPATH=$(pwd)/third_party/LIBERO-plus:$PYTHONPATH
export LIBERO_CONFIG_PATH=~/.libero
export PYOPENGL_PLATFORM=egl

# 注意：LIBERO-Plus 每个任务只有 1 个初始状态，num-trials-per-task 设为 1
python examples/libero/main.py \
    --args.task-suite-name libero_100 \
    --args.num-trials-per-task 1 \
    --args.host 0.0.0.0 \
    --args.port 8002
```
