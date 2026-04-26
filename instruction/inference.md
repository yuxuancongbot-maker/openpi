# 推理指令

## Python API 推理

```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

config = _config.get_config("pi05_droid")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_droid")

policy = policy_config.create_trained_policy(config, checkpoint_dir)

example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    "prompt": "pick up the fork",
}
action_chunk = policy.infer(example)["actions"]
```

## Policy Server 推理

```bash
# 默认策略（注意环境名大写）
uv run scripts/serve_policy.py --env LIBERO

# 指定 checkpoint
uv run scripts/serve_policy.py \
    --port 8002 \
    policy:checkpoint \
    --policy.config=pi05_libero_l1_flow \
    --policy.dir=/inspire/hdd/project/inference-chip/lijinhao-240108540148/research_yuxuancong/onestep_pi/openpi/checkpoints/pi05_libero_l1flow_pytorch
```

## 可用配置

| 配置名 | 适用平台 |
|--------|----------|
| `pi0_aloha` / `pi05_aloha` | ALOHA |
| `pi0_droid` / `pi0_fast_droid` / `pi05_droid` | DROID |
| `pi05_libero` | LIBERO |
| `pi0_aloha_sim` | ALOHA 仿真 |
| `pi05_libero_l1_flow` | LIBERO + L1 Flow |

## 客户端调用

```python
from openpi_client.websocket_client_policy import WebsocketClientPolicy

client = WebsocketClientPolicy("ws://localhost:8000")
result = client.infer(obs)
actions = result["actions"]
```
                                   


### LiBERO 评测                                                               
                                                                                        
启动策略服务后（见 Policy Server 推理），在另一个终端运行：                                  
                                                                                        
```bash
# 先激活 libero 环境（需设置 PYTHONPATH）
export PYTHONPATH=$PYTHONPATH:$(pwd)/third_party/libero

# 指定子测试集、每个任务测 10 次
python examples/libero/main.py \
    --args.task-suite-name libero_object \
    --args.num-trials-per-task 10 \
    --args.host 0.0.0.0 \
    --args.port 8002
```

可用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--args.task-suite-name` | `libero_spatial` | 子测试集：`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`, `libero_90` |
| `--args.num-trials-per-task` | `50` | 每个 task 的测试次数 |
| `--args.host` | `0.0.0.0` | 策略服务器地址 |
| `--args.port` | `8000` | 策略服务器端口 |
| `--args.num-steps-wait` | `10` | 环境启动后等待步数 |
| `--args.replan-steps` | `5` | 每次规划后执行的步数 |
| `--args.seed` | `7` | 随机种子 |

### LIBERO-Plus 评测

LIBERO-Plus 是 LIBERO 的扩展基准，增加了 7 个扰动维度（Camera、Robot、Language、Light、Background、Noise、Layout）。

通过 PYTHONPATH 在 LIBERO 和 LIBERO-Plus 之间切换：

```bash
# 切到 LIBERO-Plus
export PYTHONPATH=$(pwd)/third_party/LIBERO-plus:$PYTHONPATH
export LIBERO_CONFIG_PATH=/tmp/libero
export PYOPENGL_PLATFORM=egl

# 切回 LIBERO
export PYTHONPATH=$(pwd)/third_party/libero:$PYTHONPATH
unset LIBERO_CONFIG_PATH
```

启动策略服务后，运行评测：

```bash
# LIBERO-Plus 每个任务只有 1 个初始状态，num-trials-per-task 设为 1
python examples/libero/main.py \
    --args.task-suite-name libero_100 \
    --args.num-trials-per-task 1 \
    --args.host 0.0.0.0 \
    --args.port 8002
```

可用 benchmark：

| 参数值 | 说明 |
|--------|------|
| `libero_100` | 100 个独立任务 |
| `libero_mix` | LIBERO-Plus 混合扰动任务 |
| `libero_spatial` | 2402 个空间关系任务（含扰动） |
| `libero_object` | 2518 个物体关系任务（含扰动） |
| `libero_goal` | 2591 个目标任务（含扰动） |

详细配置见 `notes/add_liberoplus.md`。