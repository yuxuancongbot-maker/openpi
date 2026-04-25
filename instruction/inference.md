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
    --port 8000 \
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
'''
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
# Run the simulation
python examples/libero/main.py
'''