# Franka Continuous Gripper Training

This note is for the training-side agent/operator.

## Goal

Train a Franka policy that learns when to close the gripper from vision/state, instead of relying on deployment-time fixed-step grasp hacks.

The active config for the blue-cube-on-plate task is:

```text
pi05_pick_blue_cube_plate_droid_8d_continuous_gripper_low_mem
```

It uses pi0.5 with DROID base, LoRA low-memory variants, 8D Franka actions, and a continuous gripper target.

The old blue-cube binary configs still exist only for ablation:

```text
pi05_pick_blue_cube_plate_droid_8d_binary_gripper
pi05_pick_blue_cube_plate_droid_8d_binary_gripper_low_mem
```

Do not use them for the next main run. They threshold both gripper state and gripper action during training, which can make the model learn the shortcut `current gripper open -> keep open`.

For the green-plate task, use the newer action-binary variant when testing binary gripper again:

```text
pi05_green_plate_droid_8d_action_binary_gripper
pi05_green_plate_droid_8d_action_binary_gripper_low_mem
```

These configs keep `observation.state[7]` continuous, but train `action[7]` as binary open/closed. This is different from the older binary configs and is intended to avoid the `current gripper state -> copy same state` shortcut.

The older rag config still exists:

```text
pi05_pick_rag_100_droid_8d_binary_gripper_low_mem
```

Do not reuse the rag config for the blue cube task. It points at the rag dataset and uses the rag asset id / norm stats.

## What Changed

### 8D Franka gripper convention

`src/openpi/policies/franka_policy_8d.py` converts raw Franka 9D data:

```text
[7 arm joints, finger1, finger2]
```

to 8D:

```text
[7 arm joints, gripper_open_scalar]
```

where:

```text
gripper_open_scalar = clamp((finger1 + finger2) / 0.08, 0, 1)
0 = closed
1 = open
```

For the active continuous config, observation state and action gripper values stay continuous in `[0, 1]`. This matches the official DROID/OpenPI pattern: train on continuous gripper position, then binarize only when executing on robot hardware.

### Gripper target

The rag binary training config still uses:

```python
binary_gripper=True
gripper_open_threshold=0.3
```

Because `0.3 * 0.08m = 0.024m`, matching the empirically useful close threshold from the dataset/player.

This means:

```text
width >= 0.024m -> open label 1
width <  0.024m -> closed label 0
```

The blue cube dataset has a different measured gripper range. In the current 20-episode dataset:

```text
finger joint range: 0.0132m .. 0.0388m
width = finger1 + finger2: 0.0264m .. 0.0775m
width / 0.08: 0.330 .. 0.969
```

Therefore, `gripper_open_threshold=0.3` would label every blue-cube frame as open. The old blue-cube binary ablation used:

```python
binary_gripper=True
gripper_open_threshold=0.5
```

With this dataset, threshold `0.5` produces both open and closed labels, with two open/close transitions per episode.

The active blue-cube config now uses:

```python
binary_gripper=False
```

This means:

```text
training target: continuous gripper_open_scalar
robot execution: threshold the predicted scalar into open/close
```

### Action-binary green-plate ablation

The green-plate action-binary config uses:

```python
binary_gripper=True
binary_gripper_state=False
binary_gripper_action=True
gripper_open_threshold=0.5
```

This means:

```text
observation state gripper: continuous width scalar
training action gripper: binary open/closed label
execution: binary gripper threshold as before
```

Use this before collecting a large amount of new data. If it still predicts all-open when the robot is manually placed at a pre-grasp state, the issue is not just continuous-vs-binary; it is likely visual/state distribution or lack of transition-window samples.

### Gripper loss weighting

pi0.5 uses `action_dim=32`. Our Franka action is 8D and is padded to 32D, so the gripper is only 1 of 32 action dimensions.

The continuous gripper config keeps the same gripper loss weighting:

```python
action_loss_weights = (1.0,) * 7 + (8.0,) + (0.0,) * 24
```

Meaning:

```text
arm dims 0..6: weight 1
gripper dim 7: weight 8
padding dims 8..31: weight 0
```

The pi0 flow-matching MSE is now optionally weighted per action dimension. Other configs keep `action_loss_weights=None` and use the old loss path.

### Training diagnostics

When `action_loss_weights` is enabled, `scripts/train.py` logs:

```text
loss             weighted training loss
loss_unweighted  raw unweighted per-dim flow loss
loss_arm         raw flow loss on action dims 0..6
loss_gripper     raw flow loss on action dim 7
loss_padding     raw flow loss on padded dims 8..31
```

Watch `loss_gripper`. If it does not decrease, the model is still not learning the close/open decision.

## Compute Norm Stats

Run this on the machine that has the LeRobot dataset path referenced by the config:

```bash
cd /home/funsun/congyuxuan/franka/openpi

UV_CACHE_DIR=/tmp/uv-cache XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/compute_norm_stats.py \
  --config-name pi05_pick_blue_cube_plate_droid_8d_continuous_gripper_low_mem
```

Expected output directory:

```text
/home/funsun/congyuxuan/franka/openpi/assets/pi05_pick_blue_cube_plate_droid_8d_continuous_gripper_low_mem/pick_blue_cube_plate_continuous_gripper/norm_stats.json
```

If training on another machine, keep the same config and make sure the dataset path in `repo_id` exists there:

```text
/home/funsun/franka_ros2_ws/src/lerobot_dataset/pick_up_the_blue_cube_and_put_it_onto_the_plate
```

If the dataset lives at a different path on the training machine, update `repo_id` in `src/openpi/training/config.py` before computing norm stats.

Do not copy the rag norm stats into the blue cube config. The blue cube config must have its own `asset_id`:

```text
pick_blue_cube_plate_continuous_gripper
```

## Train

```bash
cd /home/funsun/congyuxuan/franka/openpi

UV_CACHE_DIR=/tmp/uv-cache XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
  pi05_pick_blue_cube_plate_droid_8d_continuous_gripper_low_mem \
  --exp_name=pick_blue_cube_plate_continuous_gripper_lora \
  --overwrite
```

## Serve After Training

Replace `/path/to/pick_blue_cube_plate_continuous_gripper_lora` with the produced checkpoint directory.

```bash
cd /home/funsun/congyuxuan/franka/openpi

UV_CACHE_DIR=/tmp/uv-cache XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/serve_policy.py \
  --port=8000 policy:checkpoint \
  --policy.config=pi05_pick_blue_cube_plate_droid_8d_continuous_gripper_low_mem \
  --policy.dir=/path/to/pick_blue_cube_plate_continuous_gripper_lora
```

## Deployment Note

The robot-side player should still use binary gripper execution. The policy predicts a continuous open scalar, and the player thresholds it into hardware open/close commands:

```bash
--action-dim 8 \
--binary-gripper \
--binary-gripper-threshold 0.7 \
--gripper-mode threshold \
--grasp-trend-delta 999
```

Do not use `--force-grasp-step` for final evaluation; that only verifies hardware grasp capability and does not test whether the policy learned grasp timing.
