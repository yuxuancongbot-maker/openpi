import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Franka 8D policy."""
    return {
        "observation/state": np.random.rand(8),  # 7 joints + 1 gripper scalar
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _gripper_9d_to_open_scalar(values: np.ndarray) -> np.ndarray:
    """Convert Franka two-finger state/action to one normalized open scalar.

    Input convention in our HDF5/LeRobot data:
      - dims 7 and 8 are the two finger joint positions in meters.
      - each finger is roughly [0.0, 0.04], so total width is [0.0, 0.08].

    Output follows the openpi-franka convention:
      - 0.0 = closed
      - 1.0 = open
    """
    width = values[..., 7] + values[..., 8]
    return np.clip(width / 0.08, 0.0, 1.0)


def _to_8d(
    values: np.ndarray,
    *,
    binary_gripper: bool = False,
    gripper_open_threshold: float = 0.5,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-1] == 8:
        result = values.astype(np.float32, copy=True)
        if binary_gripper:
            result[..., 7] = (result[..., 7] >= gripper_open_threshold).astype(np.float32)
        return result
    if values.shape[-1] < 9:
        raise ValueError(f"Expected Franka state/action dim 8 or 9, got {values.shape[-1]}")
    gripper = _gripper_9d_to_open_scalar(values)[..., None]
    if binary_gripper:
        gripper = (gripper >= gripper_open_threshold).astype(np.float32)
    return np.concatenate([values[..., :7], gripper], axis=-1).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class Franka8DInputs(transforms.DataTransformFn):
    """Map Franka observations to pi0.5 input format using 8D proprioception.

    The input data may be either:
      - 9D: [7 joints, finger1, finger2] from our current HDF5/LeRobot data, or
      - 8D: [7 joints, gripper_open_scalar].

    This transform collapses 9D -> 8D for both state and actions during training,
    while accepting 8D directly during inference.
    """

    model_type: _model.ModelType
    binary_gripper: bool = False
    binary_gripper_state: bool = True
    binary_gripper_action: bool = True
    gripper_open_threshold: float = 0.5

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        binary_state = self.binary_gripper and self.binary_gripper_state
        binary_action = self.binary_gripper and self.binary_gripper_action

        inputs = {
            "state": _to_8d(
                data["observation/state"],
                binary_gripper=binary_state,
                gripper_open_threshold=self.gripper_open_threshold,
            ),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = _to_8d(
                data["actions"],
                binary_gripper=binary_action,
                gripper_open_threshold=self.gripper_open_threshold,
            )
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class Franka8DOutputs(transforms.DataTransformFn):
    """Slice model output to [7 joints, 1 gripper_open_scalar]."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
