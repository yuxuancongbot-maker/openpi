"""Query a running OpenPI policy on saved Franka LeRobot frames.

This checks whether a checkpoint predicts gripper close on in-distribution
training frames. Start `scripts/serve_policy.py` first, then run this script.
"""

import argparse
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from openpi_client import image_tools
from openpi_client import websocket_client_policy


DEFAULT_DATASET = Path(
    "/home/funsun/franka_ros2_ws/src/lerobot_dataset/"
    "pick_up_the_blue_cube_and_put_it_onto_the_plate"
)
DEFAULT_PROMPT = "pick up the blue cube and put it onto the plate"


def read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    try:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for i, frame in enumerate(container.decode(stream)):
                if i == frame_index:
                    return frame.to_ndarray(format="rgb24")
    except Exception:
        pass

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_index} from {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def gripper_open_scalar(action_9d: np.ndarray) -> float:
    return float(np.clip((action_9d[7] + action_9d[8]) / 0.08, 0.0, 1.0))


def load_frame(dataset: Path, episode: int, frame: int) -> dict:
    ep_name = f"episode_{episode:06d}"
    parquet = dataset / "data" / "chunk-000" / f"{ep_name}.parquet"
    high_video = dataset / "videos" / "chunk-000" / "observation.images.cam_high" / f"{ep_name}.mp4"
    wrist_video = dataset / "videos" / "chunk-000" / "observation.images.cam_wrist" / f"{ep_name}.mp4"

    df = pd.read_parquet(parquet)
    if frame < 0 or frame >= len(df):
        raise ValueError(f"Frame {frame} out of range for {ep_name}: 0..{len(df) - 1}")

    row = df.iloc[frame]
    action = np.asarray(row["action"], dtype=np.float32)
    state = np.asarray(row["observation.state"], dtype=np.float32)
    high = read_video_frame(high_video, frame)
    wrist = read_video_frame(wrist_video, frame)
    return {
        "frame": frame,
        "state": state,
        "action": action,
        "gripper_open": gripper_open_scalar(action),
        "gripper_binary": int(gripper_open_scalar(action) >= 0.5),
        "high": high,
        "wrist": wrist,
    }


def default_frames(dataset: Path, episode: int) -> list[int]:
    parquet = dataset / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    df = pd.read_parquet(parquet, columns=["action"])
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    binary = ((actions[:, 7] + actions[:, 8]) / 0.08 >= 0.5).astype(np.int32)
    close = np.flatnonzero(binary == 0)
    if close.size == 0:
        return [0, len(binary) // 2, len(binary) - 1]
    first_close = int(close[0])
    return sorted(set([
        max(0, first_close - 20),
        max(0, first_close - 5),
        first_close,
        min(len(binary) - 1, first_close + 10),
        min(len(binary) - 1, first_close + 40),
    ]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    frames = args.frames if args.frames is not None else default_frames(args.dataset, args.episode)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    print(f"dataset: {args.dataset}")
    print(f"episode: {args.episode}")
    print(f"frames: {frames}")
    print(f"prompt: {args.prompt!r}")
    print()

    for frame in frames:
        sample = load_frame(args.dataset, args.episode, frame)
        obs = {
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(sample["high"], 224, 224)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(sample["wrist"], 224, 224)
            ),
            "observation/state": sample["state"],
            "prompt": args.prompt,
        }
        result = client.infer(obs)
        actions = np.asarray(result["actions"], dtype=np.float32)
        pred_g = actions[:, 7]
        print(
            f"frame={frame:04d} gt_open={sample['gripper_open']:.3f} "
            f"gt_bin={sample['gripper_binary']} pred_g={np.array2string(pred_g, precision=3)}"
        )


if __name__ == "__main__":
    main()
