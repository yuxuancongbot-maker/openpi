"""Query OpenPI policy on a saved Franka snapshot.

This script does not import ROS. Use it with the OpenPI uv environment after
capturing a snapshot with `capture_franka_policy_snapshot.py --no-policy`.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy


DEFAULT_PROMPT = "pick up the blue cube and put it onto the plate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_dir", type=Path)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    snapshot_path = args.snapshot_dir / "snapshot.npz"
    metadata_path = args.snapshot_dir / "metadata.json"
    data = np.load(snapshot_path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    prompt = args.prompt or metadata.get("prompt") or DEFAULT_PROMPT

    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    obs = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(data["observation_image"], 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(data["observation_wrist_image"], 224, 224)
        ),
        "observation/state": data["observation_state"],
        "prompt": prompt,
    }
    actions = np.asarray(client.infer(obs)["actions"], dtype=np.float32)
    np.save(args.snapshot_dir / "policy_actions.npy", actions)

    print(f"snapshot: {args.snapshot_dir}")
    print(f"prompt: {prompt!r}")
    print(f"actions shape: {actions.shape}")
    if actions.ndim == 2 and actions.shape[1] >= 8:
        print("predicted gripper:", np.array2string(actions[:, 7], precision=3))
    print(f"saved actions: {args.snapshot_dir / 'policy_actions.npy'}")


if __name__ == "__main__":
    main()
