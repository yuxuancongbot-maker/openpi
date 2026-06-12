"""Capture one live Franka OpenPI observation and optionally query policy.

Run this when the robot is at a state where the gripper should close but the
policy keeps predicting open. The output can be moved to another machine for
offline inspection.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["fr3_finger_joint1", "fr3_finger_joint2"]
STATE_JOINT_NAMES = ARM_JOINT_NAMES + FINGER_JOINT_NAMES
DEFAULT_PROMPT = "pick up the blue cube and put it onto the plate"


def joint_state_to_qpos_9d(msg: JointState) -> np.ndarray:
    positions = list(msg.position)
    if msg.name:
        by_name = dict(zip(msg.name, positions))
        if all(name in by_name for name in STATE_JOINT_NAMES):
            return np.array([by_name[name] for name in STATE_JOINT_NAMES], dtype=np.float32)
    if len(positions) < 9:
        raise ValueError(f"JointState has {len(positions)} positions, need at least 9.")
    return np.array(positions[:9], dtype=np.float32)


def qpos9_to_open_scalar(qpos: np.ndarray) -> float:
    return float(np.clip((qpos[7] + qpos[8]) / 0.08, 0.0, 1.0))


class SnapshotNode(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("franka_policy_snapshot")
        self.args = args
        self.bridge = CvBridge()
        self.state = None
        self.high = None
        self.wrist = None
        self.state_stamp = None
        self.high_stamp = None
        self.wrist_stamp = None

        self.create_subscription(JointState, args.joint_topic, self.joint_callback, 10)
        self.create_subscription(Image, args.cam_high_topic, self.high_callback, 10)
        self.create_subscription(Image, args.cam_wrist_topic, self.wrist_callback, 10)

    def joint_callback(self, msg: JointState) -> None:
        try:
            self.state = joint_state_to_qpos_9d(msg)
            self.state_stamp = self.get_clock().now().nanoseconds
        except Exception as exc:
            self.get_logger().error(f"JointState error: {exc}")

    def high_callback(self, msg: Image) -> None:
        self.high = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.high_stamp = self.get_clock().now().nanoseconds

    def wrist_callback(self, msg: Image) -> None:
        self.wrist = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.wrist_stamp = self.get_clock().now().nanoseconds

    def ready(self) -> bool:
        return self.state is not None and self.high is not None and self.wrist is not None


def query_policy(args: argparse.Namespace, state: np.ndarray, high: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    try:
        from openpi_client import image_tools
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise RuntimeError("openpi_client is not importable in this environment.") from exc
    client = websocket_client_policy.WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    obs = {
        "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(high, 224, 224)),
        "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224)),
        "observation/state": state,
        "prompt": args.prompt,
    }
    return np.asarray(client.infer(obs)["actions"], dtype=np.float32)


def save_snapshot(args: argparse.Namespace, node: SnapshotNode, actions: np.ndarray | None) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"franka_policy_snapshot_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    high = np.array(node.high, copy=True)
    wrist = np.array(node.wrist, copy=True)
    state = np.array(node.state, copy=True)

    cv2.imwrite(str(out_dir / "cam_high.png"), cv2.cvtColor(high, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "cam_wrist.png"), cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR))
    np.savez_compressed(
        out_dir / "snapshot.npz",
        observation_state=state,
        observation_image=high,
        observation_wrist_image=wrist,
        actions=actions if actions is not None else np.empty((0,), dtype=np.float32),
    )

    metadata = {
        "prompt": args.prompt,
        "joint_topic": args.joint_topic,
        "cam_high_topic": args.cam_high_topic,
        "cam_wrist_topic": args.cam_wrist_topic,
        "state": state.tolist(),
        "state_gripper_open_scalar": qpos9_to_open_scalar(state),
        "policy_host": args.policy_host,
        "policy_port": args.policy_port,
        "queried_policy": actions is not None,
    }
    if actions is not None and actions.ndim == 2 and actions.shape[1] >= 8:
        metadata["predicted_gripper"] = actions[:, 7].tolist()
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--cam-high-topic", default="/camera/cam_high/color/image_raw")
    parser.add_argument("--cam-wrist-topic", default="/camera/cam_wrist/color/image_raw")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/franka_policy_snapshots"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--no-policy", action="store_true", help="Only save observation; do not query policy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = SnapshotNode(args)
    deadline = time.time() + args.timeout
    while rclpy.ok() and time.time() < deadline and not node.ready():
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.ready():
        node.destroy_node()
        rclpy.shutdown()
        raise RuntimeError("Timed out waiting for state + high/wrist RGB images.")

    actions = None
    if not args.no_policy:
        actions = query_policy(args, node.state, node.high, node.wrist)

    out_dir = save_snapshot(args, node, actions)
    print(f"saved: {out_dir}")
    print(f"state gripper open scalar: {qpos9_to_open_scalar(node.state):.3f}")
    if actions is not None and actions.ndim == 2 and actions.shape[1] >= 8:
        print("predicted gripper:", np.array2string(actions[:, 7], precision=3))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
