#!/usr/bin/env python3
"""Save raw and policy-preprocessed Franka camera observations for inspection."""

import argparse
from pathlib import Path
import time

import cv2
import numpy as np


def resize_224(img: np.ndarray) -> np.ndarray:
    from openpi_client import image_tools

    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def dump_images(out_dir: Path, high: np.ndarray, wrist: np.ndarray) -> None:
    save_rgb(out_dir / "cam_high_raw.png", high)
    save_rgb(out_dir / "cam_wrist_raw.png", wrist)
    save_rgb(out_dir / "cam_high_policy_224.png", resize_224(high))
    save_rgb(out_dir / "cam_wrist_policy_224.png", resize_224(wrist))


def dump_from_files(args: argparse.Namespace) -> None:
    high = cv2.imread(str(args.high_image), cv2.IMREAD_COLOR)
    wrist = cv2.imread(str(args.wrist_image), cv2.IMREAD_COLOR)
    if high is None:
        raise FileNotFoundError(args.high_image)
    if wrist is None:
        raise FileNotFoundError(args.wrist_image)
    dump_images(args.out_dir, cv2.cvtColor(high, cv2.COLOR_BGR2RGB), cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB))


def dump_from_ros(args: argparse.Namespace) -> None:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    class CameraSnapshot(Node):
        def __init__(self) -> None:
            super().__init__("dump_franka_obs_images")
            self.bridge = CvBridge()
            self.high = None
            self.wrist = None
            self.create_subscription(Image, args.cam_high_topic, self.high_callback, 10)
            self.create_subscription(Image, args.cam_wrist_topic, self.wrist_callback, 10)

        def high_callback(self, msg: Image) -> None:
            self.high = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

        def wrist_callback(self, msg: Image) -> None:
            self.wrist = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    rclpy.init(args=None)
    node = CameraSnapshot()
    deadline = time.time() + args.timeout
    try:
        while rclpy.ok() and time.time() < deadline:
            if node.high is not None and node.wrist is not None:
                dump_images(args.out_dir, np.array(node.high, copy=True), np.array(node.wrist, copy=True))
                return
            rclpy.spin_once(node, timeout_sec=0.05)
        raise RuntimeError("Timed out waiting for both camera images.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/franka_obs_images"))
    parser.add_argument("--high-image", type=Path, default=None, help="Optional raw cam_high image file.")
    parser.add_argument("--wrist-image", type=Path, default=None, help="Optional raw cam_wrist image file.")
    parser.add_argument("--cam-high-topic", default="/camera/cam_high/color/image_raw")
    parser.add_argument("--cam-wrist-topic", default="/camera/cam_wrist/color/image_raw")
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.high_image is None) != (args.wrist_image is None):
        raise ValueError("--high-image and --wrist-image must be provided together.")
    if args.high_image is not None:
        dump_from_files(args)
    else:
        dump_from_ros(args)
    print(f"Wrote raw and policy 224x224 images to {args.out_dir}")


if __name__ == "__main__":
    main()
