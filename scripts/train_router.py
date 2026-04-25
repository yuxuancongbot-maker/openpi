#!/usr/bin/env python3
"""Train Router (3-layer MLP) from collected prefix features and diffs.

Usage:
    # 1. View diff distribution and pick threshold percentile
    uv run scripts/train_router.py router_data.npz --plot histogram.png

    # 2. Train with P60 threshold (60% → 1-step, 40% → 2-step, avg NFE ≈ 1.4)
    uv run scripts/train_router.py router_data.npz --percentile 60 --save router_weights.pt

    # 3. Train with specific threshold
    uv run scripts/train_router.py router_data.npz --threshold 0.05 --save router_weights.pt
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch import optim


def build_router(input_dim: int) -> nn.Sequential:
    """3-layer MLP matching pi0_pytorch.py's router architecture."""
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )


@torch.no_grad()
def compute_accuracy(router, feats, labels):
    preds = router(feats).squeeze(-1)
    acc = ((preds > 0.5) == labels).float().mean().item()
    return acc


def main():
    parser = argparse.ArgumentParser(description="Train Router from collected data")
    parser.add_argument("data", type=Path, help="Path to .npz file from collect_router_data.py")
    parser.add_argument("--threshold", type=float, default=None, help="Fixed diff threshold (mutually exclusive with --percentile)")
    parser.add_argument("--percentile", type=float, default=None, help="Threshold percentile, e.g. 60 (mutually exclusive with --threshold)")
    parser.add_argument("--plot", type=str, default=None, help="Save histogram to this path and exit (no training)")
    parser.add_argument("--save", type=str, default="router_weights.pt", help="Output weights path")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load data
    data = np.load(args.data)
    X = torch.from_numpy(data["prefix_feats"]).float()   # (N, 2048)
    diffs = data["diffs"]                                 # (N,)

    print(f"Loaded {len(X)} samples, prefix_feat dim={X.shape[1]}")
    print(f"Diff stats: mean={diffs.mean():.6f}, median={np.median(diffs):.6f}, "
          f"min={diffs.min():.6f}, max={diffs.max():.6f}")

    # Plot mode
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.hist(diffs, bins=100, alpha=0.7)
        ax1.axvline(np.median(diffs), color="r", linestyle="--", label=f"Median={np.median(diffs):.4f}")
        for p in [25, 50, 75, 90]:
            ax1.axvline(np.percentile(diffs, p), color="gray", linestyle=":", alpha=0.5)
            ax1.text(np.percentile(diffs, p), ax1.get_ylim()[1] * 0.9, f"P{p}", fontsize=8, rotation=90)
        ax1.set_xlabel("L1 diff between 1-step and 2-step")
        ax1.set_ylabel("Count")
        ax1.set_title("Diff distribution")
        ax1.legend()

        ax2.hist(diffs, bins=100, cumulative=True, density=True, alpha=0.7, histtype="step", linewidth=2)
        for p in [25, 50, 75, 90]:
            ax2.axhline(p / 100, color="gray", linestyle=":", alpha=0.5)
            ax2.axvline(np.percentile(diffs, p), color="gray", linestyle=":", alpha=0.5)
            ax2.text(np.percentile(diffs, p) + 0.001, p / 100 + 0.02, f"P{p}={np.percentile(diffs, p):.4f}", fontsize=8)
        ax2.set_xlabel("L1 diff")
        ax2.set_ylabel("Cumulative fraction")
        ax2.set_title("CDF")

        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"Histogram saved to {args.plot}")
        return

    # Determine threshold
    if args.threshold is not None and args.percentile is not None:
        raise ValueError("Specify only one of --threshold or --percentile")
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Using fixed threshold: {threshold:.6f}")
    elif args.percentile is not None:
        threshold = float(np.percentile(diffs, args.percentile))
        print(f"Using P{args.percentile} threshold: {threshold:.6f}")
    else:
        threshold = float(np.percentile(diffs, 50))
        print(f"Defaulting to P50 (median) threshold: {threshold:.6f}")

    # Create labels
    labels = torch.from_numpy((diffs > threshold).astype(np.float32))  # (N,)
    ratio = labels.mean().item()
    print(f"Label ratio (need 2-step): {ratio*100:.1f}% → avg NFE = {1 + ratio:.3f}")

    # Build router
    torch.manual_seed(args.seed)
    router = build_router(X.shape[1])
    print(f"Router params: {sum(p.numel() for p in router.parameters()):,}")

    # Split 80/20
    perm = torch.randperm(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = perm[:split], perm[split:]
    X_train, y_train = X[train_idx], labels[train_idx]
    X_val, y_val = X[val_idx], labels[val_idx]

    # Train
    opt = optim.AdamW(router.parameters(), lr=args.lr)
    best_acc = 0.0

    for epoch in range(args.epochs):
        router.train()
        perm_epoch = torch.randperm(len(X_train))
        total_loss = 0.0
        num_batches = 0

        for start in range(0, len(X_train), args.batch_size):
            idx = perm_epoch[start:start + args.batch_size]
            pred = router(X_train[idx]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy(pred, y_train[idx])

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            num_batches += 1

        # Validate
        router.eval()
        train_acc = compute_accuracy(router, X_train, y_train)
        val_acc = compute_accuracy(router, X_val, y_val)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(router.state_dict(), args.save)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs}  loss={total_loss/num_batches:.4f}  "
                  f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  best_val={best_acc:.4f}")

    print(f"\nDone! Best val_acc: {best_acc:.4f}")
    print(f"Router weights saved to: {args.save}")


if __name__ == "__main__":
    main()
