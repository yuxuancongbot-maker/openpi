#!/usr/bin/env python3
"""Collect prefix features and L1 Flow diffs for Router training.

Usage:
    uv run scripts/collect_router_data.py \
        --checkpoint_dir /path/to/pi05_libero_l1flow_pytorch \
        --num_samples 2000 \
        --output router_data.npz
"""

import argparse
import dataclasses
import logging
import pathlib

import jax
import numpy as np
import torch
from tqdm import tqdm

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import checkpoints as _checkpoints
from openpi.training.data_loader import create_torch_dataset, transform_dataset
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("collect_router_data")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Collect prefix features and diffs for Router training")
    parser.add_argument("--config", default="pi05_libero_l1_flow", help="Training config name")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to checkpoint dir (containing model.safetensors + assets)")
    parser.add_argument("--num_samples", type=int, default=2000, help="Number of samples to collect")
    parser.add_argument("--output", default="router_data.npz", help="Output .npz file path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # 1. Load config
    train_config = _config.get_config(args.config)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    logger.info("Config: %s", args.config)

    # 2. Load norm stats from checkpoint
    ckpt_dir = pathlib.Path(args.checkpoint_dir)
    norm_stats = _checkpoints.load_norm_stats(ckpt_dir / "assets", data_config.asset_id)
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
    logger.info("Loaded norm stats for: %s", data_config.asset_id)

    # 3. Load model
    weight_path = str(ckpt_dir / "model.safetensors")
    model = train_config.model.load_pytorch(train_config, weight_path)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval()
    model = model.to(device)
    logger.info("Model loaded and moved to %s", device)

    # 4. Create dataset with transforms
    raw_dataset = create_torch_dataset(data_config, train_config.model.action_horizon, train_config.model)
    dataset = transform_dataset(raw_dataset, data_config)
    logger.info("Dataset created: %d samples", len(dataset))

    # 5. Collect data
    prefix_feats = []
    diffs = []

    for i in tqdm(range(min(args.num_samples, len(dataset))), desc="Collecting"):
        sample = dataset[i]

        # Convert all numpy arrays to torch tensors (handles nested dicts like "image").
        # String/object fields (e.g. raw "prompt") will fail torch.from_numpy and are skipped.
        def to_tensor(x):
            if isinstance(x, dict):
                return {k: to_tensor(v) for k, v in x.items()}
            arr = np.asarray(x)
            try:
                return torch.from_numpy(arr).to(device)[None, ...]
            except TypeError:
                return None
        batch = {k: v for k, v in to_tensor(sample).items() if v is not None}
        obs = _model.Observation.from_dict(batch)
        obs = _model.Observation.from_dict(batch)

        # --- Model internals ---
        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(obs, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # KV cache fill + get prefix hidden states (free!)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        lm_output = model.paligemma_with_expert.paligemma.language_model.forward(
            inputs_embeds=prefix_embs,
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            use_cache=True,
            adarms_cond=None,
            output_hidden_states=True,
        )
        past_key_values = lm_output.past_key_values
        prefix_hidden = lm_output.hidden_states[-1]
        # Convert to float32 for Router training (VLM computes in bfloat16,
        # but numpy doesn't natively support bfloat16 and Router weights are float32).
        prefix_feat = prefix_hidden.mean(dim=1).to(torch.float32).detach().cpu().numpy()  # (1, 2048)

        # Sample noise (same for both steps)
        noise = model.sample_noise((1, model.config.action_horizon, model.config.action_dim), device)

        # Run 1-step and 2-step
        bsize = state.shape[0]
        actions_1step = model._l1_1step(state, prefix_pad_masks, past_key_values, noise, bsize, device)
        actions_2step = model._l1_2step(state, prefix_pad_masks, past_key_values, noise, bsize, device)

        diff = (actions_1step - actions_2step).abs().mean().item()

        prefix_feats.append(prefix_feat)
        diffs.append(diff)

    # 6. Save
    feats = np.concatenate(prefix_feats, axis=0)  # (N, 2048)
    diffs_arr = np.array(diffs)  # (N,)
    np.savez(args.output, prefix_feats=feats, diffs=diffs_arr)

    logger.info("Saved %d samples to %s", len(diffs), args.output)
    logger.info("Diff stats: mean=%.6f, median=%.6f, min=%.6f, max=%.6f, std=%.6f",
                diffs_arr.mean(), np.median(diffs_arr), diffs_arr.min(), diffs_arr.max(), diffs_arr.std())
    logger.info("Percentiles: P25=%.6f, P50=%.6f, P75=%.6f, P90=%.6f",
                np.percentile(diffs_arr, 25), np.percentile(diffs_arr, 50),
                np.percentile(diffs_arr, 75), np.percentile(diffs_arr, 90))


if __name__ == "__main__":
    main()
