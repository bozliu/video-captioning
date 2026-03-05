from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .config import load_config
from .data import load_caption_table
from .model import ModelConfig, VideoPrefixReconstructorModel
from .utils import parse_video_id, select_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate caption for one video id.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument(
        "--video-id", type=str, required=True, help="Video id, e.g. video123 or 123."
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    device = select_device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = VideoPrefixReconstructorModel(
        ModelConfig(
            decoder_model_name=cfg.decoder_model_name,
            video_dim_in=cfg.video_dim_in,
            hidden_dim=cfg.hidden_dim,
            num_prefix_tokens=cfg.num_prefix_tokens,
            num_video_layers=cfg.num_video_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            lambda_recon=cfg.lambda_recon,
        )
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    model.lm.config.pad_token_id = tokenizer.pad_token_id

    video_id = parse_video_id(args.video_id)
    feat_path = Path(cfg.feature_dir) / f"{video_id}.npy"
    if not feat_path.exists():
        raise FileNotFoundError(f"Feature file not found: {feat_path}")

    feats = torch.from_numpy(np.load(feat_path).astype(np.float32)).unsqueeze(0).to(device)
    frame_mask = torch.ones((1, feats.shape[1]), dtype=torch.long, device=device)

    caption = model.generate_captions(
        video_features=feats,
        frame_mask=frame_mask,
        tokenizer=tokenizer,
        max_new_tokens=cfg.max_new_tokens,
        beam_size=cfg.beam_size,
    )[0]

    print(f"video_id: {video_id}")
    print(f"prediction: {caption}")

    refs = load_caption_table(cfg.caption_json).get(video_id, [])
    if refs:
        print("references:")
        for i, ref in enumerate(refs[:5], start=1):
            print(f"  {i}. {ref}")


if __name__ == "__main__":
    main()
