from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import load_config
from .data import (
    CaptionDataset,
    load_caption_table,
    load_split_json,
    make_eval_collate,
    select_subset,
)
from .engine import evaluate_generation
from .metrics import save_json
from .model import ModelConfig, VideoPrefixReconstructorModel
from .utils import select_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on val/test split.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path.")
    parser.add_argument(
        "--split", type=str, default="val", choices=["val", "test"], help="Split to evaluate."
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
    model.lm.config.pad_token_id = tokenizer.pad_token_id

    split = load_split_json(cfg.split_json)
    split_ids = split[args.split]
    if args.split == "val":
        split_ids = select_subset(split_ids, cfg.val_subset_size, seed=cfg.seed)
    if args.split == "test":
        split_ids = select_subset(split_ids, cfg.test_subset_size, seed=cfg.seed)

    captions = load_caption_table(cfg.caption_json)
    dataset = CaptionDataset(
        video_ids=split_ids,
        captions_by_video=captions,
        feature_dir=cfg.feature_dir,
        mode="eval",
        seed=cfg.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=make_eval_collate(),
        pin_memory=False,
    )

    metrics, preds, refs = evaluate_generation(
        model=model,
        loader=loader,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        desc=f"eval {args.split}",
    )

    ckpt_path = Path(args.checkpoint).resolve()
    run_dir = ckpt_path.parent
    if run_dir.name == "checkpoints":
        run_dir = run_dir.parent

    save_json(metrics, str(run_dir / f"metrics_{args.split}.json"))
    save_json(preds, str(run_dir / f"predictions_{args.split}.json"))
    save_json(refs, str(run_dir / f"references_{args.split}.json"))

    print(f"split={args.split} device={device.type}")
    print(metrics)


if __name__ == "__main__":
    main()
