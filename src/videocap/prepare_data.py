from __future__ import annotations

import argparse
from pathlib import Path

from .data import create_splits_file


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic train/val/test splits for local video captioning."
    )
    parser.add_argument("--data-root", type=str, default="data", help="Path to data directory.")
    parser.add_argument("--feature-dir", type=str, default=None, help="Path to feature directory.")
    parser.add_argument("--caption-json", type=str, default=None, help="Path to caption json.")
    parser.add_argument("--info-json", type=str, default=None, help="Path to v2c info json.")
    parser.add_argument("--split-json", type=str, default=None, help="Output split json path.")
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Validation ratio from original train split."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    feature_dir = Path(args.feature_dir or (data_root / "resnet152")).resolve()
    caption_json = Path(args.caption_json or (data_root / "V2C_MSR-VTT_caption.json")).resolve()
    info_json = Path(args.info_json or (data_root / "v2c_info.json")).resolve()
    split_json = Path(args.split_json or (data_root / "splits.json")).resolve()

    split = create_splits_file(
        data_root=str(data_root),
        caption_json=str(caption_json),
        info_json=str(info_json),
        feature_dir=str(feature_dir),
        output_path=str(split_json),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )

    print("Prepared split file:", split_json)
    print("Counts:", {k: len(v) for k, v in split.items()})


if __name__ == "__main__":
    main()
