from __future__ import annotations

import argparse
import csv
import json
import random
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _try_import_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "Missing optional dependency for figure generation.\n"
            "Install: pip install matplotlib\n"
            f"Error: {exc}"
        )


def _load_run_predictions_and_references(
    run_dir: Path, split: str
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    pred_path = run_dir / f"predictions_{split}.json"
    ref_path = run_dir / f"references_{split}.json"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing file: {pred_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Missing file: {ref_path}")
    preds = _load_json(pred_path)
    refs = _load_json(ref_path)
    return preds, refs


def make_qualitative_examples(
    run_dir: Path,
    out_path: Path,
    split: str = "test",
    num_examples: int = 3,
    seed: int = 42,
    wrap_width: int = 88,
) -> None:
    plt = _try_import_matplotlib()
    preds, refs = _load_run_predictions_and_references(run_dir, split)

    keys = sorted(set(preds.keys()) & set(refs.keys()))
    if not keys:
        raise RuntimeError("No overlapping keys between predictions and references.")

    rng = random.Random(seed)
    chosen = rng.sample(keys, k=min(num_examples, len(keys)))

    lines: List[str] = [f"Qualitative Predictions ({run_dir.name} {split} split)", ""]
    for idx, vid in enumerate(chosen, start=1):
        ref_list = refs.get(vid) or []
        ref = ref_list[0] if ref_list else ""
        pred = preds.get(vid, "")

        ref_wrapped = textwrap.fill(ref, width=wrap_width, subsequent_indent="  ")
        pred_wrapped = textwrap.fill(pred, width=wrap_width, subsequent_indent="  ")

        lines.extend(
            [
                f"{idx}. {vid}",
                f"Reference: {ref_wrapped}",
                "",
                f"Prediction: {pred_wrapped}",
                "",
                "",
            ]
        )

    text = "\n".join(lines).rstrip() + "\n"

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=14,
        family="DejaVu Sans",
        transform=ax.transAxes,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoff(
    csv_path: Path,
    out_path: Path,
    metric_key: str = "cider",
    split: str = "val",
) -> None:
    plt = _try_import_matplotlib()

    run_names: List[str] = []
    runtimes: List[float] = []
    metrics: List[float] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("split") or "").strip() != split:
                continue
            run = (row.get("run_name") or "").strip()
            if not run:
                continue

            metric = float(row[metric_key])
            runtime_raw = (row.get("runtime_seconds") or "").strip()
            if not runtime_raw or runtime_raw.upper() == "NA":
                continue
            runtime = float(runtime_raw)

            run_names.append(run.replace("_stable2", ""))
            metrics.append(metric)
            runtimes.append(runtime)

    if not run_names:
        raise RuntimeError(f"No rows found for split={split} in {csv_path}")

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax2.bar(range(len(run_names)), runtimes, alpha=0.25, color="#f4a261")
    ax1.plot(range(len(run_names)), metrics, marker="o", linewidth=2.5, color="#1f77b4")

    for x, y in enumerate(metrics):
        ax1.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center")

    ax1.set_xticks(range(len(run_names)))
    ax1.set_xticklabels(run_names)
    ax1.set_ylabel("CIDEr (x100)")
    ax2.set_ylabel("Runtime (seconds)", color="#e76f51")
    ax1.set_title("Local Experiment Trade-off: Quality vs Runtime")
    ax1.grid(True, axis="y", alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate README figures from run outputs and CSV logs."
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help=(
            "Run directory containing predictions_*.json and references_*.json "
            "(typically under artifacts/)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="assets",
        help="Output directory for generated PNGs (default: assets).",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--benchmark-csv",
        type=str,
        default="results/benchmark_main.csv",
        help="CSV used for trade-off plot (default: results/benchmark_main.csv).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    csv_path = Path(args.benchmark_csv).expanduser().resolve()

    make_qualitative_examples(
        run_dir=run_dir,
        out_path=out_dir / "qualitative_examples.png",
        split=args.split,
        num_examples=args.num_examples,
        seed=args.seed,
    )
    plot_tradeoff(
        csv_path=csv_path,
        out_path=out_dir / "benchmark_tradeoff.png",
        metric_key="cider",
        split="val",
    )
    print(f"Wrote: {(out_dir / 'qualitative_examples.png').as_posix()}")
    print(f"Wrote: {(out_dir / 'benchmark_tradeoff.png').as_posix()}")


if __name__ == "__main__":
    main()
