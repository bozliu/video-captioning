from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_video_file(video_dir: Path, video_id: str) -> Path | None:
    # Fast path: common extensions in the requested folder.
    for ext in [".mp4", ".webm", ".mkv", ".mov", ".avi"]:
        candidate = video_dir / f"{video_id}{ext}"
        if candidate.exists():
            return candidate

    # Fallback: sometimes videos are stored under a nested folder.
    for ext in [".mp4", ".webm", ".mkv", ".mov", ".avi"]:
        matches = list(video_dir.glob(f"**/{video_id}{ext}"))
        if matches:
            return matches[0]

    return None


def _run_ffmpeg_make_gif(
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    fps: int,
    width: int,
    max_colors: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit(
            "ffmpeg was not found on PATH.\n"
            "Install via conda (recommended): conda install -c conda-forge ffmpeg\n"
            "Or on macOS: brew install ffmpeg"
        )

    # Palette mode gives noticeably better quality per byte than direct GIF conversion.
    vf = (
        f"fps={fps},scale={width}:-1:flags=lanczos,"
        "split[s0][s1];"
        f"[s0]palettegen=max_colors={max_colors}:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=5"
    )

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(start_seconds),
        "-t",
        str(duration_seconds),
        "-i",
        str(input_path),
        "-vf",
        vf,
        "-loop",
        "0",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate small qualitative GIF clips for README (no videos committed)."
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing local video files, e.g. data/videos (not tracked in git).",
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
        "--video-ids",
        type=str,
        required=True,
        help="Comma-separated list of video ids, e.g. video9703,video7481,video7116",
    )
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument(
        "--out-dir",
        type=str,
        default="assets/gifs",
        help="Output folder for GIFs (default: assets/gifs).",
    )
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--max-colors", type=int, default=128)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    video_dir = Path(args.video_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    pred_path = run_dir / f"predictions_{args.split}.json"
    ref_path = run_dir / f"references_{args.split}.json"
    if not pred_path.exists():
        raise SystemExit(f"Missing file: {pred_path}")
    if not ref_path.exists():
        raise SystemExit(f"Missing file: {ref_path}")

    preds: Dict[str, str] = _load_json(pred_path)
    refs: Dict[str, List[str]] = _load_json(ref_path)

    video_ids = [v.strip() for v in args.video_ids.split(",") if v.strip()]
    if not video_ids:
        raise SystemExit("No video ids provided via --video-ids")

    out_dir.mkdir(parents=True, exist_ok=True)

    missing: List[str] = []
    examples = []
    for vid in video_ids:
        video_path = _find_video_file(video_dir, vid)
        if not video_path:
            missing.append(vid)
            continue

        gif_path = out_dir / f"{vid}.gif"
        _run_ffmpeg_make_gif(
            input_path=video_path,
            output_path=gif_path,
            start_seconds=args.start_seconds,
            duration_seconds=args.duration_seconds,
            fps=args.fps,
            width=args.width,
            max_colors=args.max_colors,
        )

        ref_list = refs.get(vid) or []
        ref = (ref_list[0] if ref_list else "").strip()
        pred = (preds.get(vid) or "").strip()

        try:
            rel_gif = gif_path.relative_to(Path.cwd())
            gif_path_for_manifest = rel_gif.as_posix()
        except ValueError:
            gif_path_for_manifest = gif_path.as_posix()

        examples.append(
            {
                "video_id": vid,
                "gif_path": gif_path_for_manifest,
                "reference_caption": ref,
                "prediction_caption": pred,
            }
        )

        size_mb = gif_path.stat().st_size / (1024 * 1024)
        if size_mb > 5.0:
            print(f"WARNING: {gif_path.name} is {size_mb:.2f} MB (> 5 MB target).")

    if missing:
        expected = [f"{m}.mp4" for m in missing]
        raise SystemExit(
            "Missing video files for ids:\n"
            f"- {', '.join(missing)}\n"
            f"Expected filenames under {video_dir} (or nested):\n"
            f"- {', '.join(expected)}"
        )

    manifest = {
        "run_name": run_dir.name,
        "split": args.split,
        "generator": {
            "start_seconds": args.start_seconds,
            "duration_seconds": args.duration_seconds,
            "fps": args.fps,
            "width": args.width,
            "max_colors": args.max_colors,
        },
        "examples": examples,
    }

    manifest_path = out_dir / "qualitative_gifs.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    for ex in examples:
        print(f"Wrote: {ex['gif_path']}")
    print(f"Wrote: {manifest_path.as_posix()}")


if __name__ == "__main__":
    main()
