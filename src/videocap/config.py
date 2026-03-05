from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class AppConfig:
    data_root: str = "data"
    feature_dir: str = "data/resnet152"
    caption_json: str = "data/V2C_MSR-VTT_caption.json"
    info_json: str = "data/v2c_info.json"
    split_json: str = "data/splits.json"

    train_subset_size: Optional[int] = None
    val_subset_size: Optional[int] = None
    test_subset_size: Optional[int] = None

    decoder_model_name: str = "distilgpt2"
    max_text_tokens: int = 32
    num_prefix_tokens: int = 12
    video_dim_in: int = 2048
    hidden_dim: int = 768
    num_video_layers: int = 2
    num_heads: int = 8
    dropout: float = 0.1
    lambda_recon: float = 0.2

    batch_size: int = 16
    epochs: int = 4
    lr_main: float = 2e-4
    lr_lm: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05

    amp: bool = True
    device: str = "auto"
    beam_size: int = 1
    max_new_tokens: int = 24

    num_workers: int = 0
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 2
    seed: int = 42
    log_every: int = 20
    max_steps_per_epoch: Optional[int] = None

    run_name: str = "m3_quick"
    output_root: str = "artifacts"
    save_every_epoch: bool = True
    resume_checkpoint: Optional[str] = None


def _coerce_value(field_type: Any, value: Any) -> Any:
    if value is None:
        return None
    return value


def load_config(config_path: str) -> AppConfig:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    allowed = {f.name: f.type for f in fields(AppConfig)}
    filtered: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in allowed:
            filtered[k] = _coerce_value(allowed[k], v)

    cfg = AppConfig(**filtered)
    return resolve_paths(cfg, base_dir=path.parent)


def resolve_paths(cfg: AppConfig, base_dir: Path) -> AppConfig:
    cfg_dict = asdict(cfg)
    path_fields = {
        "data_root",
        "feature_dir",
        "caption_json",
        "info_json",
        "split_json",
        "output_root",
        "resume_checkpoint",
    }

    for key in path_fields:
        value = cfg_dict.get(key)
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute():
            cwd_candidate = (Path.cwd() / p).resolve()
            base_candidate = (base_dir / p).resolve()
            if cwd_candidate.exists():
                cfg_dict[key] = str(cwd_candidate)
            else:
                cfg_dict[key] = str(base_candidate)

    return AppConfig(**cfg_dict)


def to_dict(cfg: AppConfig) -> Dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: AppConfig, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(to_dict(cfg), f, sort_keys=False)
