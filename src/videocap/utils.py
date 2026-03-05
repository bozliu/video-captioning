from __future__ import annotations

import contextlib
import random
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str) -> torch.device:
    if device and device != "auto":
        return torch.device(device)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@contextlib.contextmanager
def maybe_autocast(device: torch.device, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    if device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            yield
        return

    if device.type == "mps":
        with torch.autocast(device_type="mps", dtype=torch.float16):
            yield
        return

    yield


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_video_id(video_id: str | int) -> str:
    if isinstance(video_id, int):
        return f"video{video_id}"
    if video_id.startswith("video"):
        return video_id
    return f"video{video_id}"


def video_numeric_id(video_id: str | int) -> int:
    vid = parse_video_id(video_id)
    return int(vid.replace("video", ""))
