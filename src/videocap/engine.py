from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import AppConfig
from .metrics import compute_caption_metrics


@torch.no_grad()
def generate_predictions(
    model,
    loader: DataLoader,
    tokenizer,
    cfg: AppConfig,
    device: torch.device,
    desc: str,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    model.eval()
    preds: Dict[str, str] = {}
    refs: Dict[str, List[str]] = {}

    iterator = tqdm(loader, desc=desc, leave=False)
    for batch in iterator:
        video_features = batch["video_features"].to(device)
        frame_mask = batch["frame_mask"].to(device)

        captions = model.generate_captions(
            video_features=video_features,
            frame_mask=frame_mask,
            tokenizer=tokenizer,
            max_new_tokens=cfg.max_new_tokens,
            beam_size=cfg.beam_size,
        )

        for vid, cap, ref in zip(batch["video_ids"], captions, batch["references"]):
            preds[str(vid)] = cap
            refs[str(vid)] = list(ref)

    return preds, refs


@torch.no_grad()
def evaluate_generation(
    model,
    loader: DataLoader,
    tokenizer,
    cfg: AppConfig,
    device: torch.device,
    desc: str,
) -> tuple[Dict[str, float], Dict[str, str], Dict[str, List[str]]]:
    preds, refs = generate_predictions(model, loader, tokenizer, cfg, device, desc)
    metrics = compute_caption_metrics(refs, preds)
    return metrics, preds, refs


def is_better_metrics(current: Dict[str, float], best: Dict[str, float] | None) -> bool:
    if best is None:
        return True

    current_tuple = (
        float(current.get("CIDEr", 0.0)),
        float(current.get("BLEU_4", 0.0)),
        float(current.get("ROUGE_L", 0.0)),
    )
    best_tuple = (
        float(best.get("CIDEr", 0.0)),
        float(best.get("BLEU_4", 0.0)),
        float(best.get("ROUGE_L", 0.0)),
    )
    return current_tuple > best_tuple
