from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import parse_video_id, video_numeric_id

SPECIAL_TOKENS = {"<sos>", "<eos>", "<pad>", "<PAD>", "<UNK>", "<unk>", "<sep>", ""}


def tokens_to_text(tokens: Sequence[str]) -> str:
    text_tokens = [
        tok.strip() for tok in tokens if tok and tok.strip() and tok not in SPECIAL_TOKENS
    ]
    text = " ".join(text_tokens).strip()
    return text if text else "a person is doing something"


def load_caption_table(caption_json: str) -> Dict[str, List[str]]:
    with Path(caption_json).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    table: Dict[str, List[str]] = {}
    for video_key, entries in raw.items():
        video_id = parse_video_id(video_key)
        refs: List[str] = []
        for item in entries:
            tokens = item.get("final_caption", [])
            if isinstance(tokens, list):
                refs.append(tokens_to_text(tokens))
        unique_refs = []
        seen = set()
        for ref in refs:
            low = ref.lower()
            if low not in seen:
                seen.add(low)
                unique_refs.append(ref)
        if unique_refs:
            table[video_id] = unique_refs
    return table


def _filter_available_ids(
    ids: Iterable[int],
    captions: Dict[str, List[str]],
    feature_dir: str,
) -> List[int]:
    feat_root = Path(feature_dir)
    filtered: List[int] = []
    for v in ids:
        key = parse_video_id(v)
        feat_file = feat_root / f"{key}.npy"
        if key in captions and feat_file.exists():
            filtered.append(int(video_numeric_id(key)))
    return filtered


def split_train_val(train_ids: Sequence[int], val_ratio: float, seed: int) -> Dict[str, List[int]]:
    ids = [int(v) for v in train_ids]
    rnd = random.Random(seed)
    rnd.shuffle(ids)

    val_count = int(len(ids) * val_ratio)
    if val_count <= 0 and len(ids) > 1:
        val_count = 1

    val_ids = sorted(ids[:val_count])
    train_main_ids = sorted(ids[val_count:])
    return {"train": train_main_ids, "val": val_ids}


def create_splits_file(
    data_root: str,
    caption_json: str,
    info_json: str,
    feature_dir: str,
    output_path: str,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[int]]:
    captions = load_caption_table(caption_json)

    with Path(info_json).open("r", encoding="utf-8") as f:
        info = json.load(f)

    videos = info.get("videos", {})
    train_raw = videos.get("train", [])
    test_raw = videos.get("test", [])

    train_ids = _filter_available_ids(train_raw, captions, feature_dir)
    test_ids = sorted(_filter_available_ids(test_raw, captions, feature_dir))

    split = split_train_val(train_ids, val_ratio=val_ratio, seed=seed)
    split["test"] = test_ids

    payload = {
        "seed": seed,
        "val_ratio": val_ratio,
        "counts": {k: len(v) for k, v in split.items()},
        "train": split["train"],
        "val": split["val"],
        "test": split["test"],
        "data_root": str(Path(data_root).resolve()),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return split


def load_split_json(split_json: str) -> Dict[str, List[int]]:
    with Path(split_json).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return {
        "train": [int(x) for x in raw.get("train", [])],
        "val": [int(x) for x in raw.get("val", [])],
        "test": [int(x) for x in raw.get("test", [])],
    }


def select_subset(video_ids: Sequence[int], subset_size: int | None, seed: int) -> List[int]:
    ids = [int(v) for v in video_ids]
    if subset_size is None or subset_size <= 0 or subset_size >= len(ids):
        return sorted(ids)

    rnd = random.Random(seed)
    copy = ids[:]
    rnd.shuffle(copy)
    selected = copy[:subset_size]
    return sorted(selected)


class CaptionDataset(Dataset):
    def __init__(
        self,
        video_ids: Sequence[int],
        captions_by_video: Dict[str, List[str]],
        feature_dir: str,
        mode: str,
        seed: int = 42,
    ) -> None:
        self.mode = mode
        self.feature_dir = Path(feature_dir)
        self.captions_by_video = captions_by_video
        self.video_ids = [parse_video_id(v) for v in video_ids]
        self.rnd = random.Random(seed)

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        video_id = self.video_ids[idx]
        feat_path = self.feature_dir / f"{video_id}.npy"
        features = np.load(feat_path).astype(np.float32)

        refs = self.captions_by_video[video_id]
        if self.mode == "train":
            target = refs[self.rnd.randint(0, len(refs) - 1)]
        else:
            target = refs[0]

        return {
            "video_id": video_id,
            "video_features": torch.from_numpy(features),
            "target_text": target,
            "references": refs,
        }


def pad_video_batch(video_tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(video_tensors)
    max_frames = max(v.shape[0] for v in video_tensors)
    feat_dim = video_tensors[0].shape[1]

    batch = torch.zeros((batch_size, max_frames, feat_dim), dtype=torch.float32)
    frame_mask = torch.zeros((batch_size, max_frames), dtype=torch.long)

    for i, v in enumerate(video_tensors):
        num_frames = v.shape[0]
        batch[i, :num_frames] = v
        frame_mask[i, :num_frames] = 1

    return batch, frame_mask


def make_train_collate(tokenizer, max_text_tokens: int):
    def collate_fn(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
        video_tensors = [s["video_features"] for s in samples]
        texts = [s["target_text"] for s in samples]
        video_ids = [s["video_id"] for s in samples]
        refs = [s["references"] for s in samples]

        video_batch, frame_mask = pad_video_batch(video_tensors)
        tokenized = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_text_tokens,
            return_tensors="pt",
        )

        return {
            "video_ids": video_ids,
            "video_features": video_batch,
            "frame_mask": frame_mask,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "references": refs,
            "target_texts": texts,
        }

    return collate_fn


def make_eval_collate():
    def collate_fn(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
        video_tensors = [s["video_features"] for s in samples]
        video_ids = [s["video_id"] for s in samples]
        refs = [s["references"] for s in samples]

        video_batch, frame_mask = pad_video_batch(video_tensors)
        return {
            "video_ids": video_ids,
            "video_features": video_batch,
            "frame_mask": frame_mask,
            "references": refs,
        }

    return collate_fn
