from __future__ import annotations

from pathlib import Path

from transformers import AutoTokenizer

from videocap.data import (
    CaptionDataset,
    load_caption_table,
    load_split_json,
    make_train_collate,
    split_train_val,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def test_split_reproducibility() -> None:
    ids = list(range(100))
    split_a = split_train_val(ids, val_ratio=0.1, seed=42)
    split_b = split_train_val(ids, val_ratio=0.1, seed=42)
    assert split_a == split_b
    assert len(split_a["val"]) == 10


def test_dataset_and_tokenizer_collate() -> None:
    split_json = DATA / "splits.json"
    if not split_json.exists():
        return

    split = load_split_json(str(split_json))
    captions = load_caption_table(str(DATA / "V2C_MSR-VTT_caption.json"))

    ds = CaptionDataset(
        video_ids=split["train"][:4],
        captions_by_video=captions,
        feature_dir=str(DATA / "resnet152"),
        mode="train",
        seed=42,
    )

    samples = [ds[i] for i in range(2)]
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    batch = make_train_collate(tokenizer, max_text_tokens=32)(samples)

    assert batch["video_features"].shape[0] == 2
    assert batch["frame_mask"].shape[0] == 2
    assert batch["input_ids"].shape[0] == 2
    assert batch["attention_mask"].shape[0] == 2
