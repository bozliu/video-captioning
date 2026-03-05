from __future__ import annotations

import torch
from transformers import AutoTokenizer

from videocap.model import ModelConfig, VideoPrefixReconstructorModel


def test_model_forward_loss_finite() -> None:
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = VideoPrefixReconstructorModel(
        ModelConfig(
            decoder_model_name="distilgpt2",
            video_dim_in=2048,
            hidden_dim=768,
            num_prefix_tokens=12,
            num_video_layers=2,
            num_heads=8,
            dropout=0.1,
            lambda_recon=0.2,
        )
    )

    texts = ["a person is dancing", "someone is cooking food"]
    tok = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")

    video_features = torch.randn(2, 40, 2048)
    frame_mask = torch.ones(2, 40, dtype=torch.long)

    outputs = model(
        video_features=video_features,
        frame_mask=frame_mask,
        input_ids=tok["input_ids"],
        attention_mask=tok["attention_mask"],
    )

    assert torch.isfinite(outputs["loss_total"])
    assert outputs["loss_total"].item() > 0
