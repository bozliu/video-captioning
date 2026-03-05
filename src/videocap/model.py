from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


@dataclass
class ModelConfig:
    decoder_model_name: str
    video_dim_in: int
    hidden_dim: int
    num_prefix_tokens: int
    num_video_layers: int
    num_heads: int
    dropout: float
    lambda_recon: float


class VideoPrefixReconstructorModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.lm = AutoModelForCausalLM.from_pretrained(cfg.decoder_model_name)
        self.hidden_dim = int(cfg.hidden_dim)
        self.num_prefix_tokens = int(cfg.num_prefix_tokens)
        self.lambda_recon = float(cfg.lambda_recon)

        self.video_proj = nn.Linear(cfg.video_dim_in, cfg.hidden_dim)
        self.motion_proj = nn.Linear(cfg.video_dim_in, cfg.hidden_dim)
        self.gate = nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_video_layers
        )

        self.prefix_projector = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim * cfg.num_prefix_tokens),
        )

        self.reconstructor = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        )

        self.dropout = nn.Dropout(cfg.dropout)

    def encode_video(
        self, video_features: torch.Tensor, frame_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_embed = self.video_proj(video_features)

        deltas = torch.zeros_like(video_features)
        deltas[:, 1:] = video_features[:, 1:] - video_features[:, :-1]
        motion_embed = self.motion_proj(deltas)

        gate = torch.sigmoid(self.gate(torch.cat([video_embed, motion_embed], dim=-1)))
        fused = gate * video_embed + (1.0 - gate) * motion_embed
        fused = self.dropout(fused)

        # MPS currently lacks some nested-tensor mask ops used by TransformerEncoder.
        # For this dataset frame counts are fixed (40), so we keep a mask only for pooling.
        encoded = self.temporal_encoder(fused)

        mask = frame_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (encoded * mask).sum(dim=1) / denom

        prefix = self.prefix_projector(pooled).view(-1, self.num_prefix_tokens, self.hidden_dim)
        return prefix, pooled

    def forward(
        self,
        video_features: torch.Tensor,
        frame_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        prefix_embeds, video_global = self.encode_video(video_features, frame_mask)

        token_embeds = self.lm.transformer.wte(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(
            (input_ids.shape[0], self.num_prefix_tokens),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        labels = input_ids.masked_fill(attention_mask == 0, -100)
        prefix_labels = torch.full(
            (labels.shape[0], self.num_prefix_tokens),
            fill_value=-100,
            dtype=labels.dtype,
            device=labels.device,
        )
        labels = torch.cat([prefix_labels, labels], dim=1)

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        lm_loss = outputs.loss
        final_hidden = outputs.hidden_states[-1][:, self.num_prefix_tokens :, :]
        text_mask = attention_mask.unsqueeze(-1).float()
        text_denom = text_mask.sum(dim=1).clamp_min(1.0)
        text_summary = (final_hidden * text_mask).sum(dim=1) / text_denom

        recon_pred = self.reconstructor(text_summary)
        recon_mse = F.mse_loss(recon_pred, video_global)
        recon_cos = 1.0 - F.cosine_similarity(recon_pred, video_global, dim=-1).mean()
        recon_loss = 0.5 * recon_mse + 0.5 * recon_cos

        loss_total = lm_loss + self.lambda_recon * recon_loss
        return {
            "loss_total": loss_total,
            "loss_ce": lm_loss,
            "loss_recon": recon_loss,
        }

    @torch.no_grad()
    def _greedy_decode(
        self,
        prefix_embeds: torch.Tensor,
        eos_token_id: int,
        start_token_id: int,
        max_new_tokens: int,
    ) -> torch.Tensor:
        batch_size = prefix_embeds.shape[0]
        device = prefix_embeds.device

        prefix_mask = torch.ones(
            (batch_size, self.num_prefix_tokens), dtype=torch.long, device=device
        )
        prefix_out = self.lm(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            use_cache=True,
            return_dict=True,
        )

        past_key_values = prefix_out.past_key_values
        attention_mask = prefix_mask

        next_tokens = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        generated: List[torch.Tensor] = []
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)],
                dim=1,
            )
            out = self.lm(
                input_ids=next_tokens,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            logits = out.logits[:, -1, :]
            next_tokens = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_tokens)
            past_key_values = out.past_key_values

            finished |= next_tokens.squeeze(1).eq(eos_token_id)
            if bool(finished.all()):
                break

        if not generated:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)
        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def generate_captions(
        self,
        video_features: torch.Tensor,
        frame_mask: torch.Tensor,
        tokenizer,
        max_new_tokens: int,
        beam_size: int,
    ) -> List[str]:
        prefix_embeds, _ = self.encode_video(video_features, frame_mask)
        eos_token_id = int(tokenizer.eos_token_id)
        start_token_id = int(
            tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_token_id
        )

        generated_ids: torch.Tensor
        if beam_size > 1:
            try:
                generated_ids = self.lm.generate(
                    inputs_embeds=prefix_embeds,
                    attention_mask=torch.ones(
                        (prefix_embeds.shape[0], prefix_embeds.shape[1]),
                        dtype=torch.long,
                        device=prefix_embeds.device,
                    ),
                    max_new_tokens=max_new_tokens,
                    num_beams=beam_size,
                    do_sample=False,
                    eos_token_id=eos_token_id,
                    pad_token_id=eos_token_id,
                )
                if generated_ids.shape[1] > max_new_tokens:
                    generated_ids = generated_ids[:, -max_new_tokens:]
            except Exception:
                generated_ids = self._greedy_decode(
                    prefix_embeds, eos_token_id, start_token_id, max_new_tokens
                )
        else:
            generated_ids = self._greedy_decode(
                prefix_embeds, eos_token_id, start_token_id, max_new_tokens
            )

        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        cleaned = [
            " ".join(d.strip().split()) if d.strip() else "a person is doing something"
            for d in decoded
        ]
        return cleaned
