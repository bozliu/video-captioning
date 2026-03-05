from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .config import AppConfig, load_config, save_config
from .data import (
    CaptionDataset,
    create_splits_file,
    load_caption_table,
    load_split_json,
    make_eval_collate,
    make_train_collate,
    select_subset,
)
from .engine import evaluate_generation, is_better_metrics
from .metrics import save_json
from .model import ModelConfig, VideoPrefixReconstructorModel
from .utils import (
    AverageMeter,
    ensure_dir,
    maybe_autocast,
    seed_everything,
    select_device,
    timestamp,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train modern video captioning with reconstruction."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run name override.")
    return parser


def _build_model_cfg(cfg: AppConfig) -> ModelConfig:
    return ModelConfig(
        decoder_model_name=cfg.decoder_model_name,
        video_dim_in=cfg.video_dim_in,
        hidden_dim=cfg.hidden_dim,
        num_prefix_tokens=cfg.num_prefix_tokens,
        num_video_layers=cfg.num_video_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        lambda_recon=cfg.lambda_recon,
    )


def _has_nonfinite_grads(model: torch.nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is None:
            continue
        if not torch.isfinite(param.grad).all():
            return True
    return False


def _prepare_split_if_needed(cfg: AppConfig) -> None:
    split_path = Path(cfg.split_json)
    if split_path.exists():
        return

    create_splits_file(
        data_root=cfg.data_root,
        caption_json=cfg.caption_json,
        info_json=cfg.info_json,
        feature_dir=cfg.feature_dir,
        output_path=cfg.split_json,
        val_ratio=0.05,
        seed=cfg.seed,
    )


def _build_dataloaders(
    cfg: AppConfig, tokenizer
) -> tuple[DataLoader, DataLoader, Dict[str, List[int]]]:
    _prepare_split_if_needed(cfg)

    splits = load_split_json(cfg.split_json)
    train_ids = select_subset(splits["train"], cfg.train_subset_size, seed=cfg.seed)
    val_ids = select_subset(splits["val"], cfg.val_subset_size, seed=cfg.seed)

    captions = load_caption_table(cfg.caption_json)

    train_ds = CaptionDataset(
        video_ids=train_ids,
        captions_by_video=captions,
        feature_dir=cfg.feature_dir,
        mode="train",
        seed=cfg.seed,
    )
    val_ds = CaptionDataset(
        video_ids=val_ids,
        captions_by_video=captions,
        feature_dir=cfg.feature_dir,
        mode="eval",
        seed=cfg.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=make_train_collate(tokenizer, cfg.max_text_tokens),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=make_eval_collate(),
        pin_memory=False,
    )

    return train_loader, val_loader, {"train": train_ids, "val": val_ids, "test": splits["test"]}


def _save_checkpoint(
    output_path: Path,
    model: VideoPrefixReconstructorModel,
    optimizer,
    scheduler,
    epoch: int,
    cfg: AppConfig,
    best_metrics: Dict[str, float] | None,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": asdict(cfg),
        "best_metrics": best_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    if args.run_name:
        cfg.run_name = args.run_name

    seed_everything(cfg.seed)
    device = select_device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader, split_info = _build_dataloaders(cfg, tokenizer)

    model = VideoPrefixReconstructorModel(_build_model_cfg(cfg)).to(device)
    model.lm.config.pad_token_id = tokenizer.pad_token_id

    lm_params = []
    main_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("lm."):
            lm_params.append(param)
        else:
            main_params.append(param)

    optimizer = AdamW(
        [
            {"params": main_params, "lr": cfg.lr_main},
            {"params": lm_params, "lr": cfg.lr_lm},
        ],
        weight_decay=cfg.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    if cfg.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))
    amp_runtime_enabled = bool(cfg.amp and device.type == "cuda")
    if cfg.amp and device.type == "mps":
        print("AMP requested but disabled on MPS for numerical stability.")

    run_stamp = timestamp()
    run_dir = ensure_dir(str(Path(cfg.output_root) / f"{cfg.run_name}-{run_stamp}"))
    ckpt_dir = ensure_dir(str(run_dir / "checkpoints"))

    save_config(cfg, str(run_dir / "resolved_config.yaml"))
    save_json(split_info, str(run_dir / "split_info.json"))

    start_epoch = 1
    best_metrics: Dict[str, float] | None = None
    best_epoch = -1
    best_predictions: Dict[str, str] = {}
    best_references: Dict[str, List[str]] = {}
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    if cfg.resume_checkpoint:
        resume = torch.load(cfg.resume_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(resume["model"], strict=True)
        if resume.get("optimizer"):
            optimizer.load_state_dict(resume["optimizer"])
        if resume.get("scheduler"):
            scheduler.load_state_dict(resume["scheduler"])
        start_epoch = int(resume.get("epoch", 0)) + 1
        best_metrics = resume.get("best_metrics")
        print(f"Resumed from {cfg.resume_checkpoint} at epoch {start_epoch}.")

    print(
        f"Device={device.type}, train_videos={len(split_info['train'])}, "
        f"val_videos={len(split_info['val'])}, total_steps={total_steps}"
    )

    train_start = perf_counter()
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        meter_total = AverageMeter()
        meter_ce = AverageMeter()
        meter_recon = AverageMeter()

        iterator = tqdm(train_loader, desc=f"train epoch {epoch}", leave=False)
        for step, batch in enumerate(iterator, start=1):
            if cfg.max_steps_per_epoch is not None and step > cfg.max_steps_per_epoch:
                break

            video_features = batch["video_features"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with maybe_autocast(device, amp_runtime_enabled):
                outputs = model(
                    video_features=video_features,
                    frame_mask=frame_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss_total = outputs["loss_total"]

            if not torch.isfinite(loss_total):
                # MPS float16 can be unstable for some batches; auto-fallback to full precision.
                if amp_runtime_enabled:
                    print("Non-finite loss detected with AMP, falling back to full precision.")
                amp_runtime_enabled = False
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    video_features=video_features,
                    frame_mask=frame_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss_total = outputs["loss_total"]
                if not torch.isfinite(loss_total):
                    print("Skipping batch due to non-finite loss in full precision.")
                    continue

            if scaler.is_enabled():
                scaler.scale(loss_total).backward()
                scaler.unscale_(optimizer)
                if _has_nonfinite_grads(model):
                    print("Skipping optimizer step due to non-finite gradients.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_total.backward()
                if _has_nonfinite_grads(model):
                    print("Skipping optimizer step due to non-finite gradients.")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            scheduler.step()

            meter_total.update(float(outputs["loss_total"].detach().cpu().item()))
            meter_ce.update(float(outputs["loss_ce"].detach().cpu().item()))
            meter_recon.update(float(outputs["loss_recon"].detach().cpu().item()))

            if step % cfg.log_every == 0:
                iterator.set_postfix(
                    loss=f"{meter_total.avg:.4f}",
                    ce=f"{meter_ce.avg:.4f}",
                    recon=f"{meter_recon.avg:.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )

        val_metrics, val_predictions, val_references = evaluate_generation(
            model=model,
            loader=val_loader,
            tokenizer=tokenizer,
            cfg=cfg,
            device=device,
            desc=f"val epoch {epoch}",
        )

        row: Dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": meter_total.avg,
            "train_ce": meter_ce.avg,
            "train_recon": meter_recon.avg,
        }
        row.update({k: float(v) for k, v in val_metrics.items()})
        history.append(row)

        loss_line = (
            f"epoch={epoch} "
            f"loss={meter_total.avg:.4f} "
            f"ce={meter_ce.avg:.4f} "
            f"recon={meter_recon.avg:.4f}"
        )
        metric_line = (
            f"CIDEr={val_metrics.get('CIDEr', 0.0):.4f} "
            f"BLEU4={val_metrics.get('BLEU_4', 0.0):.4f} "
            f"ROUGE_L={val_metrics.get('ROUGE_L', 0.0):.4f}"
        )
        print(f"{loss_line} {metric_line}")

        if cfg.save_every_epoch:
            _save_checkpoint(
                output_path=Path(ckpt_dir) / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                cfg=cfg,
                best_metrics=best_metrics,
            )

        if is_better_metrics(val_metrics, best_metrics):
            best_metrics = val_metrics
            best_epoch = epoch
            best_predictions = val_predictions
            best_references = val_references
            bad_epochs = 0
            _save_checkpoint(
                output_path=Path(run_dir) / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                cfg=cfg,
                best_metrics=best_metrics,
            )
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.early_stopping_patience:
            print(f"Early stopping at epoch {epoch} after {bad_epochs} non-improving epochs.")
            break

    elapsed_sec = perf_counter() - train_start

    summary = {
        "device": device.type,
        "run_dir": str(run_dir),
        "elapsed_seconds": elapsed_sec,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics or {},
        "history": history,
        "config": asdict(cfg),
    }

    save_json(summary, str(Path(run_dir) / "metrics.json"))
    save_json(best_predictions, str(Path(run_dir) / "predictions_val_best.json"))
    save_json(best_references, str(Path(run_dir) / "references_val_best.json"))

    print(f"Training complete. Best epoch: {best_epoch}")
    print(f"Best metrics: {best_metrics}")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
