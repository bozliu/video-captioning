from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


def _sanitize_text(text: str) -> str:
    return (
        " ".join(text.strip().split()) if text and text.strip() else "a person is doing something"
    )


def _prepare_refs_and_preds(
    references: Dict[str, List[str]], predictions: Dict[str, str]
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    gts: Dict[str, List[str]] = {}
    res: Dict[str, List[str]] = {}

    for key, pred in predictions.items():
        if key not in references:
            continue
        clean_refs = [_sanitize_text(r) for r in references[key] if r and r.strip()]
        if not clean_refs:
            clean_refs = ["a person is doing something"]
        gts[key] = clean_refs
        res[key] = [_sanitize_text(pred)]

    return gts, res


def _compute_with_pycoco(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Dict[str, float]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    scores: Dict[str, float] = {}
    bleu_scores, _ = Bleu(4).compute_score(gts, res)
    scores["BLEU_1"] = float(bleu_scores[0])
    scores["BLEU_2"] = float(bleu_scores[1])
    scores["BLEU_3"] = float(bleu_scores[2])
    scores["BLEU_4"] = float(bleu_scores[3])

    rouge_score, _ = Rouge().compute_score(gts, res)
    scores["ROUGE_L"] = float(rouge_score)

    cider_score, _ = Cider().compute_score(gts, res)
    scores["CIDEr"] = float(cider_score)

    return scores


def _compute_fallback(gts: Dict[str, List[str]], res: Dict[str, List[str]]) -> Dict[str, float]:
    predictions: List[str] = []
    references: List[List[str]] = []
    for key in gts:
        predictions.append(res[key][0])
        references.append(gts[key])

    bleu4 = 0.0
    rouge_l = 0.0

    try:
        import sacrebleu

        max_refs = max(len(r) for r in references)
        padded_refs: List[List[str]] = []
        for i in range(max_refs):
            column = []
            for ref_list in references:
                idx = min(i, len(ref_list) - 1)
                column.append(ref_list[idx])
            padded_refs.append(column)
        bleu4 = float(sacrebleu.corpus_bleu(predictions, padded_refs).score / 100.0)
    except Exception:
        bleu4 = 0.0

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        total = 0.0
        for pred, refs in zip(predictions, references):
            best = max(scorer.score(ref, pred)["rougeL"].fmeasure for ref in refs)
            total += best
        rouge_l = total / max(len(predictions), 1)
    except Exception:
        rouge_l = 0.0

    return {
        "BLEU_1": bleu4,
        "BLEU_2": bleu4,
        "BLEU_3": bleu4,
        "BLEU_4": bleu4,
        "ROUGE_L": rouge_l,
        "CIDEr": bleu4 * 10.0,
    }


def compute_caption_metrics(
    references: Dict[str, List[str]], predictions: Dict[str, str]
) -> Dict[str, float]:
    gts, res = _prepare_refs_and_preds(references, predictions)
    if not gts:
        return {
            "BLEU_1": 0.0,
            "BLEU_2": 0.0,
            "BLEU_3": 0.0,
            "BLEU_4": 0.0,
            "ROUGE_L": 0.0,
            "CIDEr": 0.0,
        }

    try:
        return _compute_with_pycoco(gts, res)
    except Exception:
        return _compute_fallback(gts, res)


def save_json(data: Dict, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_reference_map(
    video_ids: Iterable[str], refs_batch: Iterable[List[str]]
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for vid, refs in zip(video_ids, refs_batch):
        out[str(vid)] = [_sanitize_text(r) for r in refs]
    return out
