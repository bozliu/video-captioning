#!/usr/bin/env bash
set -euo pipefail

python -m videocap.prepare_data --data-root data --val-ratio 0.05 --seed 42

python -m videocap.train --config configs/smoke_128.yaml
python -m videocap.train --config configs/baseline_quick.yaml
python -m videocap.train --config configs/recon_quick.yaml
python -m videocap.train --config configs/full_quick.yaml

python - <<'PY'
import json
from pathlib import Path

root = Path('artifacts')
rows = []
for m in sorted(root.glob('*/metrics.json')):
    data = json.loads(m.read_text(encoding='utf-8'))
    best = data.get('best_metrics', {})
    rows.append((m.parent.name, best.get('CIDEr', 0.0), best.get('BLEU_4', 0.0), best.get('ROUGE_L', 0.0), str(m.parent / 'best.pt')))

rows.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
print('Run ranking (CIDEr, BLEU_4, ROUGE_L):')
for row in rows:
    print(row)

if rows:
    print('Best checkpoint:', rows[0][-1])
PY
