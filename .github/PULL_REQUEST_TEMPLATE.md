## Summary
- What changed and why?

## Checklist
- [ ] No dataset/model binaries added
- [ ] No personal/sensitive info (paths, IDs, secrets)
- [ ] README/docs updated (if behavior changed)
- [ ] CI passes (lint + tests + CLI help smoke)
- [ ] Backward-compatible CLI (`python -m videocap.*`)

## Validation
- Commands run locally:
  - `pytest -q -k "not integration"`
  - `python -m videocap.train --help`
  - `python -m videocap.evaluate --help`
