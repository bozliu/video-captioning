# Dataset Policy and Preparation

This repository is distributed as **code-only** for public release.
No dataset files, extracted features, checkpoints, or proprietary media are included.

## Public Data Sources

1. Video2Commonsense annotations
- https://drive.google.com/file/d/1qt0JsOAqBsdCTlDUw0gw7c_IosShysoW/view?usp=sharing

2. ResNet152 features used by V2C baseline
- https://drive.google.com/file/d/1yUu4zMQzw_YOO5M8_i91ht3PmfkytolG/view?usp=sharing

3. MSVD project page
- https://www.cs.utexas.edu/~ml/clamp/videoDescription/

4. MSR-VTT provenance
- See paper reference [1] in README and V2C source [3].

## Expected Local Layout (not tracked in git)

```text
data/
  resnet152/
    video0.npy
    ...
  V2C_MSR-VTT_caption.json
  v2c_info.json
  msrvtt_new_info.json
```

## Notes
- Keep `data/` outside version control.
- Respect each dataset's license and terms of use.
- For reproducibility, use fixed split generation via:
  - `python -m videocap.prepare_data --data-root data --val-ratio 0.05 --seed 42`
