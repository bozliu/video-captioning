# Data Download Guide (Public Links)

This project intentionally excludes dataset/feature blobs from git.
Use the links below and place files into `data/` locally.

## Links
- Video2Commonsense annotations:
  - https://drive.google.com/file/d/1qt0JsOAqBsdCTlDUw0gw7c_IosShysoW/view?usp=sharing
- ResNet152 features used by V2C baseline:
  - https://drive.google.com/file/d/1yUu4zMQzw_YOO5M8_i91ht3PmfkytolG/view?usp=sharing
- MSVD official project page:
  - https://www.cs.utexas.edu/~ml/clamp/videoDescription/

## Local Layout

```text
data/
  resnet152/
  V2C_MSR-VTT_caption.json
  v2c_info.json
  msrvtt_new_info.json

  # Optional: local videos (not required for training/eval).
  # Used only for generating README qualitative GIF clips.
  videos/
    video9703.mp4
    video7481.mp4
    video7116.mp4
```

## Rebuild split file

```bash
python -m videocap.prepare_data --data-root data --val-ratio 0.05 --seed 42
```

## Optional: Generate README GIFs

If you have local clips under `data/videos/`, you can generate small GIF thumbnails for README:

```bash
python scripts/make_gifs.py --video-dir data/videos --run-dir artifacts/<your-run-dir> --video-ids video9703,video7481,video7116 --split test --out-dir assets/gifs
```
