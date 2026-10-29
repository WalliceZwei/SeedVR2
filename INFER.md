# SeedVR2 Inference Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install flash_attn==2.5.9.post1 --no-build-isolation
```

### 2. Download Model Checkpoints

```bash
# For 3B model
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    local_dir='./ckpts',
    repo_id='ByteDance-Seed/SeedVR2-3B',
    allow_patterns=['*ema_3b.pth', '*vae.pth', '*.json']
)
"
```

### 3. Prepare Input Videos

Place your videos in `./test_videos/` folder.

### 4. Run Inference

```bash
# Basic usage (3B model)
python inference.py --video_path ./test_videos --output_dir ./results

# With custom resolution
python inference.py --video_path ./test_videos --output_dir ./results --res_h 1080 --res_w 1920

# Multi-GPU (sequence parallel)
torchrun --nproc-per-node=4 inference.py --video_path ./test_videos --output_dir ./results --sp_size 4
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--video_path` | `./test_videos` | Folder containing input videos |
| `--output_dir` | `./results` | Output folder for restored videos |
| `--seed` | `666` | Random seed for reproducibility |
| `--res_h` | `720` | Output video height |
| `--res_w` | `1280` | Output video width |
| `--sp_size` | `1` | Sequence parallel size (multi-GPU) |
| `--out_fps` | `None` | Output FPS (uses input FPS by default) |

## GPU Requirements

| Model | Resolution | GPUs Required |
|-------|------------|---------------|
| 3B | 720p | 1 H100-80G |
| 3B | 1080p | 1-2 H100-80G (sp_size=2) |
| 3B | 2K | 4 H100-80G (sp_size=4) |
| 7B | 720p+ | Multiple GPUs recommended |

## Notes

- Output is saved to `./results/` with the same filename as input
- Supports arbitrary resolutions via `--res_h` and `--res_w`
- For best results, use videos with moderate degradations
