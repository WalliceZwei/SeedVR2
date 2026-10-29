# Model Checkpoints

Download the SeedVR2 model checkpoints here:

## 3B Model
```python
from huggingface_hub import snapshot_download
snapshot_download(
    local_dir="./ckpts",
    repo_id="ByteDance-Seed/SeedVR2-3B",
    allow_patterns=["*ema_3b.pth", "*vae.pth", "*.json"],
)
```

## 7B Model
```python
from huggingface_hub import snapshot_download
snapshot_download(
    local_dir="./ckpts",
    repo_id="ByteDance-Seed/SeedVR2-7B",
    allow_patterns=["*ema_7b.pth", "*vae.pth", "*.json"],
)
```
