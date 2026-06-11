# CSE 144 Final Project

[Report & Weights (Google Drive)](https://drive.google.com/drive/folders/1BjIqZdYgbDBN4LeKBsoVe0rSGbXeAZcB?usp=sharing)

## Kaggle Leaderboard

![Kaggle Leaderboard](leaderboard.png)

## Setup

```bash
pip install -r requirements.txt
```

Download the Kaggle competition data and place it so the directory structure looks like:

```
ucsc-cse-144-spring-2026-final-project/
    train/
        0/
        1/
        ...
        99/
    test/
        0.jpg
        1.jpg
        ...
    sample_submission.csv
```

## Training

Open and run all cells in `vit_dual_backbone.ipynb`:

```bash
jupyter notebook vit_dual_backbone.ipynb
```

This will extract features from both backbones (DINOv2 ViT-L/14 + SWAG ViT-B/16), train an ensemble of 5 linear heads on the cached features, and save the weights to `best_vit_model.pth` and predictions to `submission.csv`.

## Inference

Download `best_vit_model.pth` and place it in the repo root, then run only the following sections of `vit_dual_backbone.ipynb`:

- **Section 1–4** (imports, config, dataset classes, backbones)
- **Section 10** (test predictions with TTA)
- **Section 11** (save `submission.csv`)

Or use this Python snippet directly:

```python
import torch, torch.nn as nn, torchvision.models as models
from torchvision.models import ViT_B_16_Weights

device = torch.device('mps' if torch.backends.mps.is_available()
                      else 'cuda' if torch.cuda.is_available() else 'cpu')

# DINOv2 ViT-L/14 (1024-dim)
dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dino = dino.to(device).eval()

# SWAG ViT-B/16 (768-dim)
swag = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
swag.heads = nn.Identity()
swag = swag.to(device).eval()

# Load ensemble of 5 heads (1792-dim input)
checkpoint = torch.load('best_vit_model.pth', map_location=device)
heads = []
for k in range(5):
    h = nn.Sequential(nn.Dropout(0.3), nn.Linear(1792, 100)).to(device)
    h.load_state_dict(checkpoint[f'head_{k}'])
    h.eval()
    heads.append(h)

# Pass images through both backbones, concatenate, then average ensemble:
# dino_feat = dino(image_224)          # (B, 1024)
# swag_feat = swag(image_384)          # (B, 768)
# feat = torch.cat([dino_feat, swag_feat], dim=1)  # (B, 1792)
# logits = sum(h(feat) for h in heads) / 5
```
