# CSE 144 Final Project — ViT Transfer Learning (Frozen Feature Extraction)

100-class image classification using a SWAG-pretrained Vision Transformer (ViT-B/16) as a frozen feature extractor, with a lightweight linear classifier head trained on cached features.

## Team

Ashmita Dua

## Kaggle Leaderboard

![Kaggle Leaderboard — Rank 31, Score 0.82727](leaderboard.png)

## Trained Model Weights

The trained classifier head weights (`best_vit_model.pth`) are hosted on Google Drive:

**[Download best_vit_model.pth](YOUR_GOOGLE_DRIVE_LINK_HERE)**

> Replace the link above with a shareable Google Drive link to your `.pth` file. To get one: upload the file to Drive → right-click → "Share" → "Anyone with the link" → "Copy link".

---

## Method Overview

| Component | Choice |
|-----------|--------|
| Backbone | ViT-B/16, fully frozen (SWAG IMAGENET1K_SWAG_E2E_V1 weights) |
| Image size | 384×384 (matches SWAG pretraining resolution) |
| Feature dim | 768-d CLS token |
| Training augmentation | 4 views per image (center crop, h-flip, 2× random crop + jitter) |
| Classifier head | Dropout(0.3) → Linear(768, 100) |
| Optimizer | AdamW, lr=1e-3, wd=1e-3 |
| Scheduler | CosineAnnealingLR over 60 epochs |
| Loss | CrossEntropyLoss with label smoothing 0.1 |
| Validation split | 15% image-level holdout |
| Inference TTA | 2 views (center crop + h-flip), softmax-averaged |
| Best val accuracy | 77.0% |
| Kaggle public score | **0.82727** (rank 31) |

The key insight is that SWAG-pretrained ViT features are powerful enough that keeping the backbone completely frozen and training only a ~77K-parameter linear head avoids overfitting in this ~10-images-per-class few-shot regime.

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download the Kaggle competition data and place it so the directory structure looks like:

```
ucsc-cse-144-spring-2026-final-project/
    train/
        0/   (images for class 0)
        1/
        ...
        99/
    test/
        0.jpg
        1.jpg
        ...
    sample_submission.csv
```

---

## Training

Open and run `vit_frozen_features.ipynb` end-to-end. The notebook:

1. Loads the frozen ViT-B/16 SWAG backbone.
2. Extracts and caches features for 4 augmented views of every training image.
3. Trains a linear head on the cached features (60 epochs, ~seconds per epoch).
4. Retrains the head on all data (no holdout) for the final submission model.
5. Saves the classifier head to `best_vit_model.pth`.
6. Generates `submission.csv` using 2-view TTA on the test set.

```bash
jupyter notebook vit_frozen_features.ipynb
```

The notebook auto-selects MPS (Apple Silicon), CUDA, or CPU.

---

## Inference (with pre-trained weights)

To run inference using the downloaded weights without re-training:

1. Download `best_vit_model.pth` from the Google Drive link above and place it in the repo root.
2. Run only the inference cells in `vit_frozen_features.ipynb`:
   - **Section 1–4** (imports, config, dataset classes, frozen backbone)
   - **Section 10** (test predictions with TTA)
   - **Section 11** (save `submission.csv`)

Or run the equivalent Python snippet:

```python
import torch, torch.nn as nn, torchvision.models as models
from torchvision.models import ViT_B_16_Weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load backbone
weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
backbone = models.vit_b_16(weights=weights)
backbone.heads = nn.Identity()
backbone = backbone.to(device).eval()

# Load trained head
head = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, 100)).to(device)
head.load_state_dict(torch.load('best_vit_model.pth', map_location=device))
head.eval()

# Pass a (1, 3, 384, 384) image tensor through backbone then head to get logits
# logits = head(backbone(image_tensor))
```

---

## Repository Structure

```
cse-144-final/
├── vit_frozen_features.ipynb   # Training + inference notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── leaderboard.png             # Kaggle leaderboard screenshot (add this)
└── report.pdf                  # Project report (add this)
```
