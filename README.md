# Enhanced EfficientNet-B4 for AI-Generated Media Detection

> **Thesis Research Project** — Detection of AI-generated images and videos on social media using a spatiotemporal deep learning framework with Explainable AI.
>
> **Authors:** Merka, Nathaniel Keene M. · Mallari, Neil Ian R. · Sevellino, Kent Lenoel C.
> **Adviser:** Ramcis N. Vilchez, DIT

---

## Overview

This repository extends the [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) library for a thesis research project on detecting AI-generated media. The project compares two model configurations:

| Model | Description |
|---|---|
| **Baseline** | Frozen EfficientNet-B4 backbone with a trainable classification head (image-level, no temporal modeling) |
| **Proposed** | EfficientNet-B4 + Temporal Shift Module (TSM) + Multi-Head Self-Attention (MHSA) — a unified spatiotemporal framework for both images and videos |

The goal is to classify media as **Real** or **AI-Generated**, and to explain predictions using Grad-CAM heatmaps, attention weights, and Gemini LLM-generated text explanations.

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Models](#models)
   - [Baseline: Image-Only EfficientNet-B4](#baseline-image-only-efficientnet-b4)
   - [Proposed: Enhanced Spatiotemporal Model](#proposed-enhanced-spatiotemporal-model)
   - [Ablation Configurations](#ablation-configurations)
3. [Dataset](#dataset)
4. [Training](#training)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Installation](#installation)
7. [References](#references)

---

## Project Structure

```
EfficientNet-PyTorch/
├── efficientnet_pytorch/        # Base EfficientNet library (lukemelas)
├── thesis_experiment/
│   ├── models.py                # Baseline + Proposed model definitions
│   ├── train.py                 # Training script (all 4 ablation configs)
│   ├── dataset.py               # Video frame dataset loader
│   ├── data/                    # Primary dataset (real/ and fake/ splits)
│   ├── data_v2/                 # Secondary dataset version
│   └── output/                  # Saved checkpoints per config
└── README.md                    # This file
```

---

## Models

### Baseline: Image-Only EfficientNet-B4

The baseline treats every media file as a **static image**. For video inputs, frames are evaluated independently with no cross-frame information exchange.

**Architecture:**
```
Input (B, T, 3, 380, 380)
    → EfficientNet-B4 Backbone (frozen, 19.3M params)
    → Global Average Pooling  → (B, T, 1792)
    → Temporal Mean Pooling   → (B, 1792)
    → FC(1792→512) + GELU + Dropout(0.3)
    → FC(512→128)  + GELU + Dropout(0.2)
    → FC(128→2)    + Softmax
    → [Real | AI-Generated]
```

**Limitations addressed by the proposed model:**
- Evaluates video frames as independent static images, blind to temporal inconsistencies
- Cannot detect motion-based artifacts (flickering, morphing between frames)
- Susceptible to single-generator overfitting when trained only on GAN or diffusion data

**Load the baseline:**
```python
from thesis_experiment.models import build_model

model = build_model(config='baseline', num_frames=16)
```

---

### Proposed: Enhanced Spatiotemporal Model

The proposed model extends the baseline with two enhancements inserted between the backbone and the classifier:

1. **Temporal Shift Module (TSM)** — shifts 25% of feature channels forward and backward along the temporal axis, enabling zero-parameter inter-frame information exchange. Acts as a no-op for single images (T=1).

2. **Multi-Head Self-Attention (MHSA)** — 4-head self-attention across the frame sequence, with LayerNorm and a Feed-Forward Network (FFN). Learns which frames contain the most forensically relevant artifacts and captures long-range temporal dependencies.

**Architecture:**
```
Input (B, T, 3, 380, 380)
    → EfficientNet-B4 Backbone (frozen, 19.3M params)
    → Global Average Pooling         → (B, T, 1792)
    ┌─────────────────────────────────────────────┐  ★ Proposed
    │ Temporal Shift Module (TSM)   → (B, T, 1792) │  Contribution
    │ Multi-Head Self-Attention     → (B, T, 1792) │
    │   4 heads · d_k=448 · LayerNorm + FFN        │
    └─────────────────────────────────────────────┘
    → Temporal Mean Pooling          → (B, 1792)
    → FC(1792→512) + GELU + Dropout(0.3)
    → FC(512→128)  + GELU + Dropout(0.2)
    → FC(128→2)    + Softmax
    → [Real | AI-Generated]
```

**Key properties:**
- Works for **both images and videos** in a single unified architecture
- TSM adds **zero additional parameters**
- MHSA attention weights are exported for XAI (temporal explainability)
- Backbone is frozen; only the TSM, MHSA, and classifier head are trained

**Load the proposed model:**
```python
from thesis_experiment.models import build_model

model = build_model(config='proposed', num_frames=16)
```

---

### Ablation Configurations

Four configurations are available for comparative analysis:

| Config | TSM | MHSA | Purpose |
|---|---|---|---|
| `baseline` | ❌ | ❌ | Image-only baseline — no temporal modeling |
| `tsm_only` | ✅ | ❌ | Isolate TSM contribution |
| `mhsa_only` | ❌ | ✅ | Isolate MHSA contribution |
| `proposed` | ✅ | ✅ | Full proposed model |

```python
from thesis_experiment.models import build_model

baseline  = build_model(config='baseline')
tsm_only  = build_model(config='tsm_only')
mhsa_only = build_model(config='mhsa_only')
proposed  = build_model(config='proposed')
```

---

## Dataset

Data is organized under `thesis_experiment/data/` with the following structure:

```
data/
├── train/
│   ├── real/       # Authentic images or videos
│   └── fake/       # AI-generated images or videos
└── val/
    ├── real/
    └── fake/
```

- **Sources:** Publicly available GAN-based and diffusion-based synthetic media datasets
- **Formats supported:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, and standard image formats
- **Input resolution:** 380×380 (EfficientNet-B4 native resolution)
- **Frames per video:** 16 evenly-spaced frames sampled per video clip
- **Class balancing:** Explicit sample weighting applied during training to counteract class imbalance

---

## Training

### Quick Smoke Test (no real data required)
Verifies the full pipeline (forward pass, backward pass, VRAM usage) across all 4 configs using synthetic random tensors:

```bash
cd thesis_experiment
python train.py --smoke-test
```

### Train the Baseline (image-only)
```bash
python train.py --config baseline --data-dir data --epochs 20 --batch-size 4 --lr 1e-4
```

### Train the Proposed Model
```bash
python train.py --config proposed --data-dir data --num-frames 16 --epochs 20 --batch-size 4 --lr 1e-4
```

### Train All Ablation Configs
```bash
for config in baseline tsm_only mhsa_only proposed; do
    python train.py --config $config --data-dir data --epochs 20
done
```

### Optional: Load Pretrained Backbone Weights
```bash
python train.py --config proposed --backbone-weights /path/to/pretrained_b4.pth
```

**Training arguments:**

| Argument | Default | Description |
|---|---|---|
| `--config` | `proposed` | Model configuration (`baseline`, `tsm_only`, `mhsa_only`, `proposed`) |
| `--data-dir` | `data` | Dataset root directory |
| `--num-frames` | `16` | Frames sampled per video |
| `--batch-size` | `4` | Training batch size |
| `--epochs` | `20` | Number of training epochs |
| `--lr` | `1e-4` | Learning rate (AdamW optimizer) |
| `--num-workers` | `2` | Dataloader worker threads |
| `--backbone-weights` | `None` | Path to pretrained backbone checkpoint |
| `--smoke-test` | `False` | Run pipeline verification with synthetic data |

Best model checkpoints are saved to `thesis_experiment/output/<config>/best_model.pth` based on validation AUC.

---

## Evaluation Metrics

Both the baseline and proposed models are evaluated using:

| Metric | Description |
|---|---|
| **Accuracy** | Overall classification correctness |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of Precision and Recall |
| **AUC** | Area under the ROC curve |

Comparative analysis is conducted to quantify the impact of TSM, MHSA, and the full proposed model over the baseline. Models are also tested under real-world social media distortions (compression, resolution variation) to assess robustness.

---

## Installation

**Requirements:** Python 3.8+, PyTorch 2.0+, CUDA (recommended)

```bash
# Clone the repository
git clone https://github.com/Siege365/Enhanced-EfficientNet-PyTorch
cd Enhanced-EfficientNet-PyTorch

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install efficientnet_pytorch torch torchvision
pip install opencv-python scikit-learn numpy
```

---

## References

1. Tan, M. & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
2. Lin, J., Gan, C., & Han, S. (2022). TSM: Temporal Shift Module for Efficient Video Understanding. *IEEE TPAMI*, 45(1), 133–145.
3. Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
4. Corvi, R. et al. (2023). On the Detection of Synthetic Images Generated by Diffusion Models. *IEEE ICASSP*.
5. Ojha, U., Li, Y., & Lee, Y. J. (2023). Towards Universal Fake Image Detectors that Generalize Across Generative Models. *CVPR*.
6. Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*.

---

*This project builds upon the [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch) library by Luke Melas-Kyriazi.*
