# Pixel-Wise Binary Change Segmentation via Early Fusion of Electro-Optical and Synthetic Aperture Radar Imagery

**GalaxEye Space — Satellite AI Research Intern | Technical Assessment**

---

## Quickstart

```bash
# 1. Setup environment
python -m venv venv && venv\Scripts\activate     # Windows
pip install -r requirements.txt

# 2. Train from scratch
python -m src.train

# 3. Evaluate on test split
python -m src.eval --data_path ./dataset --weights ./outputs/unet/checkpoints/best_unet_model.pth

# 4. Run inference (generate prediction masks)
python -m src.inference --weights ./outputs/unet/checkpoints/best_unet_model.pth
```

---

## Project Description

Binary pixel-level change detection on co-registered EO and SAR satellite image pairs. Given pre-event and post-event imagery, the model classifies each pixel as **Changed (1)** or **Unchanged (0)** to identify disaster-induced building damage.

**Approach:** Early fusion — pre-event EO (RGB, 3-channel) and post-event SAR (grayscale, 1-channel) are concatenated into a 4-channel input tensor, processed by a **UNet with ImageNet-pretrained ResNet-34 encoder**. Class imbalance (~74% empty masks) is addressed at both data level (Weighted Random Sampling) and loss level (Focal + Dice combined loss).

---

## Table of Contents

1. [Requirements](#requirements)
2. [Environment Setup](#environment-setup)
3. [Dataset Structure](#dataset-structure)
4. [Label Remapping](#label-remapping)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Inference](#inference)
8. [Model Weights](#model-weights)
9. [Results](#results)
10. [Project Structure](#project-structure)
11. [Design Decisions](#design-decisions)
12. [Citations and References](#citations-and-references)

---

## Requirements

- **Python:** 3.10+
- **GPU:** NVIDIA GPU with CUDA (tested on GTX 1650, 4GB VRAM)
- **RAM:** Minimum 8GB

```
torch==2.1.0
torchvision==0.16.0
segmentation-models-pytorch==0.3.3
albumentations==1.3.1
tifffile==2023.4.12
numpy==1.24.3
opencv-python==4.8.0.76
scikit-learn==1.3.0
matplotlib==3.7.2
pandas==2.0.3
pyyaml==6.0
tqdm==4.65.0
```

---

## Environment Setup

**Using venv:**

```bash
git clone https://github.com/<your-username>/GalaxEye_Segmentation_Assignment.git
cd GalaxEye_Segmentation_Assignment

python -m venv venv

# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

**Using conda:**

```bash
conda create -n galaxeye python=3.10
conda activate galaxeye
pip install -r requirements.txt
```

> **PyTorch + CUDA version mismatch?** Install PyTorch separately first:
> ```bash
> pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Dataset Structure

Download from [Hugging Face](https://huggingface.co/datasets/doron333/change-detection-dataset) and place under `dataset/` in the project root:

```
GalaxEye_Segmentation_Assignment/
│
├── dataset/
│   ├── train/
│   │   ├── pre-event/       # EO RGB images (.tif)
│   │   ├── post-event/      # SAR grayscale images (.tif)
│   │   └── target/          # Annotation masks (.tif)
│   ├── val/
│   │   ├── pre-event/
│   │   ├── post-event/
│   │   └── target/
│   └── test/
│       ├── pre-event/
│       ├── post-event/
│       └── target/
```

**Key dataset statistics (train split):**

| Property | Value |
|---|---|
| Total image triplets | 2,781 |
| Image resolution | 512 × 512 px |
| EO modality | 3-channel RGB (uint8) |
| SAR modality | 1-channel grayscale (uint8) |
| Empty masks (no damage) | ~74.4% (2,068 images) |
| Non-empty masks (damage) | ~25.6% (713 images) |
| Mean damage pixel coverage | ~1.57% |

---

## Label Remapping

The original 4-class annotations are remapped to binary labels **automatically inside the dataloader**. No manual step required.

| Original Class | Original Value | Remapped Value | Remapped Class |
|---|---|---|---|
| Background | 0 | 0 | No-Change |
| Intact | 1 | 0 | No-Change |
| Damaged | 2 | 1 | Change |
| Destroyed | 3 | 1 | Change |

---

## Training

All hyperparameters are controlled via `configs/config.yaml`:

```yaml
dataset:
  image_size: 512
  num_workers: 0

model:
  name: "unet"
  encoder_name: "resnet34"
  encoder_weights: "imagenet"
  in_channels: 4
  classes: 1

training:
  batch_size: 2
  epochs: 25
  learning_rate: 0.001
  device: "cuda"
  mixed_precision: false
  seed: 42

scheduler:
  name: "cosine"

loss:
  focal_weight: 1.0
  dice_weight: 1.0

checkpoint:
  save_dir: "outputs/unet/checkpoints"
```

**Run training:**

```bash
python -m src.train
```

Training automatically:
- Oversamples high-damage images via `WeightedRandomSampler`
- Saves best checkpoint (by val IoU) to `outputs/unet/checkpoints/best_unet_model.pth`
- Logs epoch metrics to `outputs/unet/logs/unet_training_metrics.csv`
- **Resumes from checkpoint if one exists** (crash-safe)

**Plot training curves after training:**

```bash
python -m src.plot_metrics
```

---

## Evaluation

```bash
python -m src.eval --data_path ./dataset --weights ./outputs/unet/checkpoints/best_unet_model.pth
```

Outputs IoU, Precision, Recall, F1 Score, and Confusion Matrix on the test split.

---

## Inference

```bash
python -m src.inference --weights ./outputs/unet/checkpoints/best_unet_model.pth
```

Prediction masks saved to `outputs/unet/predictions/` as `.png` files. Inference applies:
- **Test-Time Augmentation (TTA):** horizontal-flip ensemble
- **Post-processing:** morphological opening + closing (3×3 kernel)
- **Decision threshold:** 0.5

---

## Model Weights

Download the final trained checkpoint (UNet + ResNet-34, best val IoU epoch 16):

> **[Download best_unet_model.pth — Google Drive](#)**
> *(Replace `#` with your actual public link before submission)*

Place the downloaded file at:

```
outputs/unet/checkpoints/best_unet_model.pth
```

---

## Results

All metrics computed for the **Change class (label = 1)** only. The final submitted model is **UNet + ResNet-34**. UNet++ + EfficientNet-B0 was trained as an exploratory comparison and ran for 5 epochs before being discontinued due to inferior convergence stability.

### Validation Split — Best Checkpoint (Epoch 16)

| Metric | UNet + ResNet-34 ✅ | UNet++ + EfficientNet-B0 |
|---|---|---|
| Val Loss | 0.9461 | 0.8928 |
| IoU | **0.3988** | 0.3608 |
| Precision | **0.5243** | 0.4992 |
| Recall | **0.6703** | 0.7229 |
| F1 Score | **0.5094** | 0.3951 |

### Test Split — Final Evaluation

| Metric | UNet + ResNet-34 ✅ | UNet++ + EfficientNet-B0 |
|---|---|---|
| IoU | **0.2464** | 0.2335 |
| Precision | **0.5179** | 0.3315 |
| Recall | **0.4096** | 0.3898 |
| F1 Score | **0.2542** | 0.2362 |

### UNet Confusion Matrix (Test Split)

```
                      Predicted No-Change    Predicted Change
Actual No-Change           19,976,491              56,258
Actual Change                 149,895               2,444
```

### Error Analysis

- **False Negatives (missed damage):** Model misses low-contrast damage zones where SAR backscatter variation is subtle — the early-fusion pathway cannot disentangle cross-sensor signals from background noise in these cases.
- **False Positives (false alarms):** Occur along land-cover boundaries and in high-speckle SAR regions where radar coherence noise mimics structural change signatures.
- **Val → Test performance drop:** The gap (Val IoU 0.3988 → Test IoU 0.2464) reflects cross-event distributional shift and the extreme foreground sparsity — only ~2,444 true positive change pixels exist against ~20 million background pixels in the test split.

---

## Project Structure

```
GalaxEye_Segmentation_Assignment/
│
├── configs/
│   └── config.yaml                      # All hyperparameters
│
├── notebooks/
│   └── exploration.ipynb                # EDA and dataset visualisations
│
├── src/
│   ├── datasets/
│   │   ├── change_detection_dataset.py  # Dataset, label remapping, augmentation
│   │   └── samplers.py                  # Weighted random sampler
│   │
│   ├── models/
│   │   ├── unet.py                      # UNet + ResNet-34 (final model)
│   │   └── unetplusplus.py              # UNet++ + EfficientNet-B0 (exploratory)
│   │
│   ├── losses/
│   │   └── losses.py                    # DiceLoss, FocalLoss, FocalDiceLoss
│   │
│   ├── metrics/
│   │   └── metrics.py                   # IoU, Precision, Recall, F1, Confusion Matrix
│   │
│   ├── utils/
│   │   ├── checkpoint.py                # Save / resume checkpoints
│   │   ├── logger.py                    # CSV metric logger
│   │   ├── scheduler.py                 # LR scheduler factory
│   │   ├── seed.py                      # Global reproducibility seed
│   │   └── visualization.py            # EO / SAR / GT / prediction overlay plots
│   │
│   ├── train.py                         # Training loop with crash-safe resumption
│   ├── eval.py                          # CLI evaluation script
│   ├── inference.py                     # Inference with TTA and post-processing
│   └── plot_metrics.py                  # Training curve plots from CSV logs
│
├── outputs/                             # Auto-created during training (gitignored)
│   └── unet/
│       ├── checkpoints/
│       ├── predictions/
│       ├── plots/
│       └── logs/
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Design Decisions

**Architecture — UNet + ResNet-34 (over UNet++):** UNet++'s nested dense skip connections introduced optimisation instability on sparse supervision signals under the limited epoch budget (batch size 2, GTX 1650). UNet's simpler decoder converged more reliably and produced consistently higher val and test IoU. ResNet-34 was chosen over EfficientNet-B0 for better gradient flow and feature map stability on 4-channel multi-modal inputs.

**Fusion — Early Fusion (4-channel input):** Pre-event EO (3ch) and post-event SAR (1ch) are concatenated before the encoder. Simpler and faster to train than Siamese dual-stream architectures — a deliberate trade-off under strict time and compute constraints.

**Class Imbalance — Two-Level Strategy:**
- *Data level:* `WeightedRandomSampler` assigns higher sampling weight to images with more damage pixels, ensuring balanced gradient exposure each epoch.
- *Loss level:* Focal Loss down-weights easy background pixels; Dice Loss directly maximises spatial overlap on the sparse change class. Equal weights (1.0 + 1.0).

**Augmentation:** Horizontal flip, vertical flip, random 90° rotation on training split only. Heavy augmentations avoided to preserve EO-SAR spatial co-registration integrity.

**Optimizer / Scheduler:** AdamW at lr=0.001, cosine annealing over 25 epochs. Mixed precision disabled for numerical stability on multi-modal radar + optical inputs.

---

## Citations and References

**Architectures:**
- Ronneberger et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI. https://arxiv.org/abs/1505.04597
- Zhou et al. (2018). *UNet++: A Nested U-Net Architecture.* DLMIA. https://arxiv.org/abs/1807.10165
- He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR. https://arxiv.org/abs/1512.03385

**Loss Functions:**
- Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV. https://arxiv.org/abs/1708.02002
- Milletari et al. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.* 3DV. https://arxiv.org/abs/1606.04797

**Change Detection:**
- Peng et al. (2019). *End-to-End Change Detection for High Resolution Satellite Images Using Improved UNet++.* Remote Sensing. https://doi.org/10.3390/rs11111382
- Chen et al. (2021). *Remote Sensing Image Change Detection with Transformers.* IEEE TGRS. https://arxiv.org/abs/2103.00208

**Libraries:**
- Iakubovskii (2019). *Segmentation Models PyTorch.* https://github.com/qubvel/segmentation_models.pytorch
- Buslaev et al. (2020). *Albumentations: Fast and Flexible Image Augmentations.* https://doi.org/10.3390/info11020125

**Dataset:** https://huggingface.co/datasets/doron333/change-detection-dataset

---

*Submitted by Krit Prasad | kritrp05@gmail.com | GalaxEye Space — Satellite AI Research Intern | May 2026*
