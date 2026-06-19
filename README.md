# Image Blur vs. Sharp Classification Using Neural Networks

A lightweight machine learning system that classifies images as **Sharp** or **Blurred** using handcrafted features from both the spatial and frequency domains, fed into a Multilayer Perceptron (MLP).

This project explores the real-world challenge of **domain shift** — where a model trained on clean data struggles with noisy real-world images — and engineers a solution using **Adaptive Thresholding via Youden's J Statistic**.

---

## Results Summary

| Experiment | Dataset | Accuracy |
|---|---|---|
| Initial Training | Kaggle Blur Dataset (5-Fold CV) | **94.48%** |
| Cross-Domain Test | Kaggle model → KADID-10K | ~70% (domain shift) |
| Reverse Training | KADID-10K (5-Fold CV) | **93.31%** |
| Final Live Demo (with adaptive threshold) | Cross-domain | **90.00%** |

---

## How It Works

### 1. Feature Extraction
Instead of using a CNN (which would require GPU resources and large datasets), this project extracts 6 handcrafted mathematical features per image:

**Spatial Domain:**
- `Grad_Mean` — Mean gradient magnitude (Sobel filter)
- `Grad_Max` — Maximum gradient magnitude
- `Local_Var` — Local pixel variance
- `Zero_Cross` — Laplacian zero-crossing rate (edge density)

**Frequency Domain (FFT):**
- `High_Freq` — Total high-frequency energy
- `Power_Dist` — Low-frequency power concentration ratio

The intuition: blurred images lose sharp edges and high-frequency detail, so these features directly measure that loss.

### 2. Model Architecture (MLP)
```
Input (6 features)
    ↓
Dense(64) + BatchNorm + ReLU + Dropout(0.2)
    ↓
Dense(32) + ReLU
    ↓
Dense(1) + Sigmoid  →  Sharpness probability
```
Trained with **Adam optimizer**, **Binary Cross-Entropy loss**, and **Early Stopping** (patience=10).

### 3. Rigorous Evaluation
- **5-Fold Stratified Cross-Validation** — avoids lucky splits, ensures reliability
- Tested on **two different datasets** to measure generalization

### 4. Adaptive Thresholding (Key Innovation)
A standard 0.5 threshold fails when switching between clean and noisy image domains. This project automatically computes the **optimal decision boundary** using Youden's J Statistic from the ROC curve:

- For **clean images**: threshold rises to ~0.85 (stricter)
- For **noisy/distorted images**: threshold drops to ~0.15 (more tolerant)

This eliminated the domain shift problem, recovering accuracy from ~70% to **90%**.

---

## Datasets

| Dataset | Images | Source |
|---|---|---|
| [Kwentar Blur Dataset](https://www.kaggle.com/datasets/kwentar/blur-dataset) | ~2,100 | Kaggle |
| [KADID-10K](https://www.kaggle.com/datasets/kadid10k/kadid10k) | 10,125 | Kaggle |

The KADID-10K dataset contains 25 types of real-world distortions. Blur distortion types (1, 2, 3) were mapped to Class 0 (Blurred), all others to Class 1 (Sharp).

---

## Setup & Usage

### Run in Google Colab (Recommended)
Click the badge below to open directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SamarSaleh00/blur-detection-mlp/blob/main/blur_detection_mlp.ipynb)


### Requirements
```
tensorflow
opencv-python
scikit-learn
numpy
pandas
matplotlib
kagglehub
```

You will need a [Kaggle account](https://www.kaggle.com) and API token to download the datasets automatically via `kagglehub`.

---

## Project Structure

```
blur-detection-mlp/
│
├── blur_detection_mlp.ipynb   # Main notebook (full code + outputs)
├── README.md                  # This file
└── documentation.pdf          # Full project report with theory & results
```

---

## Key Concepts Demonstrated

- **Feature Engineering** over raw pixel data
- **Frequency Domain Analysis** using FFT
- **Multilayer Perceptron** design and training
- **Cross-Validation** for honest performance estimation
- **Domain Shift** identification and mitigation
- **ROC Analysis** and optimal threshold selection

---

## Author

**Samar Kh. Saleh**  
Data Science and Artificial Intelligence — University College of Applied Sciences, Gaza  
