# A Multi-Layered Defense Against Adversarial Attacks in Brain Tumor Classification

Official implementation of:

"A multi-layered defense against adversarial attacks in brain tumor classification using ensemble adversarial training and feature squeezing"

Scientific Reports, 2025


## Overview

This repository provides a complete implementation of the proposed methodology. Experimental validation requires a GPU-enabled environment.

The framework combines:

- Transfer learning using VGG16
- FGSM adversarial attacks
- PGD adversarial attacks
- Feature squeezing defense
- Ensemble adversarial training


The objective is improving robustness of medical AI systems against adversarial perturbations.


---
## Project Status

Implementation status:

- Dataset loading and preprocessing pipeline  
- VGG16 transfer learning architecture  
- FGSM attack implementation  
- PGD attack implementation  
- Feature squeezing defense  
- Adversarial training pipeline  
- Evaluation scripts  


Note:

Full experimental validation requires a GPU-enabled environment.

---

# Method Pipeline


MRI Images

↓

Image preprocessing

↓

VGG16 Feature Extractor

↓

Brain Tumor Classifier

↓

Adversarial Attacks

(FGSM + PGD)

↓

Feature Squeezing

(Bit-depth reduction + Gaussian blur)

↓

Adversarial Training

↓

Robust Evaluation



---

# Dataset


The experiments use a composite brain MRI dataset containing:

- Glioma
- Meningioma
- Pituitary tumor
- No tumor


The dataset combines:

- Figshare
- SARTAJ
- Br35H


Total images:

7023


Preprocessing:

- Resize images to 128 × 128 pixels
- Normalize pixel values to the range [0,1]

---

## Dataset Download

The dataset is not included in this repository due to size limitations.

Please download the dataset from Kaggle and organize it as follows:


```
data/

└── raw/

├── glioma/

├── meningioma/

├── pituitary/

└── no_tumor/
```

The training scripts automatically load images from this directory.

---

# Model Architecture


Backbone:

VGG16 pretrained on ImageNet


Classifier:


Global Average Pooling

↓

Dense(128, ReLU)

↓

Dropout(0.5)

↓

Dense(4, Softmax)



---

# Training Strategy


## Baseline Training


Stage 1:

Frozen VGG16 backbone

Epochs:
5


Stage 2:

Fine tuning last 10 layers

Epochs:
10



---

# Adversarial Attacks


## FGSM


Parameters:

epsilon:

0.01


## PGD


Parameters:


epsilon:

0.01


alpha:

0.002


iterations:

10



---

# Defense Strategy


Feature squeezing:


Bit depth:

4 bits


Gaussian blur:

3×3 kernel



Adversarial training uses:


- clean images
- FGSM images
- PGD images

---
# Repository Structure

```
01_brain_tumor_classification/

│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/
│   ├── config.yaml
│   ├── training.yaml
│   └── attacks.yaml
│
├── data/
│   └── README.md
│
├── notebooks/
│
├── src/
│   ├── data/
│   ├── models/
│   ├── attacks/
│   ├── defenses/
│   ├── evaluation/
│   └── utils/
│
├── scripts/
│   ├── train_baseline.py
│   ├── train_defense.py
│   ├── evaluate.py
│   └── run_experiment.py
│
├── experiments/
│   └── README.md
│
├── checkpoints/
│   └── .gitkeep
│
└── results/
    ├── figures/
    ├── tables/
    └── README.md
```
---

# Installation


Clone repository:


```bash
git clone https://github.com/Elahe-Akbarnia/medical-ai-paper-reproduction/tree/main/01_brain_tumor_classification

cd brain-tumor-adversarial-defense
