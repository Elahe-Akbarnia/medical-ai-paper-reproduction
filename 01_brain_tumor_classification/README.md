# A Multi-Layered Defense Against Adversarial Attacks in Brain Tumor Classification

Official implementation of:

"A multi-layered defense against adversarial attacks in brain tumor classification using ensemble adversarial training and feature squeezing"

Scientific Reports, 2025


## Overview

This repository implements a robust deep learning framework for brain MRI tumor classification under adversarial attacks.

The framework combines:

- Transfer learning using VGG16
- FGSM adversarial attacks
- PGD adversarial attacks
- Feature squeezing defense
- Ensemble adversarial training


The objective is improving robustness of medical AI systems against adversarial perturbations.


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


Images are resized to:

128 × 128


Pixel values are normalized:

0-1



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

# Installation


Clone repository:


```bash
git clone https://github.com/Elahe-Akbarnia/medical-ai-paper-reproduction/tree/main/01_brain_tumor_classification

cd brain-tumor-adversarial-defense
