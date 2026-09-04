# Adversarial Robustness of Multimodal Medical AI Systems

Implementation of:

"Assessing the adversarial robustness of multimodal medical AI systems: insights into vulnerabilities and modality interactions"

Frontiers in Medicine, 2025


## Overview

This repository provides a complete implementation of the proposed multimodal adversarial robustness framework.

Experimental validation requires a GPU-enabled environment.

The framework combines:

- SE-ResNet-154 chest X-ray classification
- Bio_ClinicalBERT clinical text classification
- Multimodal fusion strategies
- FGSM and PGD adversarial attacks
- Text adversarial attacks


The objective is evaluating the robustness of multimodal medical AI systems against adversarial perturbations.


---
## Project Status

Implementation status:

- Dataset loading and preprocessing pipeline
- SE-ResNet-154 vision model
- Bio_ClinicalBERT text model
- Early fusion implementation
- Late fusion implementation
- Ensemble fusion implementation
- Image adversarial attacks
- Text adversarial attacks
- Evaluation pipeline


Note:

Full experimental validation requires a GPU-enabled environment.

---

# Method Pipeline


Chest X-ray Images

+

Clinical Text Reports

↓

Preprocessing

↓

SE-ResNet-154 Image Encoder

+

Bio_ClinicalBERT Text Encoder

↓

Multimodal Fusion

↓

Adversarial Attacks

(FGSM + PGD + Text Attacks)

↓

Robust Evaluation


---

# Dataset


The experiments use the Indiana University Chest X-ray Dataset.


The dataset contains:

- Frontal X-ray images
- Lateral X-ray images
- Clinical reports


The dataset is created by combining:

- indiana_reports.csv
- indiana_projections.csv


Preprocessing:

- Image resizing
- Image normalization
- Text lowercase conversion
- Whitespace cleaning


---

## Dataset Download

The dataset is not included in this repository due to size limitations.

Organize the dataset as follows:

```
data/

└── raw/

├── images/

├── indiana_reports.csv

└── indiana_projections.csv
```


The preprocessing pipeline automatically creates the multimodal dataset.

---

# Model Architecture


## Vision Model


Backbone:

SE-ResNet-154


Input:

- Frontal X-ray
- Lateral X-ray


Output:

Image feature representation


---

## Text Model


Backbone:

Bio_ClinicalBERT


Input:

Clinical text reports


Output:

CLS feature representation


---

## Multimodal Fusion


Implemented fusion strategies:


### Early Fusion

Feature-level fusion:
Image features + Text features

↓

Classifier



### Late Fusion

Two-stage training:

Image model

Text model

↓

Fusion classifier



### Ensemble Fusion

Prediction-level combination:

Image prediction

Text prediction



---

# Training Strategy


## Image Model

Optimizer:

Adam


Learning rate:

1e-4


Epochs:

13


Batch size:

128



---

## Text Model


Optimizer:

Adam


Learning rate:

2e-5


Epochs:

5



---

# Adversarial Attacks


## FGSM


Parameters:

epsilon:

- 8/255
- 0.2



## PGD


Parameters:

epsilon:

- 8/255
- 0.2


iterations:

10



## Text Attacks


Implemented:

- Synonym replacement
- Half sentence deletion


---

# Evaluation


Metrics:

- Accuracy
- F1-score
- Precision
- Recall
- ROC-AUC


Evaluation includes:

- Clean samples
- Image attacks
- Text attacks
- Combined attacks


---

# Repository Structure

```

02_multimodal_medical_ai_robustness/

│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── .gitignore
│
├── configs/
│
├── data/
│ └── README.md
│
├── notebooks/
│
├── src/
│ ├── train.py
│ ├── test.py
│ └── evaluate_attacks.py
│
├── datasets/
│
├── models/
│
├── attacks/
│
├── engine/
│
├── scripts/
│
├── checkpoints/
│ └── .gitkeep
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
git clone https://github.com/Elahe-Akbarnia/medical-ai-paper-reproduction

cd multimodal-medical-ai-robustness
```

Install dependencies:

```
pip install -r requirements.txt
```
or:
```
conda env create -f environment.yml
```
