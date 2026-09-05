# Adversarial Robustness of Multimodal Medical AI Systems

Implementation of:

"Assessing the adversarial robustness of multimodal medical AI systems: insights into vulnerabilities and modality interactions"

Frontiers in Medicine, 2025


## Overview

This repository provides an implementation of a multimodal medical AI framework for chest X-ray classification and adversarial robustness evaluation.

Experimental validation requires a GPU-enabled environment.

The framework combines:

- Dual-view chest X-ray classification
- Bio_ClinicalBERT clinical text representation
- Multimodal fusion strategies
- FGSM and PGD adversarial attacks
- Text adversarial attack modules


The objective is evaluating the robustness of multimodal medical AI systems against adversarial perturbations.


---

## Project Status

Implementation status:

- Dataset loading and preprocessing pipeline
- Dual-view chest X-ray processing
- SE-ResNet image classification pipeline
- Bio_ClinicalBERT text model implementation
- Early fusion implementation
- Late fusion implementation
- Ensemble fusion implementation
- Image adversarial attack modules
- Text adversarial attack modules
- Evaluation pipeline


Current evaluated experiment:

- Image-only dual-view chest X-ray baseline


Note:

Full multimodal and adversarial experiments require further GPU-enabled validation.

---

# Method Pipeline


Chest X-ray Images

+

Clinical Text Reports

↓

Preprocessing

↓

SE-ResNet Image Encoder

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
- Frontal/lateral view pairing
- Text preprocessing
- Label generation from clinical findings


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

SE-ResNet152d pretrained model


Note:

The original methodology uses SE-ResNet-154.
Due to pretrained checkpoint availability limitations,
this implementation uses SE-ResNet152d as the available SE-ResNet backbone.


Input:

- Frontal X-ray
- Lateral X-ray


Output:

Dual-view image feature representation


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

Image features

+

Text features

↓

Feature concatenation

↓

Classifier



### Late Fusion

Two-stage training:


Image model

+

Text model

↓

Fusion classifier



### Ensemble Fusion

Prediction-level combination:


Image prediction

+

Text prediction

↓

Final prediction



---

# Training Strategy


## Image Baseline Model


Backbone:

SE-ResNet152d


Optimizer:

Adam


Learning rate:

1e-4


Epochs:

13


Batch size:

128



Note:

The first evaluated experiment focuses on the image-only baseline using frontal and lateral chest X-ray views.


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


## Image Attacks


Implemented:

- FGSM
- PGD


Attack target:

Chest X-ray images



Parameters:

epsilon:

- 8/255
- 0.2



PGD:

iterations:

10



---

## Text Attacks


Implemented:

- Synonym replacement
- Half sentence deletion


Attack target:

Clinical text reports



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
- Image adversarial samples
- Text adversarial samples
- Combined adversarial samples



---

# Current Baseline Result


The first completed experiment evaluates the dual-view image classification pipeline.


Dataset:

Indiana University Chest X-ray Dataset


Input:

- Frontal X-ray
- Lateral X-ray


Validation performance:


| Metric | Value |
|---|---:|
| Accuracy | 0.7109 |
| F1-score | 0.7888 |
| ROC-AUC | 0.7618 |


The experiment showed strong training performance with lower validation performance, indicating overfitting during fine-tuning.


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
├── docs/
│ ├── REPRODUCTION.md
│ └── MODEL_ARCHITECTURE.md
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
├── experiments/
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

---

# Reproducibility Note


This repository provides the implementation of the proposed multimodal medical AI framework.

The image baseline experiment has been successfully executed in a GPU-enabled environment.

Due to computational requirements, full multimodal fusion and adversarial robustness experiments require additional validation.

The original paper uses SE-ResNet-154. This implementation uses SE-ResNet152d because of publicly available pretrained weights.
