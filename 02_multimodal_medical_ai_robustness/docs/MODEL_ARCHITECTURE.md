# Model Architecture


This document describes the architecture of the implemented multimodal medical AI framework.


The framework consists of:

- Image modality encoder
- Text modality encoder
- Multimodal fusion modules
- Classification heads


---

# Overall Pipeline


Chest X-ray Images

↓

Clinical Text Reports

↓

Separate Modality Encoders

↓

Feature Extraction

↓

Multimodal Fusion

↓

Binary Classification

(Normal / Abnormal)



---

# Vision Model


## Backbone

SE-ResNet-154 pretrained model


The image model processes two chest X-ray views:

- Frontal view
- Lateral view


Each image is passed through an independent SE-ResNet-154 feature extractor.


Architecture:
Frontal X-ray

↓

SE-ResNet-154

↓

2048-dimensional feature vector

Lateral X-ray

↓

SE-ResNet-154

↓

2048-dimensional feature vector



The extracted image features are used for:

- Image-only classification
- Multimodal fusion



---

# Text Model


## Backbone

Bio_ClinicalBERT


The text model processes clinical reports associated with chest X-ray images.


Architecture:

Clinical Report

↓

Tokenizer

↓

Bio_ClinicalBERT

↓

CLS representation

↓

Classification layer



The CLS representation is used as the text feature representation.


Feature dimension:


768



---

# Multimodal Fusion


The repository implements three fusion strategies.


---

## Early Fusion


Early fusion combines modality features before classification.


Feature representation:



Frontal image features

Lateral image features

Text features

=

4864-dimensional vector



The fused representation is passed through a classification head.


Pipeline:



Image Features

Text Features

↓

Feature Concatenation

↓

Linear Classifier

↓

Prediction



---

## Late Fusion


Late fusion combines predictions from separately trained modality models.


Training procedure:


Stage 1:

Train image and text models independently.


Stage 2:

Freeze modality models and train the fusion classifier.


Pipeline:



Image Model

↓

Image Prediction

Text Model

↓

Text Prediction

↓

Fusion Classifier

↓

Prediction



---

## Ensemble Fusion


Ensemble fusion combines modality outputs without an additional fusion classifier.


Pipeline:



Image Logits

Text Logits

↓

Final Prediction



---

# Adversarial Attack Integration


The framework evaluates model robustness by applying attacks to different modalities.


## Image Attacks

Implemented:

- FGSM
- PGD


Attack target:

Chest X-ray images



## Text Attacks

Implemented:

- Synonym replacement
- Half-sentence deletion


Attack target:

Clinical text reports



---

# Evaluation Pipeline


Each experiment evaluates:


- Clean samples
- Image adversarial samples
- Text adversarial samples
- Combined adversarial samples


Metrics:

- Accuracy
- F1-score
- Precision
- Recall
- ROC-AUC
