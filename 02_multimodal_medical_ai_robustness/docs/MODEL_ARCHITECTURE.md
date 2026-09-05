# Model Architecture


This document describes the architecture of the implemented multimodal medical AI framework.

The framework is designed for chest X-ray classification using image and clinical text modalities.

The repository includes:

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

(Optional) Multimodal Fusion

↓

Binary Classification

(Normal / Abnormal)



---

# Vision Model


## Backbone


The image encoder uses a pretrained Squeeze-and-Excitation ResNet architecture.

Implementation:

SE-ResNet152d pretrained on ImageNet


Note:

The original methodology uses SE-ResNet154.
Due to pretrained checkpoint availability,
this implementation uses SE-ResNet152d as the available equivalent SE-ResNet backbone.



---

# Dual-View Image Processing


The model processes two chest X-ray views:

- Frontal view
- Lateral view


Each view is passed through an independent image encoder.



Architecture:



Frontal X-ray

↓

SE-ResNet152d

↓

2048-dimensional feature vector



Lateral X-ray

↓

SE-ResNet152d

↓

2048-dimensional feature vector



The two feature vectors are concatenated:


2048 + 2048

=

4096-dimensional image representation



The image representation is used for:

- Image-only classification
- Multimodal fusion experiments



---

# Image Classification Head


The image classifier receives the fused dual-view representation.


Pipeline:


Frontal Features

+

Lateral Features

↓

Feature Concatenation

↓

Fully Connected Layer

↓

Binary Classification

↓

Normal / Abnormal



---

# Text Model


## Backbone


BioClinicalBERT


The text encoder processes clinical reports associated with chest X-ray images.


Architecture:



Clinical Report

↓

Tokenizer

↓

BioClinicalBERT

↓

CLS Representation

↓

Classification Layer



The CLS token representation is used as the text feature representation.


Feature dimension:


768



---

# Multimodal Fusion


The repository implements multiple fusion strategies for combining image and text information.


These modules are implemented for future multimodal experiments.



---

# Early Fusion


Early fusion combines image and text representations before classification.


Feature representation:



Image Features

(4096 dimensions)


+

Text Features

(768 dimensions)


=

4864-dimensional multimodal representation



Pipeline:



Image Encoder

↓

Image Features


Text Encoder

↓

Text Features


↓

Feature Concatenation

↓

Classification Head

↓

Prediction



---

# Late Fusion


Late fusion combines predictions from independently trained modality models.


Training procedure:


Stage 1:

Train image and text models independently.


Stage 2:

Combine modality predictions using a fusion classifier.



Pipeline:



Image Model

↓

Image Prediction



Text Model

↓

Text Prediction



↓

Fusion Layer

↓

Final Prediction



---

# Ensemble Fusion


Ensemble fusion combines modality outputs directly.


Pipeline:



Image Logits

+

Text Logits

↓

Final Prediction



---

# Current Evaluated Experiment


The first completed experiment evaluates the image-only baseline.


Configuration:


Input:

- Frontal chest X-ray
- Lateral chest X-ray


Backbone:

- Dual-view SE-ResNet152d


Task:

Binary classification


Classes:

- Normal
- Abnormal


Dataset:

Indiana University Chest X-ray Dataset



---

# Adversarial Robustness Extension


The repository structure includes adversarial robustness modules.

Planned evaluations include:


## Image Attacks


Methods:

- FGSM
- PGD


Target:

Chest X-ray images



## Text Attacks


Methods:

- Synonym replacement
- Sentence deletion


Target:

Clinical text reports



---

# Evaluation Pipeline


Experiments evaluate:


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



---

# Implementation Status


Implemented:

- Dual-view chest X-ray processing
- SE-ResNet image encoder
- Binary image classification pipeline
- Dataset preparation
- Training and validation pipeline


Implemented but not yet fully evaluated:

- BioClinicalBERT text encoder
- Multimodal fusion strategies
- Adversarial robustness experiments
