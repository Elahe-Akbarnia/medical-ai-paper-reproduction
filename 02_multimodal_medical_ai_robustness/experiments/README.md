# Dual-View Chest X-ray Image Baseline


## Description

This experiment reproduces the image classification component
using frontal and lateral chest X-ray views.


## Model

Backbone:

SE-ResNet152d pretrained on ImageNet

Input:

- Frontal chest X-ray
- Lateral chest X-ray


Architecture:

Frontal image
    |
SE-ResNet backbone
    |
2048 features


Lateral image
    |
SE-ResNet backbone
    |
2048 features


Feature concatenation

4096-dimensional representation

Binary classifier


## Dataset

Indiana University Chest X-ray Dataset


Training samples:

2710


Validation samples:

678


## Training Configuration

Epochs:

13


Batch size:

128


Optimizer:

Adam


Learning rate:

1e-4


## Notes

This experiment represents the baseline reproduction.
Further experiments will investigate multimodal fusion
and robustness improvements.
