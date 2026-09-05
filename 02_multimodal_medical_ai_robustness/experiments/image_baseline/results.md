# Image Baseline Results


## Experiment


Model:

SE-ResNet152d Dual-View Image Classifier


Dataset:

Indiana University Chest X-ray Dataset


Input:

- Frontal view
- Lateral view


Training:

- Epochs: 13
- Batch size: 128
- Optimizer: Adam
- Learning rate: 1e-4


---

# Validation Results


| Metric | Value |
|---|---:|
| Accuracy | 0.7109 |
| F1-score | 0.7888 |
| Precision | 0.7231 |
| Recall | 0.8764 |
| ROC-AUC | 0.7618 |


---

# Observation


The model achieved strong training performance but showed a noticeable gap between training and validation performance.

This suggests overfitting during fine-tuning of the large pretrained backbone.

Future experiments will investigate:

- regularization methods
- data augmentation
- multimodal fusion
