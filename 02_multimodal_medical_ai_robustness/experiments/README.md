# Experiments


This directory organizes experiment configurations and generated experiment records.


The implemented experiments correspond to the main components described in the paper.


---

# Experiment Groups


## 1. Vision Model


Single-modality image classification using:

- SE-ResNet-154
- Chest X-ray images


Configuration:
configs/image_model.yaml


---

## 2. Text Model


Single-modality text classification using:

- Bio_ClinicalBERT
- Clinical reports


Configuration:
configs/text_model.yaml


---

## 3. Early Fusion


Feature-level multimodal fusion.


Configuration:
configs/early_fusion.yaml


---

## 4. Late Fusion


Decision-level multimodal fusion.


Configuration:
configs/late_fusion.yaml


---

## 5. Adversarial Robustness Evaluation


Experiments evaluating model performance under:


Image attacks:

- FGSM
- PGD


Text attacks:

- Synonym replacement
- Half-sentence deletion


Configuration:
configs/attacks.yaml


---

# Experiment Outputs


Generated outputs are stored in:

```
results/

├── checkpoints/

├── logs/

├── figures/

├── tables/

└── predictions/
```

No experimental results are stored in this repository.



