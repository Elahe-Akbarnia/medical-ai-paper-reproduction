# Reproduction Notes


This document describes reproduction details, implementation assumptions,
and differences between the original paper and this implementation.


---

# Original Paper


Title:

"Assessing the adversarial robustness of multimodal medical AI systems:
insights into vulnerabilities and modality interactions"


The original study investigates adversarial robustness of multimodal medical AI systems using:

- Chest X-ray images
- Clinical text reports


---

# Dataset


The original paper uses:

Indiana University Chest X-ray Dataset


The dataset is not included in this repository.

Users must download the dataset separately and prepare it according to:
data/README.md



---

# Dataset Split


The paper reports:


80% training

20% testing



The exact split indices used in the original experiments are not publicly specified.


This implementation follows the reported split ratio.


---

# Random Seed


The original paper does not specify the random seed used during experiments.


This implementation uses:
seed = 42


for reproducibility.


---

# Image Preprocessing


The paper reports normalization using:


Mean:
0.61


Standard deviation:
0.24


The same values are used in this implementation.


---

# Text Processing


The paper specifies:

- lowercase conversion
- whitespace cleaning


Additional tokenizer settings may depend on the Bio_ClinicalBERT implementation.


---

# Model Initialization


The implementation uses pretrained:

- SE-ResNet-154
- Bio_ClinicalBERT


The exact pretrained checkpoint versions may affect final results.


---

# PGD Configuration


The paper specifies:

- epsilon values
- number of attack iterations


However, the exact PGD step size is not reported.


This implementation uses:
alpha = 0.01


---

# Experimental Results


This repository does not include generated experimental results.


The following directories are reserved for outputs:
```
results/

├── logs/

├── checkpoints/

├── figures/

├── tables/

└── predictions/
```


Results are generated after running the experiment scripts.


---

# Reproducibility Goal

The objective of this repository is to provide:

- complete implementation
- reproducible pipeline
- modular experiments
- extensible adversarial evaluation framework
