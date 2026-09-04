# Dataset

This project uses the Brain Tumor MRI Dataset.

The dataset contains four classes:

- glioma
- meningioma
- pituitary
- no tumor


The original study used a composite dataset collected from:

- Figshare
- SARTAJ
- Br35H


Total images:
7023


## Download

Download the dataset from Kaggle:

(https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)


## Directory Structure

After downloading, organize the dataset as:


data/raw/

├── glioma/

├── meningioma/

├── pituitary/

└── no_tumor/


The training scripts automatically load images from this directory.
