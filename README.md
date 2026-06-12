CPREBV-MDD-Severity
This repository provides the implementation of CPREBV: ConvNeXt-powered Procrustes-Rotated Equiangular Basis Vectors for multicenter diagnosis and severity stratification of major depressive disorder (MDD) using multi-tissue structural MRI.
Overview
CPREBV is an end-to-end deep learning framework designed for two related tasks:
MDD versus healthy control classification
Intra-MDD severity stratification, including mild, moderate, and severe MDD
The model uses voxel-based morphometry (VBM)-derived structural MRI tissue maps, including:
Gray matter (GM)
White matter (WM)
Cerebrospinal fluid (CSF)
The framework mainly contains three components:
PRF: Procrustes-based Orthogonal Alignment Fusion for cross-tissue feature alignment
D-AG: BioBERT-assisted Demographics-Aware Gating for incorporating sex, age, and education information
RMP-EBV: Residual Multi-Prototype Equiangular Basis Vector head for geometry-aware classification
File Description
CPREBV.py
This file contains the main implementation of the CPREBV framework, including the 3D ConvNeXt backbone, Procrustes-based cross-tissue fusion, demographics-aware gating module, and EBV-based classification head.
Requirements
The code is implemented in Python and PyTorch. The main dependencies include:
python >= 3.8
torch
torchvision
numpy
pandas
scikit-learn
scipy
transformers
tqdm
You can install the required packages using:
pip install torch torchvision numpy pandas scikit-learn scipy transformers tqdm
Data Availability
The REST-meta-MDD dataset is not included in this repository due to privacy restrictions and data-use agreements. Researchers should apply for access to the dataset from the official data provider and organize the data according to the required input format.
Please do not upload raw MRI data, subject-level demographic files, labels, or trained model checkpoints to this public repository.
