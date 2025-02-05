# TAPB: An Interventional Debiasing Framework for Alleviating Target Prior Bias in Drug-Target Interaction Prediction

fully codes will be uploaded in several hours

This repository contains the PyTorch implementation of **TAPB** framework, as described in our paper
## Framework
![TAPB](image/TAPB.jpg)
## System Requirements

The source code was developed in Python 3.8 using PyTorch 1.7.1. The required python dependencies are given below. TAPB is supported for any standard computer and operating system (Windows/macOS/Linux) with enough RAM to run. There is no additional non-standard hardware requirements.

```
torch>=1.7.1
dgl>=0.7.1
dgllife>=0.2.8
numpy>=1.20.2
scikit-learn>=0.24.2
pandas>=1.2.4
prettytable>=2.2.1
rdkit~=2021.03.2
yacs~=0.1.8
comet-ml~=3.23.1 # optional
```

## Datasets

The `datasets` folder contains all experimental data used in TAPB: [BindingDB](https://github.com/peizhenbai/DrugBAN), [BioSNAP](https://github.com/kexinhuang12345/MolTrans)
