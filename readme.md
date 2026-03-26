# DCHGCN: Dynamic Causal Heterogeneous Graph Convolutional Network
## Overview
DCHGCN is a novel model for drug-disease association prediction, which integrates causal inference and heterogeneous graph neural networks to eliminate confounding bias, distinguish causal relationships from spurious correlations, and improve the generalization and interpretability of prediction.

## Features
- Dynamic causal graph learning with multi-source biological evidence
- Causal intervention and bias correction modules
- Heterogeneous graph convolution for drug-disease association prediction


## Requirements
```
torch
torch_geometric
numpy
pandas
scikit-learn
scipy
matplotlib
tqdm
```

pip install torch==1.13+ transformers==4.30.0 pandas==2.0.0 scikit-learn==1.2.2
pip install torch_geometric==2.3.1
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install biopython==1.81 requests==2.31.0 tqdm==4.65.0 openpyxl




## Citation
If you use this code, please cite our work.
```