 # MER2024-SEMI
<h2 align="center">
Leveraging Contrastive Learning and Self-Training for Multimodal Emotion Recognition with Limited Labeled Samples
</h2>

<p align="center">
  <!-- <img src="https://img.shields.io/badge/EMNLP-2023-brightgreen"> -->
  <!-- <under review><img src="http://img.shields.io/badge/Paper-PDF-red.svg"></a> -->
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white">
</p>

### Overview
<div style="text-align: center">
<img src="resource/framework1.jpg" width = "100%"/>
</div>


### Key Implementations

- Noisy embedding construction `models/Noise_scheduler.py Line 74`;
- Contrastive learning between modalities `models/Contrastive_model.py Line 182`;
- Contrastive learning between original and noisy features `models/Contrastive_model.py Line 190`;
- Calculate contrastive loss `models/Contrastive_model.py Line 202`;

### More Analysis

#### Q1: TBD? 

[Table](./figures/table.md) presents xxx.

#### Q2: TBD


### Installations

Create a conda environment with Pytorch

```
conda create --name contrastive python=3.9
conda activate contrastive

pip install torch torchvision torchaudio numpy pandas sklearn scipy tqdm pickle omegaconf
```

This repository is constructed and gives the main modules used in our work, which are based on the codebase from [MER2024](https://github.com/zeroQiaoba/MERTools/tree/master/MER2024). 

You can get more information about the training framework or the competition from the link above.

Other requirements can also refer to the MER2024 GitHub repository.



### Datasets and Models
#### MER2024 Dataset

Please download the End User License Agreement, fill it out, and send it to merchallenge.contact@gmail.com to access the data. The EULA file can be found at [MER2024](https://github.com/zeroQiaoba/MERTools/tree/master/MER2024). 


   *Notes: It appears that the original dataset [website](https://www.fbk.eu/en/research-centers/) hides the download link. However, the dataset can still be downloaded after filling out the dataset request [form](https://docs.google.com/forms/d/e/1FAIpQLSer9jNfUtxbi610n3T6diXRlANBbuzShsCje-GtKs1Sngh0YQ/viewform?pli=1) directly. So we recommend that you use this method.*

1. Make directories to store ST (MuST-C) and datasets. Please specify the target language.

2.  Preprocess spm data. 

#### Speech Pre-trained Model 

We use HuBERT model for speech pre-trained model for training. Before training, please download the [HuBERT-Base](https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt) model.

