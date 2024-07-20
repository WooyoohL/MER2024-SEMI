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

#### Q1: Why use token pruning rather than token merging? 

[Table](./figures/table.md) presents the specific numerical values for the visualization of Fig. 3 in the paper. It can be observed that the attention scores between many tokens in the table are 0, indicating that there is no mutual influence between some tokens. Furthermore, merging tokens consumes additional time; therefore, our proposed method uses token pruning rather than token merging.



#### Q2: Why not pruning based on a threshold?
If a threshold is used, it will result in different numbers of tokens being pruned for each input sequence, rendering the model unable to batch process. If we employ a MASK matrix to mask the pruning tokens, however, it contradicts the original intention of model acceleration.

#### Q3: Why set a pruning rate instead of a fixed pruning number?
The differences in patch length for various image inputs are not significant. For example, most images fed into pre-training models have both H and W dimensions set to 224, corresponding to a patch length of 588 (3 × 224 × 224 // 16 ×16). Therefore, a fixed number of tokens can be gradually clipped in the field of CV.

However, the token length for different speech inputs varies significantly. Additionally, the length of speech sequences is much longer than that of corresponding text sequences. Therefore, we use a pruning rate to ensure that longer speech inputs prune more tokens, and shorter ones prune fewer tokens.
