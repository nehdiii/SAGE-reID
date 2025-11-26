<h1 align="center">
  <i>Low-Rank Expert Merging for Multi-Source Domain Adaptation in Person Re-Identification</i>
</h1>

<p align="center">
  <a href="https://www.linkedin.com/in/taha-mustapha-nehdi-240585203/" target='_blank'>Taha Mustapha Nehdi</a>,&nbsp;
  <a href="https://scholar.google.com/citations?user=pJm5B2YAAAAJ&hl=en" target='_blank'>Nairouz Mrabah</a>,&nbsp;
  <a href="https://www.linkedin.com/in/atif-belal-15779821a/" target='_blank'>Atif Belal</a>,&nbsp;
  <a href="https://www.linkedin.com/in/marco-pedersoli-50677321b/" target='_blank'>Marco Pedersoli</a>,&nbsp;
  <a href="https://www.linkedin.com/in/eric-granger-4062324/" target='_blank'>Eric Granger </a>,&nbsp;
  <br>
  École de technologie supérieure (ÉTS) <br>
  📧 Primary Contact: taha-mustapha.nehdi.1@ens.etsmtl.ca
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2508.06831" target='_blank'>
    <img alt="Static Badge" src="https://img.shields.io/badge/arXiv-2403.16848-b31b1b?style=flat-square">
  </a>
  <a href="">
    <img alt="Static Badge" src="https://img.shields.io/badge/WACV%202026-%F0%9F%8C%B5-%235E86C1?style=flat-square">
  </a>
</p>

## :mag: Overview

**TL; DR.** We propose SAGE-reID, a source-free multi-source domain adaptation framework for person re-identification based on gated LoRA experts. It first learns lightweight source-specific LoRA adapters without accessing source data during adaptation, then uses a small gating network to dynamically merge these experts while keeping the backbone fixed, achieving state-of-the-art accuracy with <2% extra parameters on Market-1501, DukeMTMC-reID, and MSMT17.

![Overview](./assets/wacv2026pap.png)


## :fire: News
- <span style="font-variant-numeric: tabular-nums;">**2025.10.06**</span>: Our paper is accepted by WACV 2026 :tada: :tada:. The revised paper and a more efficient codebase will be released in December. Almost there :nerd_face: ~

- <span style="font-variant-numeric: tabular-nums;">**2025.08.09**</span>: The first version of our paper is released at [arXiv:2508.06831v1](https://arxiv.org/abs/2508.06831)  :pushpin:.

## ⬇️ Installation
Our codebase is built upon **Python 3.12, PyTorch 2.5.0 (recommended)**. 

### Setup scripts

```shell
conda create -n SAGE-reID python=3.12		# suggest to use virtual envs
conda activate SAGE-reID
# PyTorch:
# CUDA 12.4
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
# Other dependencies:
pip install -t requirments.txt
```

# 📋 Data Preparation

:link: For all the datasets we used in our experiments, you can access them from the following public link:

- msmt17 [paper](https://arxiv.org/abs/1711.08565), [Link](https://www.pkuvmc.com/dataset.html)
- DukeMTMC-reID [paper](https://arxiv.org/abs/1609.01775), [Link](https://www.kaggle.com/datasets/whurobin/dukemtmcreid)
- Market1501 [paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Zheng_Scalable_Person_Re-Identification_ICCV_2015_paper.html), [Link](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?resourcekey=0-8nyl7K9_x37HlQm34MmrYQ)
- CUHK03 [paper](https://arxiv.org/abs/2101.10774), [Link](https://www.kaggle.com/datasets/priyanagda/cuhk03)

## File Tree

```text
./data
├── dukemtmc
│  └── DukeMTMC-reID
├── market1501
│  └── Market-1501-v15.09.15
├── msmt17
|   └── MSMT17_V2
├── cuhk03
|   └── images_detected
|   └── images_labeled
|   └── cuhk03_new_protocol_config_detected.mat
|   └── cuhk03_new_protocol_config_labeled.mat
    ...
```

## :dash: Quick Start

In this documentation, we will primarily focus on pre-training, adaptation, and Low-Rank Merging on msmt17 benchmarks as example feel free to try other benchmarks. All the configurations corresponding to our experiments are stored in the [scripts](../scripts/) folder. You can also customize the configuration files according to your own requirements.

### ImageNet Pre-trained Weights

:floppy_disk: ​Similar to many methods in the literature, we use the ViT-B/16 (vit_base_patch16_224) model initialized with ImageNet-1k pre-trained weights. Various ViT model variants are also available in [vit.py](../SAGE_reID/models/vit.py).


### Pre-train ViT-B/16 on Specific Source Datasets

we will first pre-train ViT-B/16 on the corresponding dataset (source dataset) to serve as the initialization for subsequent domain adaptation step.

For example, you can pre-train a ViT-B/16 model on market1501 as follows:

```bash
sh scripts/pre_train/pre_market1501.sh
```

### Source to Target Domain Adaptation 

In this step, we perform source-to-target domain adaptation using deep clustering in a source-free setting, relying only on initializing our model with source pre-trained weights.

For example, you can adapt from market1501 to msmt17 as follows:

```bash
sh scripts/adaptation/adapt_market1501_to_msmt17.sh
```

### Low-Rank Adapters Merging 

Once the low-rank experts from Market1501, DukeMTMC-reID, and CUHK03 have been adapted to MSMT17 and the adapted experts have been saved, we proceed in this step to merge them.

```bash
sh scripts/merging/adapt_market1501_cuhk03_dukemtmc_to_msmt17.sh
```
Using this script, you can achieve 44.1 mAP & 69.8 R1 on msmt17 test set. There is a relatively high instability (~ 0.8) 

## Model Zoo

### Pre-trained models 

| dataset |                          Resources                           |  mAP  |  R1  |
| :-----: | :----------------------------------------------------------: | :---: | :--: |
| Market1501 | :----------------------------------------------------------: | :---: | :--: |
| DukeMTMC-reID | :----------------------------------------------------------: | :---: | :--: |
| CUHK03 | :----------------------------------------------------------: | :---: | :--: |


### Adapted Modles

| dataset |                          Resources                           |  mAP  |  R1  |
| :-----: | :----------------------------------------------------------: | :---: | :--: |
| Market1501 to msmt17| :----------------------------------------------------------: | :---: | :--: |
| DukeMTMC-reID to msmt17 | :----------------------------------------------------------: | :---: | :--: |
| CUHK03 to msmt17 | :----------------------------------------------------------: | :---: | :--: |

### Low-Rank Merging 

| dataset |                          Resources                           |  mAP  |  R1  |
| :-----: | :----------------------------------------------------------: | :---: | :--: |
| Market1501 & DukeMTMC-reID & CUHK03  to msmt17| :----------------------------------------------------------: | :---: | :--: |




## :bouquet: Acknowledgements

This project is built upon [UDAStrongBaseline](https://github.com/zkcys001/UDAStrongBaseline), [LoRA](https://github.com/microsoft/LoRA) .

## :pencil2: Citation

If you think this project is helpful, please feel free to leave a :star: and cite our paper:

```tex
@article{nehdi2025low,
  title={Low-Rank Expert Merging for Multi-Source Domain Adaptation in Person Re-Identification},
  author={Nehdi, Taha Mustapha and Mrabah, Nairouz and Belal, Atif and Pedersoli, Marco and Granger, Eric},
  journal={arXiv preprint arXiv:2508.06831},
  year={2025}
}
```
