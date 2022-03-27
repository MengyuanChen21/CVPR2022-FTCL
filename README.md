# Fine-grained Temporal Contrastive Learning for Weakly-supervised Temporal Action Localization
[Paper]()

Junyu Gao, Mengyuan Chen, Changsheng Xu

IEEE/CVF Conference on Computer Vision and Pattern Recognition (**CVPR**), 2022.


## Table of Contents
1. [Introduction](#introduction)
1. [Installation](#installation)
1. [Datasets](#datasets)
1. [Testing](#testing)
1. [Training](#training)
1. [Pre-trained Model](#pre-trained model)
1. [Citation](#citation)

## Introduction
We target at the task of weakly-supervised action localization (WSAL), where only video-level action labels are available during model training. Despite the recent progress, existing methods mainly embrace a localization-by-classification paradigm and overlook the fruitful fine-grained temporal distinctions between video sequences, thus suffering from severe ambiguity in classification learning and classification-to-localization adaption. This paper argues that learning by contextually comparing sequence-to-sequence distinctions offers an essential inductive bias in WSAL and helps identify coherent action instances. Specifically, under a differentiable dynamic programming formulation, two complementary contrastive objectives are designed, including Fine-grained Sequence Distance (FSD) contrasting and Longest Common Subsequence (LCS) contrasting, where the first one considers the relations of various action/background proposals by using match, insert, and delete operators and the second one mines the longest common subsequences between two videos. Both contrasting modules can enhance each other and jointly enjoy the merits of discriminative action-background separation and alleviated task gap between classification and localization. Extensive experiments show that our method achieves state-of-the-art performance on three popular benchmarks.

<!-- <div align="center">
  <img src="figs/arch.png" width="800px"/><br>
    Overview of the FTCL
</div> -->

## Preparations
### Requirements and Dependencies
Here we only list our used requirements and dependencies. It would be great if you can work around with the latest versions of the listed softwares and hardwares.
 - Linux: Ubuntu 20.04 LTS
 - GPU: GeForce RTX 3090
 - CUDA: 11.4
 - GCC: 9.4.0
 - Python: 3.7.11
 - Anaconda: 4.10.1
 - PyTorch: 1.9.0

### Installation

Required packages are listed in [requirements.txt](/requirements.txt). You can install by running:

```
pip install -r requirements.txt
```

## Datasets

#### THUMOS14 Dataset：

We use the 2048-d features provided by arXiv 2021 paper: ACM-Net Action Context Modeling Network for Weakly-Supervised Temporal Action Localization. You can get access of the THUMOS14 dataset from [Google Drive](https://drive.google.com/drive/folders/1C4YG01X9IIT1a568wMM8fgm4k4xTC2EQ?usp=sharing) /  [Baidu Wangpan](https://pan.baidu.com/s/1rt8szoDspzJ5SjpcjccFXg) (pwd: vc21).

Before running the code, please download the target dataset and unzip it under the `data/` folder.

## Testing

To test your model, you can run following command:

```bash
# For the THUMOS-14 dataset.
python main_thu.py --test --checkpoint $checkpoint_path
```

## Training

In order to reduce the training time, we utilize a pre-trained model produced by the backbone network ACM-Net to initialize our FTCL-Net, which can be accessed [here](). For how to produce the initialization network, please see [ACM-Net](https://github.com/ispc-lab/ACM-Net).

You can train your own model by running:

```bash
# For the THUMOS-14 dataset.
python main_thu.py --batch_size 16 --checkpoint $initialization_network_path
```

You can configure your own hyper-parameters in `config/model_config.py` .

Note that we apply the [`wandb`](https://github.com/wandb/client) client to log the experiments, if you don't want to use this tool, you can disable it in the command with   `--without_wandb` like 

```bash
python main_thu.py --without_wandb
```

## Pre-trained Model

The pre-trained model (checkpoint) of our FTCL-Net for the THUMOS-14 dataset can be downloaded from the [Google Drive]().

## Citation
If you find the code useful in your research, please cite:

    @inproceedings{junyu2022CVPR_FTCL,
      author = "Gao, Junyu and Chen, Mengyuan and Xu, Changsheng",
      title = "Fine-grained Temporal Contrastive Learning for Weakly-supervised Temporal Action Localization",
      booktitle = "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
      year = "2022"
    }

## License

See [MIT License](/LICENSE)

## Acknowledgement

This repo contains modified codes from:
 - [ACM-Net](https://github.com/ispc-lab/ACM-Net): for implementation of the backbone [ACM-Net (arXiv-2021)](https://arxiv.org/abs/2104.02967).
 - [VideoAlignment](https://github.com/hadjisma/VideoAlignment): for implementation of the smooth max operation.

We sincerely thank the owners of all these great repos!