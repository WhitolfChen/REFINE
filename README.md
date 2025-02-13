# REFINE

This is the official implementation of our paper [REFINE: Inversion-Free Backdoor Defense via Model Reprogramming](https://openreview.net/pdf?id=4IYdCws9fc), accepted by ICLR 2025.

## Abstract

Backdoor attacks on deep neural networks (DNNs) have emerged as a significant security threat, allowing adversaries to implant hidden malicious behaviors during the model training phase. Pre-processing-based defense, which is one of the most important defense paradigms, typically focuses on input transformations or backdoor trigger inversion (BTI) to deactivate or eliminate embedded backdoor triggers during the inference process. However, these methods suffer from inherent limitations: transformation-based defenses often fail to balance model utility and defense performance, while BTI-based defenses struggle to accurately reconstruct trigger patterns without prior knowledge. In this paper, we propose REFINE, an inversion-free backdoor defense method based on model reprogramming. REFINE consists of two key components: **(1)** an input transformation module that disrupts both benign and backdoor patterns, generating new benign features; and **(2)** an output remapping module that redefines the model's output domain to guide the input transformations effectively. By further integrating supervised contrastive loss, REFINE enhances the defense capabilities while maintaining model utility. Extensive experiments on various benchmark datasets demonstrate the effectiveness of our REFINE and its resistance to potential adaptive attacks. 

## Getting Started

### Installation

#### 1. Install the Environment

Install the required dependencies using `conda`:

```
conda env create -f environment.yaml
conda activate refine
```

#### 2. Download the Dataset

Download the pre-processed dataset `CIFAR10.tar.gz` from [link](https://drive.google.com/file/d/1vZtzN3a33rKitnqz5ad9S8v5Rv6CK2-Q/view?usp=sharing), and unzip it into folder `data`.


### Usage

#### 1. Train Backdoored Models

To train a backdoored model on CIFAR10 for BadNets attack:
```
python attack.py --dataset CIFAR10 --attack BadNets
```
The results of attack can be found in folder `attack`.


#### 2. Utilize REFINE for Defense

To defend a backdoored model on CIFAR10 for BadNets attack:
```
python refine.py --dataset CIFAR10 --attack BadNets
```
The results of defense can be found in folder `refine_res`.

### Acknowledgments

Our code is built upon [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox). We also integrate REFINE into BackdoorBox for easy access and usage.

### Citation

If you find our work useful for your research, please consider citing our paper:

```
@inproceedings{chen2025refine,
  title={{REFINE}: Inversion-Free Backdoor Defense via Model Reprogramming},
  author={Chen, Yukun and Shao, Shuo and Huang, Enhao and Li, Yiming and Chen, Pin-Yu and Qin, Zhan and Ren, Kui},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
