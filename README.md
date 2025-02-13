# REFINE

This is the official implementation of our paper [REFINE: Inversion-Free Backdoor Defense via Model Reprogramming](https://openreview.net/pdf?id=4IYdCws9fc), accepted by ICLR 2025.

# Getting Started

## Installation

Install the required dependencies using `conda`:

```
conda env create -f environment.yaml
conda activate refine
```

## Usage

### 1. Training Backdoored Models

To train a backdoored model on CIFAR10 for BadNets attack:
```
python attack.py --dataset CIFAR10 --attack BadNets
```

### 2. Using REFINE for defense

To defence a backdoored model on CIFAR10 for BadNets attack:
```
python refine.py --dataset CIFAR10 --attack BadNets
```

## Citation

If you find our work useful for your research, please consider citing our paper:

```
@inproceedings{chen2025refine,
  title={{REFINE}: Inversion-Free Backdoor Defense via Model Reprogramming},
  author={Chen, Yukun and Shao, Shuo and Huang, Enhao and Li, Yiming and Chen, Pin-Yu and Qin, Zhan and Ren, Kui},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
