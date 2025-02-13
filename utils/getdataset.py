import core
import sys
import os
sys.path.append(os.getcwd())
import cv2
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, dataloader
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import DatasetFolder
import torchvision.models as models
import argparse
from PIL import Image
import copy

class GetDataset():

    def __init__(self, dataset, model, attack, tlabel):
        self.dataset = dataset
        self.model =model
        self.attack = attack
        self.tlabel = tlabel
        self.global_seed=666
        self.deterministic=True
        self.box_path = './attack'
        self.datasets_root_dir = f'./data/{self.dataset}'
        self.exp_path = os.path.join(self.box_path, self.dataset, self.model, self.attack)
        self.init_dataset()

    def init_dataset(self):

        if self.dataset == 'CIFAR10':
            self.img_size = 32
            self.mean = [0.4914, 0.4822, 0.4465]
            self.std = [0.2023, 0.1994, 0.2010]
        elif self.dataset == 'ImageNet50':
            self.img_size = 224
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            raise NotImplementedError

        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=self.img_size, scale=(0.8, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(p=0.2),
            transforms.Resize(self.img_size),
            transforms.Normalize(self.mean, self.std),
        ])
        
        self.trainset = DatasetFolder(root=os.path.join(self.datasets_root_dir, 'train'),
                                    transform=transform_train,
                                    loader=cv2.imread,
                                    extensions=('png','jpeg',),
                                    target_transform=None,
                                    is_valid_file=None,
                                    )
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size),
            transforms.Normalize(self.mean, self.std),
        ])
        
        self.testset = DatasetFolder(root=os.path.join(self.datasets_root_dir, 'test'),
                                    transform=transform_test,
                                    loader=cv2.imread,
                                    extensions=('png','jpeg',),
                                    target_transform=None,
                                    is_valid_file=None,
                                    )
        
        self.ptestset = DatasetFolder(root=os.path.join(self.datasets_root_dir, f'test_remove_{self.tlabel}'),
                                    transform=transform_test,
                                    loader=cv2.imread,
                                    extensions=('png','jpeg',),
                                    target_transform=None,
                                    is_valid_file=None,
                                    )

    def get_benign_dataset(self):
        return self.trainset, self.testset

    def get_poisoned_dataset(self):

        if self.attack in ['BadNets', 'Physical', 'LC', 'Benign']:
            pattern = torch.load(os.path.join(self.box_path, f'triggers/{self.dataset}_pattern.pth'))
            weight = torch.zeros((self.img_size, self.img_size), dtype=torch.float32)
            if self.dataset == 'CIFAR10':
                weight[-3:, -3:] = 1.0
            elif self.dataset == 'ImageNet50':
                weight[-20:, -20:] = 1.0

        if self.attack == 'BadNets':
            attacker = core.BadNets(
                train_dataset=self.trainset,
                test_dataset=self.ptestset,
                model=core.models.ResNet(18),
                loss=nn.CrossEntropyLoss(),
                y_target=0,
                poisoned_rate=0.1,
                pattern=pattern,
                weight=weight,
                seed=self.global_seed,
                deterministic=self.deterministic
            )
        
        elif self.attack == 'BATT':
            attacker = core.BATT(
                train_dataset=self.trainset,
                test_dataset=self.ptestset,
                model=core.models.ResNet(18),
                loss=nn.CrossEntropyLoss(),
                y_target=0,
                poisoned_rate=0.1,
                seed=self.global_seed,
                deterministic=self.deterministic
            )

        elif self.attack == 'Blended':
            pattern = cv2.imread(os.path.join(self.box_path, 'triggers/kitty.png'))
            tf = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Resize(self.img_size)
            ])
            pattern = tf(pattern)
            weight = torch.full_like(pattern, 0.1, dtype=torch.float32)

            attacker = core.Blended(
                train_dataset=self.trainset,
                test_dataset=self.ptestset,
                model=core.models.ResNet(18),
                loss=nn.CrossEntropyLoss(),
                y_target=0,
                poisoned_rate=0.1,
                pattern=pattern,
                weight=weight,
                seed=self.global_seed,
                deterministic=self.deterministic,
                poisoned_transform_train_index=1,
                poisoned_transform_test_index=1
            )

        elif self.attack == 'WaNet':
            def gen_grid(height, k):
                """Generate an identity grid with shape 1*height*height*2 and a noise grid with shape 1*height*height*2
                according to the input height ``height`` and the uniform grid size ``k``.
                """
                ins = torch.rand(1, 2, k, k) * 2 - 1
                ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
                noise_grid = nn.functional.upsample(ins, size=height, mode="bicubic", align_corners=True)
                noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
                array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
                x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
                identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2
            
                return identity_grid, noise_grid

            identity_grid, noise_grid = gen_grid(self.img_size, self.img_size)

            attacker = core.WaNet(
                train_dataset=self.trainset,
                test_dataset=self.ptestset,
                model=core.models.ResNet(18),
                loss=nn.CrossEntropyLoss(),
                y_target=0,
                poisoned_rate=0.1,
                identity_grid=identity_grid,
                noise_grid=noise_grid,
                noise=False,
                seed=self.global_seed,
                deterministic=self.deterministic
            )

        elif self.attack == 'Physical':
            if self.dataset == 'CIFAR10':
                tmean = [-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010]
                tstd = [1/0.2023, 1/0.1994, 1/0.2010]
            elif self.dataset == 'ImageNet50':
                tmean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
                tstd = [1/0.229, 1/0.224, 1/0.225]
            else:
                raise NotImplementedError 

            attacker = core.PhysicalBA(
                train_dataset=self.trainset,
                test_dataset=self.ptestset,
                model=core.models.ResNet(18),
                loss=nn.CrossEntropyLoss(),
                y_target=0,
                poisoned_rate=0.1,
                pattern=pattern,
                weight=weight,
                seed=self.global_seed,
                deterministic=self.deterministic,
                physical_transformations=transforms.Compose([
                    transforms.Normalize(tmean, tstd),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.1),
                    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 0.9)),
                    transforms.Normalize(self.mean, self.std)
                ])
            )

        elif self.attack == 'LC':
            eps = 8
            alpha = 1.5
            steps = 100
            max_pixel = 255
            if self.dataset == 'CIFAR10':
                poisoned_rate = 0.25
            elif self.dataset == 'ImageNet50':
                poisoned_rate = 1.0

            # schedule = {
            #     'device': 'GPU',
            #     'CUDA_VISIBLE_DEVICES': '0',
            #     'GPU_num': 1,
            #     'batch_size': 128,
            #     'num_workers': 8,
            # }

            adv_dataset_dir = os.path.join(self.exp_path, f'adv_dataset/{self.dataset}_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{self.global_seed}')
            attacker = core.LabelConsistent(
                train_dataset=self.trainset,
                test_dataset=self.ptestset,
                model=core.models.ResNet(18),
                adv_model=None,
                adv_dataset_dir=adv_dataset_dir,
                loss=nn.CrossEntropyLoss(),
                y_target=0,
                poisoned_rate=poisoned_rate,
                pattern=pattern,
                weight=weight,
                eps=eps,
                alpha=alpha,
                steps=steps,
                max_pixel=max_pixel,
                # schedule=schedule,
                seed=self.global_seed,
                deterministic=self.deterministic
            )
            
        elif self.attack == 'Adaptive':
            trigger_dir = os.path.join(self.box_path, 'triggers/adaptive_triggers')
            trigger_names = [
                f'phoenix_corner_{self.img_size}.png',
                f'badnet_patch4_{self.img_size}.png',
                f'firefox_corner_{self.img_size}.png',
                f'trojan_square_{self.img_size}.png',
            ]
            trigger_path = [os.path.join(trigger_dir, name) for name in trigger_names]

            patterns = []
            for path in trigger_path:
                pattern = cv2.imread(path)
                pattern = transforms.ToTensor()(pattern)
                patterns.append(pattern)

            alphas = [0.5, 0.5, 0.2, 0.3]

            attacker = core.AdaptivePatch(
                train_dataset=self.trainset,
                test_dataset=self.ptestset,
                model=core.models.ResNet(18),
                loss=nn.CrossEntropyLoss(),
                y_target=0,
                poisoned_rate=0.01,
                covered_rate=0.02,
                patterns=patterns,
                alphas=alphas,
                seed=self.global_seed,
                deterministic=self.deterministic,
            )

        poisoned_trainset, poisoned_testset = attacker.get_poisoned_dataset()
        return poisoned_trainset, poisoned_testset
    