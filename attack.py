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
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import torchvision.models as models
import core
import argparse

parser = argparse.ArgumentParser(description='PyTorch Attack')
parser.add_argument('--gpu', default='0', type=str, choices=[str(x) for x in range(8)])
parser.add_argument('--dataset', default='CIFAR10', type=str, choices=['CIFAR10', 'ImageNet50']) 
parser.add_argument('--model', default='ResNet18', type=str, choices=['ResNet18', 'ResNet50', 'VGG16', 'InceptionV3', 'DenseNet121', 'ViT']) 
parser.add_argument('--attack', default='BadNets', type=str, choices=['Benign', 'BadNets', 'Blended', 'WaNet', 'BATT', 'Physical', 'LC','Adaptive']) 

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
 
# ========== Set global settings ==========
global_seed = 666
deterministic = True
torch.manual_seed(global_seed)
CUDA_VISIBLE_DEVICES = args.gpu
dataset=args.dataset
model=args.model
attack=args.attack
datasets_root_dir = f'./data/{dataset}'
save_path = f'./attack/{dataset}/{model}/{attack}'

if dataset == 'CIFAR10':
    img_size = 32
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    num_classes = 10
elif dataset == 'ImageNet50':
    img_size = 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_classes = 50
else:
    raise NotImplementedError

input_size = img_size

if model == 'ResNet18':
    my_model = core.models.ResNet(18, num_classes=num_classes)
    if dataset == 'ImageNet50':
        my_model = models.resnet18(weights=None, num_classes=num_classes)
    lr = 0.1
elif model == 'ResNet50':
    my_model = core.models.ResNet(50, num_classes=num_classes)
    if dataset == 'ImageNet50':
        my_model = models.resnet50(pretrained=False, num_classes=num_classes)
    lr = 0.1
elif model == 'VGG16':
    deterministic = False
    my_model = core.models.vgg16(num_classes=num_classes)
    if dataset == 'ImageNet50':
        my_model = models.vgg16(pretrained=False, num_classes=num_classes)
    lr = 0.01
elif model == 'InceptionV3':
    my_model = models.inception_v3(pretrained=False, num_classes=num_classes, aux_logits = False)   # 299*299
    if dataset == 'CIFAR10':
        input_size = 96
    lr = 0.1
elif model == 'DenseNet121':
    my_model = models.densenet121(pretrained=False, num_classes=num_classes)    # 224*224
    lr = 0.1
elif model == 'ViT':
    my_model = core.models.ViT(
        image_size = img_size,
        patch_size = int(img_size / 8),
        num_classes = num_classes,
        dim = int(512),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    lr = 1e-3
else:
    raise NotImplementedError

my_model = my_model.to('cuda' if torch.cuda.is_available() else 'cpu')
 
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(input_size),
    transforms.Normalize(mean, std),
])
 
trainset = DatasetFolder(root=os.path.join(datasets_root_dir, 'train'),
                         transform=transform_train,
                         loader=cv2.imread,
                         extensions=('png','jpeg',),
                         target_transform=None,
                         is_valid_file=None,
                         )
 
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(input_size),
    transforms.Normalize(mean, std),
])
 
testset = DatasetFolder(root=os.path.join(datasets_root_dir, 'test'),
                         transform=transform_test,
                         loader=cv2.imread,
                         extensions=('png','jpeg',),
                         target_transform=None,
                         is_valid_file=None,
                         )

if attack in ['BadNets', 'Physical', 'LC', 'Benign']:
    pattern = torch.load(f'./attack/triggers/{dataset}_pattern.pth')
    weight = torch.zeros((img_size, img_size), dtype=torch.float32)
    if dataset == 'CIFAR10':
        weight[-3:, -3:] = 1.0
    elif dataset == 'ImageNet50':
        weight[-20:, -20:] = 1.0

if attack == 'BadNets' or attack == 'Benign':
    attacker = core.BadNets(
        train_dataset=trainset,
        test_dataset=testset,
        model=my_model,
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=0.1,
        pattern=pattern,
        weight=weight,
        seed=global_seed,
        deterministic=deterministic,
    )

elif attack == 'BATT':
    attacker = core.BATT(
        train_dataset=trainset,
        test_dataset=testset,
        model=my_model,
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=0.1,
        seed=global_seed,
        deterministic=deterministic,
    )

elif attack == 'Blended':
    pattern = cv2.imread('./attack/triggers/kitty.png')
    tf = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Resize(img_size)
    ])
    pattern = tf(pattern)
    weight = torch.full_like(pattern, 0.1, dtype=torch.float32)

    attacker = core.Blended(
        train_dataset=trainset,
        test_dataset=testset,
        model=my_model,
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=0.1,
        pattern=pattern,
        weight=weight,
        seed=global_seed,
        deterministic=deterministic,
        poisoned_transform_train_index=1,
        poisoned_transform_test_index=1,
    )

elif attack == 'WaNet':
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

    identity_grid, noise_grid = gen_grid(img_size, img_size) #old: 128

    attacker = core.WaNet(
        train_dataset=trainset,
        test_dataset=testset,
        model=my_model,
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=0.1,
        identity_grid=identity_grid,
        noise_grid=noise_grid,
        noise=False,
        seed=global_seed,
        deterministic=deterministic
    )

elif attack == 'Physical':
    if dataset == 'CIFAR10':
        tmean = [-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010]
        tstd = [1/0.2023, 1/0.1994, 1/0.2010]
    elif dataset == 'ImageNet50':
        tmean = [-0.485/0.229, -0.456/0.224, -0.406/0.225]
        tstd = [1/0.229, 1/0.224, 1/0.225]
    else:
        raise NotImplementedError 
    
    attacker = core.PhysicalBA(
        train_dataset=trainset,
        test_dataset=testset,
        model=my_model,
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=0.1,
        pattern=pattern,
        weight=weight,
        seed=global_seed,
        deterministic=deterministic,
        physical_transformations=transforms.Compose([
            transforms.Normalize(tmean, tstd),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 0.9)),
            transforms.Normalize(mean, std)
        ])
    )

elif attack == 'LC':
    adv_model = my_model
    adv_dir = os.path.join(os.path.dirname(save_path), 'Benign')
    for item in os.listdir(adv_dir):
        item_path = os.path.join(adv_dir, item)
        if os.path.isdir(item_path) and item.startswith("Normalize"):
            adv_path = item_path
            break
    adv_path = os.path.join(adv_path, 'ckpt_epoch_150.pth')
    print(adv_path)
    adv_model.load_state_dict(torch.load(adv_path))

    eps = 8
    alpha = 1.5
    steps = 100
    max_pixel = 255
    if dataset == 'CIFAR10':
        poisoned_rate = 0.25
    elif dataset == 'ImageNet50':
        poisoned_rate = 1.0

    print(trainset[0][0].shape)

    schedule = {
        'device': 'GPU',
        'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
        'GPU_num': 1,
        'batch_size': 128,
        'num_workers': 8,
    }

    attacker = core.LabelConsistent(
        train_dataset=trainset,
        test_dataset=testset,
        model=my_model,
        adv_model=adv_model,
        adv_dataset_dir=os.path.join(save_path, f'adv_dataset/{dataset}_eps{eps}_alpha{alpha}_steps{steps}_poisoned_rate{poisoned_rate}_seed{global_seed}'),
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=poisoned_rate,
        pattern=pattern,
        weight=weight,
        eps=eps,
        alpha=alpha,
        steps=steps,
        max_pixel=max_pixel,
        schedule=schedule,
        seed=global_seed,
        deterministic=deterministic,
    )

elif attack == 'Adaptive':
    trigger_dir = './attack/triggers/adaptive_triggers'
    trigger_names = [
        f'phoenix_corner_{img_size}.png',
        f'badnet_patch4_{img_size}.png',
        f'firefox_corner_{img_size}.png',
        f'trojan_square_{img_size}.png',
    ]
    trigger_path = [os.path.join(trigger_dir, name) for name in trigger_names]

    patterns = []
    for path in trigger_path:
        pattern = cv2.imread(path)
        pattern = transforms.ToTensor()(pattern)
        patterns.append(pattern)

    alphas = [0.5, 0.5, 0.2, 0.3]
    if model == 'ResNet18':
        poisoned_rate = 0.01
        covered_rate = 0.02
    elif model in ['VGG16', 'DenseNet121', 'ViT']:
        poisoned_rate = 0.03
        covered_rate = 0.06

    attacker = core.AdaptivePatch(
        train_dataset=trainset,
        test_dataset=testset,
        model=my_model,
        loss=nn.CrossEntropyLoss(),
        y_target=0,
        poisoned_rate=poisoned_rate,
        covered_rate=covered_rate,
        patterns=patterns,
        alphas=alphas,
        seed=global_seed,
        deterministic=deterministic,
    )

else:
    raise NotImplementedError

benign_training = False
if attack == 'Benign':
    benign_training = True

# Train Attacked Model
schedule = {
    'device': 'GPU',
    'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES,
    'GPU_num': 1,
 
    'benign_training': benign_training,
    'batch_size': 128,
    'num_workers': 8,
 
    'lr': lr,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gamma': 0.1,
    'schedule': [100, 130],
 
    'epochs': 150,
 
    'log_iteration_interval': 100,
    'test_epoch_interval': 10,
    'save_epoch_interval': 20,
 
    'save_dir': save_path,
    'experiment_name': f'Normalize_{model}_{dataset}_{attack}'
}
attacker.train(schedule)
