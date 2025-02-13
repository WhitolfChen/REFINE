import torch
import os.path as osp
import sys
import os
sys.path.append(os.getcwd())
import core
from torchvision import models
import copy

class GetModel():
    def __init__(self, dataset, model, attack):
        self.dataset, self.model, self.attack = dataset, model, attack
        self.box_path = './attack'
        self.init_model()

    def init_model(self):
        if self.dataset == 'CIFAR10':
            self.num_classes = 10
        elif self.dataset == 'ImageNet50':
            self.num_classes = 50
        else:
            raise NotImplementedError

        if self.model == 'ResNet18':
            if self.dataset == 'CIFAR10':
                self.my_model = core.models.ResNet(18, self.num_classes)
            elif self.dataset == 'ImageNet50':
                self.my_model = models.resnet18(weights=None, num_classes=self.num_classes)
        elif self.model == 'ResNet50':
            if self.dataset == 'CIFAR10':
                self.my_model = core.models.ResNet(50, self.num_classes)
            elif self.dataset == 'ImageNet50':
                self.my_model = models.resnet50(weights=None, num_classes=self.num_classes)
        elif self.model == 'DenseNet121':
            self.my_model = models.densenet121(weights=None, num_classes=self.num_classes)
        elif self.model == 'VGG16':
            self.my_model = core.models.vgg16(num_classes=self.num_classes)
        elif self.model == 'ViT':
            self.my_model = core.models.ViT(image_size = 32,
                                            patch_size = 4,
                                            num_classes = 10,
                                            dim = int(512),
                                            depth = 6,
                                            heads = 8,
                                            mlp_dim = 512,
                                            dropout = 0.1,
                                            emb_dropout = 0.1
                                        )
        else:
            raise NotImplementedError

        exp_dir = osp.join(self.box_path, self.dataset, self.model, self.attack)
        for item in os.listdir(exp_dir):
            item_path = os.path.join(exp_dir, item)
            if os.path.isdir(item_path) and item.startswith("Normalize"):
                model_dir = item_path
                break
        model_path = os.path.join(model_dir, 'ckpt_epoch_150.pth')
        print('model_path: ', model_path)
        self.my_model.load_state_dict(torch.load(model_path))
        self.my_model.eval()

    def get_model(self):
        return self.my_model
    
    def get_feature_model(self):
        feature_model = copy.deepcopy(self.my_model)

        if self.model == 'DenseNet121':
            feature_model = core.models.DenseNetWithFeatures(self.num_classes)
        elif self.model in ['ResNet18', 'ResNet50'] and self.dataset == 'ImageNet50':
            feature_model = core.models.ResNetWithFeatures(self.model, self.num_classes)
        
        feature_model.set_model(copy.deepcopy(self.my_model))
        feature_model.eval()
        return feature_model
        
        
        