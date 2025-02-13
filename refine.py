from config import parser_handle
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import time
import numpy as np
from torch.utils.data import DataLoader

from utils.unet import UNet, UNetLittle
from utils.log import Log
from utils.losses import SupConLoss
from utils.getdataset import GetDataset
from utils.getmodel import GetModel

parser = parser_handle()
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

if not os.path.exists(args.refine_res):
    os.mkdir(args.refine_res)

if args.dataset == 'CIFAR10':
    num_classes = 10
    first_channel = 64
elif args.dataset == 'ImageNet50':
    num_classes = 50
    first_channel = 32
else:
    raise NotImplementedError

class Adv_Program(nn.Module):
    def __init__(self):
        super(Adv_Program, self).__init__()
        self.init_net()
        self.init_label_shuffle()
        self.init_unet()
        
    def init_net(self):
        getmodel = GetModel(args.dataset, args.model, args.attack)
        self.net = getmodel.get_model()
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

    def init_unet(self):
        self.unet = UNetLittle(args=None, n_channels=3, n_classes=3, first_channels=first_channel)

    def init_label_shuffle(self):
        start = 0
        end = num_classes
        arr = np.array([i for i in range(num_classes)])
        arr_shuffle = np.array([i for i in range(num_classes)])
        while True:
            num = sum(arr[start:end] == arr_shuffle[start:end])
            if num == 0:
                break
            np.random.shuffle(arr_shuffle[start:end])
        args.arr_shuffle = arr_shuffle

    def label_shuffle(self, label):
        label_new = torch.zeros_like(label)
        index = torch.from_numpy(args.arr_shuffle).repeat(label.shape[0], 1).cuda()
        label_new = label_new.scatter(1, index, label)
        return label_new
    
    def label_shift(self, label):
        return torch.roll(label, 1, 1)

    def forward(self, image):
        self.X_adv = torch.clamp(self.unet(image), 0, 1)
        # self.X_adv = F.normalize(self.X_adv)
        self.Y_adv = self.net(self.X_adv)
        Y_adv = F.softmax(self.Y_adv, 1)
        return self.label_shuffle(Y_adv)
        # return Y_adv

class Adversarial_Reprogramming():
    def __init__(self):
        self.init_dataset()
        self.program = Adv_Program()
        self.init_mode()
    
    def init_dataset(self):     
        getdataset = GetDataset(args.dataset, args.model, args.attack, args.tlabel)
        train_dataset, test_dataset = getdataset.get_benign_dataset()
        _, poisoned_test_dataset = getdataset.get_poisoned_dataset()

        kwargs = {
            'batch_size': args.batch_size,
            'num_workers': 4,
            'shuffle': True,
            'pin_memory': True,
            'drop_last': True,
        }
        self.train_loader = DataLoader(train_dataset, **kwargs)
        self.test_loader = DataLoader(test_dataset, **kwargs)
        self.poisoned_test_loader = DataLoader(poisoned_test_dataset, **kwargs)

    def init_mode(self):
        self.save_dir = os.path.join(args.refine_res, args.dataset, args.model, args.attack, time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.log = Log(os.path.join(self.save_dir, 'log.txt'))
        for arg, value in args.__dict__.items():
            self.log(arg + ':' + str(value))
        torch.save(args.arr_shuffle, os.path.join(self.save_dir, 'label_shuffle.pth'))

        self.lossfunc = torch.nn.BCELoss().cuda()
        self.supconlossfunc = SupConLoss().cuda()
        
        if args.optim == 'SGD':
            self.optim = torch.optim.SGD(self.program.unet.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        elif args.optim == 'Adam':
            self.optim = torch.optim.Adam(self.program.unet.parameters(), lr=args.lr, betas=(0.9, 0.999))

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[args.epoch-50, args.epoch-20], gamma=args.decay)
        self.lossfunc.cuda()
        self.program.cuda()
        
    def compute_loss(self, y_out, y_label):
        y_label = torch.zeros(args.batch_size, num_classes).cuda().scatter_(1, y_label.view(-1, 1), 1)
        # y_label = y_label.cuda()
        return self.lossfunc(y_out, y_label)

    def validate(self):
        top1 = 0
        for image, label in self.test_loader:
            image = image.cuda()
            # image = 0.95 * image + 0.05 * torch.rand(size=image.shape, device='cuda')
            out = self.program(image)
            # if k == 1:
            #     transforms.ToPILImage()(image[0]).save('./pics/benign_image.png')
            #     transforms.ToPILImage()(self.program.X_adv[0]).save('./pics/benign_x_adv.png')
            pred = out.detach().cpu().numpy().argmax(1)
            top1 += sum(label.numpy() == pred)
        acc = top1 / float(args.batch_size * len(self.test_loader))
        self.log('==========Test result on benign test dataset==========')
        self.log('[%s] Top-1 correct / Total: %d/%d, Top-1 accuracy: %.6f' % 
                 (time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()), top1, args.batch_size * len(self.test_loader), acc))

    def validate_poisoned(self):
        top1 = 0
        top1_1 = 0
        for image, label in self.poisoned_test_loader:
            image = image.cuda()
            out = self.program(image)
            # if k == 1:
            #     transforms.ToPILImage()(image[0]).save('./pics/poisoned_image.png')
            #     transforms.ToPILImage()(self.program.X_adv[0]).save('./pics/poisoned_x_adv.png')
            pred = out.detach().cpu().numpy().argmax(1)
            top1 += sum(label.numpy() == pred)
            # asr_1 += np.sum(pred == 1)
            top1_1 += np.sum(pred == args.arr_shuffle[0])
        asr = top1 / float(args.batch_size * len(self.poisoned_test_loader))
        asr_1 = top1_1 / float(args.batch_size * len(self.poisoned_test_loader))
        self.log('==========Test result on poisoned test dataset==========')
        self.log('[%s] Top-1 correct / Total: %d/%d, Top-1 accuracy: %.6f' % 
                 (time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()), top1, args.batch_size * len(self.poisoned_test_loader), asr))
        # self.log('[%s] Top-1 (label %d) correct / Total: %d/%d, Top-1 accuracy: %.6f' % 
        #          (time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()), args.arr_shuffle[0], top1_1, args.batch_size * len(self.poisoned_test_loader), asr_1))

    def train(self):
        for epoch in range(args.epoch):
            self.program.unet.train()
            self.log(f'----- Epoch: {epoch+1}/{args.epoch} -----')
            for image, label in self.train_loader:
                images = image.cuda()
                # images = 0.95 * images + 0.05 * torch.rand(size=images.shape, device='cuda')
                # label = label.cuda()
                bsz = label.shape[0]
                f_logit = self.program.net(images)

                f_index = f_logit.argmax(1)
                f_label = torch.zeros_like(f_logit).cuda().scatter_(1, f_index.view(-1, 1), 1)

                logit = self.program(images)

                features = self.program.X_adv.view(bsz, -1)
                features = F.normalize(features, dim=1)
                # features = F.normalize(self.program.net.feat, dim=1)
                # features = F.normalize(self.program.Y_adv, dim=1)
                f1, f2 = features, features
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                supconloss = self.supconlossfunc(features, f_index)
                
                self.loss = self.lossfunc(logit, f_label) + args.sup * supconloss
                
                self.optim.zero_grad()
                self.loss.backward()
                self.optim.step()

            self.lr_scheduler.step()
            self.log('[%s] Epoch: %d/%d, lr: %lf, loss: %lf' % 
                     (time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()), epoch+1, args.epoch, self.optim.param_groups[0]['lr'], self.loss))
           
            if epoch > args.epoch - 10:
                torch.save(self.program.unet.state_dict(), 
                           os.path.join(self.save_dir, f'unet_epoch{epoch+1}.pth'))

            self.program.unet.eval()
            if epoch % 10 == 0 or epoch > args.epoch - 10:
                with torch.no_grad():
                    self.test()
    
    def test(self):
        self.validate()
        if self.poisoned_test_loader:
            self.validate_poisoned()

    def valid_net(self):
        top1 = 0
        for image, label in self.test_loader:
            image = image.cuda()
            # image = 0.95 * image + 0.05 * torch.rand(size=image.shape, device='cuda')
            out = self.program.net(image)
            # if k == 1:
            #     transforms.ToPILImage()(image[0]).save('./pics/benign_image.png')
            #     transforms.ToPILImage()(self.program.X_adv[0]).save('./pics/benign_x_adv.png')
            pred = out.detach().cpu().numpy().argmax(1)
            top1 += sum(label.numpy() == pred)
        acc = top1 / float(args.batch_size * len(self.test_loader))
        self.log('==========Test origin model result on benign test dataset==========')
        self.log('[%s] Top-1 correct / Total: %d/%d, Top-1 accuracy: %.6f' % 
                 (time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()), top1, args.batch_size * len(self.test_loader), acc))

        if self.poisoned_test_loader:
            top1 = 0
            for image, label in self.poisoned_test_loader:
                image = image.cuda()
                out = self.program.net(image)
                # if k == 1:
                #     transforms.ToPILImage()(image[0]).save('./pics/poisoned_image.png')
                #     transforms.ToPILImage()(self.program.X_adv[0]).save('./pics/poisoned_x_adv.png')
                pred = out.detach().cpu().numpy().argmax(1)
                top1 += sum(label.numpy() == pred)
            asr = top1 / float(args.batch_size * len(self.poisoned_test_loader))
            self.log('==========Test origin model result on poisoned test dataset==========')
            self.log('[%s] Top-1 correct / Total: %d/%d, Top-1 accuracy: %.6f' % 
                    (time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime()), top1, args.batch_size * len(self.poisoned_test_loader), asr))

def main():
    AR = Adversarial_Reprogramming()
    with torch.no_grad():
        AR.valid_net()
    AR.train()

if __name__ == '__main__':
    main()