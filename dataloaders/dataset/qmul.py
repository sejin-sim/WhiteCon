import os, random
import numpy as np
import torch, glob
from PIL import Image, ImageFilter
import torchvision.transforms as T
import torch.nn.functional as F
import xmltodict
import cv2
import math
from torch.utils.data import Dataset
from torchvision import transforms
from natsort import natsorted
import matplotlib.pyplot as plt
from utils.randaugment import RandAugmentMC
# https://github.com/BayesWatch/deep-kernel-transfer/blob/61d95d6ab783be679c09803a33e3c2302287cbc4/data/qmul_loader.py#L4

portion = {
    '5': "labeled_5",
    '10': "labeled_10",
    '20': "labeled_20",
    '30': "labeled_30",
    '100' : "labeled_full",
}

def num_to_str(num):
    str_ = ''
    if num == 0:
        str_ = '000'
    elif num < 100:
        str_ = '0' + str(int(num))
    else:
        str_ = str(int(num))
    return str_

def load_gt(file_path):
    # pitch(=tilt) 60~120, yaw 0~180(정면 90도)
    regression_targets = [float(x) - 90 for x in file_path.split('.')[0].split('_')[-2:]] #  +/-30 & +/-90 로 변경 
    return regression_targets


def qmul_collections(args):
    args.data_root = 'data/' + args.dataset
    data_collection = {
        'source':{
            'train': {'ids':[], 'gt':[]},
        },
        'target':{
            'labeled': {'ids':[], 'gt':[]},
            'unlabeled': {'ids':[], 'gt':[]},
            'valid': {'ids':[], 'gt':[]},
            'test': {'ids':[], 'gt':[]}
        }
    }

    domain_ls_path = os.path.join(args.data_root, str(args.source) + '_source.txt')
    domain_reader = open(domain_ls_path, 'r')

    data_lines = domain_reader.readlines()
    domain_reader.close()
    for i, line in enumerate(data_lines):
        id = line.strip()
        gt = load_gt(id)
        data_collection['source']['train']['ids'].append(id)
        data_collection['source']['train']['gt'].append(gt)
            
    target_partitions = [portion.get(args.label_target_per, "labeled"), 'unlabeled', 'valid', 'test']
    for item in target_partitions:
        t_p = item.split("_")[0]
        domain_ls_path = os.path.join(
            args.data_root, str(args.target) + '_' + item  + '.txt'
        )
        domain_reader = open(domain_ls_path, 'r')
        for line in domain_reader:
            id = line.strip()
            gt = load_gt(id)

            data_collection['target'][t_p]['ids'].append(id)
            data_collection['target'][t_p]['gt'].append(gt)

        domain_reader.close()

    return data_collection

class qmul(Dataset):
    def __init__(self, data):
        self.data = data
        self.mean_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225]
        self.mean_gray = [0.449, 0.449, 0.449]
        self.std_gray = [0.226, 0.226, 0.226]
        self.resize_size=(224, 224)
        self.transform = T.Compose([
                                    T.Resize(self.resize_size),
                                    T.ToTensor()
                                    ])   

        self.images, self.gt = self.data['ids'], self.data['gt']

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image_pth = self.images[index % len(self.images)]

        if 'Colour' in image_pth:
            img = Image.open(image_pth).convert('RGB')
            mean = self.mean_rgb
            std = self.std_rgb
        else:
            img = Image.open(image_pth).convert('RGB')
            mean = self.mean_gray 
            std = self.std_gray

        label = torch.Tensor(self.gt[index % len(self.images)])

        normalize = T.Compose([
                    T.Normalize(mean=mean, std=std)                                    
                ])
        
        return dict(feature=normalize(self.transform(img)), label=label)

class qmul_aug(Dataset):
    def __init__(self, data): 
        self.data = data
        self.mean_rgb = [0.485, 0.456, 0.406]
        self.std_rgb = [0.229, 0.224, 0.225]
        self.mean_gray = [0.449, 0.449, 0.449]
        self.std_gray = [0.226, 0.226, 0.226]

        self.resize_size=(224, 224)
        
        self.images, self.gt = self.data['ids'], self.data['gt']

        # Weak augmentation for angle estimation 
        # https://arxiv.org/html/2408.01566v1
        self.augment_pipeline = transforms.Compose([
                                transforms.RandomCrop(
                                    size=self.resize_size,
                                    padding=int(self.resize_size[0]*0.125),
                                    padding_mode='constant'
                                ),
                                transforms.GaussianBlur(kernel_size=3),
                                ])

        self.weak_aug = T.Compose([
                                T.Resize(self.resize_size),
                                self.augment_pipeline,
                                T.ToTensor(),
                                ])

        self.strong_aug = T.Compose([
                                    T.Resize(self.resize_size),
                                    self.augment_pipeline,
                                    RandAugmentMC(n=2, m=10), 
                                    T.ToTensor(),
                                    ])
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image_pth = self.images[index % len(self.images)]

        if 'Colour' in image_pth:
            img = Image.open(image_pth).convert('RGB')
            mean = self.mean_rgb
            std = self.std_rgb
        else: # 흑백인 경우
            img = Image.open(image_pth).convert('RGB')
            mean = self.mean_gray 
            std = self.std_gray

        img = Image.open(image_pth).convert('RGB')

        label = torch.Tensor(self.gt[index % len(self.images)])

        weak = self.weak_aug(img)
        strong = self.strong_aug(img)
        alpha = np.random.uniform(0, 1)
        mix = (weak * alpha) + (strong * (1 - alpha))

        normalize = T.Compose([
                    T.Normalize(mean=mean, std=std)                                    
                ])

        return dict(weak=normalize(weak), strong=normalize(strong), mix=normalize(mix), label=label)
    