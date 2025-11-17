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
# https://github.com/ismailnejjar/DARE-GRAM/tree/main?tab=readme-ov-file
# https://drive.google.com/drive/folders/1HBZgMxf_KgbIench770SG_ii4PgxPkO0

portion = {
    '5': "labeled_5",
    '10': "labeled_10",
    '20': "labeled_20",
    '30': "labeled_30",
    '100' : "labeled_full",
}

def load_gt(target, dataset):
    regression_targets = [float(x) for x in target.split(' ')[-2:]]

    return regression_targets

def mpi3d_collections(args):
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
    domain_ls_path = os.path.join(args.data_root, str(args.source)+ '_source.txt')
    domain_reader = open(domain_ls_path, 'r')

    data_lines = domain_reader.readlines()
    domain_reader.close()
    for i, line in enumerate(data_lines):
        id = line.strip().split(' ')[0]
        gt = load_gt(line, args.dataset)
        data_collection['source']['train']['ids'].append(args.data_root +'/'+ id)
        data_collection['source']['train']['gt'].append(gt)
        
    target_partitions = [portion.get(args.label_target_per, "labeled"), 'unlabeled', 'valid', 'test']
    for item in target_partitions:
        t_p = item.split("_")[0]
        domain_ls_path = os.path.join(
            args.data_root, str(args.target) + '_' + item  + '.txt'
        )
        domain_reader = open(domain_ls_path, 'r')
        for line in domain_reader:
            id = line.strip().split(' ')[0]
            gt = load_gt(line, args.dataset)
            data_collection['target'][t_p]['ids'].append(args.data_root +'/'+ id)
            data_collection['target'][t_p]['gt'].append(gt)

        domain_reader.close()

    return data_collection

class mpi3d(Dataset):
    def __init__(self, data):
        self.data = data
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225] 
        self.resize_size=(224, 224)
        self.transform = T.Compose([
                                    T.Resize(self.resize_size),
                                    T.ToTensor(),
                                    T.Normalize(mean=self.mean, std=self.std )                                    
                                    ])   

        self.images, self.gt = self.data['ids'], self.data['gt']

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image_pth = self.images[index % len(self.images)]
        img = Image.open(image_pth).convert('RGB')

        label = torch.Tensor(self.gt[index % len(self.images)])

        return dict(feature=self.transform(img), label=label)

class mpi3d_aug(Dataset):
    def __init__(self, data): 
        self.data = data
        self.images, self.gt = self.data['ids'], self.data['gt']
        self.len_images = len(self.images)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225] 
        self.resize_size=(224, 224)

        # Weak augmentation for angle estimation 
        # https://arxiv.org/html/2408.01566v1
        self.augment_pipeline = transforms.Compose([
                                transforms.RandomCrop(
                                    size=self.resize_size,
                                    padding=int(self.resize_size[0]*0.125),
                                    padding_mode='constant'
                                ),
                                # transforms.GaussianBlur(kernel_size=3),
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
        
        self.normalize = T.Compose([
                                    T.Normalize(mean=self.mean, std=self.std )                                    
                                    ])
                                            
    def __len__(self):
        return self.len_images
        
    def __getitem__(self, index):
        image_pth = self.images[index % len(self.images)]
        img = Image.open(image_pth).convert('RGB')

        label = torch.Tensor(self.gt[index % len(self.images)])
     
        weak = self.weak_aug(img)
        strong = self.strong_aug(img)
        alpha = np.random.uniform(0, 1)
        mix = (weak * alpha) + (strong * (1 - alpha))

        return dict(weak=self.normalize(weak), strong=self.normalize(strong), mix=self.normalize(mix), label=label)
        # transforms.ToPILImage()(img).show()