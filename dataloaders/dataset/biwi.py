import os, random
import numpy as np
import torch, glob
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from natsort import natsorted
import matplotlib.pyplot as plt
from utils.randaugment import RandAugmentMC
# https://github.com/kuhnkeF/headposeplus/tree/main (bounding box)


portion = {
    '5': "labeled_5",
    '10': "labeled_10",
    '20': "labeled_20",
    '30': "labeled_30",
    '100' : "labeled_full",
}

def load_gt(file_path):

    with open(file_path, 'r') as pose_annot:
        R = []
        for line in pose_annot:
            line = line.strip('\n').split(' ')
            l = []
            if line[0] != '':
                for nb in line:
                    if nb == '':
                        continue
                    l.append(float(nb))
                R.append(l)
        
        R = np.array(R)
        R = R[:3, :]
        pose_annot.close()
        
        R = np.transpose(R)
        
        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

    regression_targets = list(map(float, [yaw, pitch, roll]))
    return regression_targets

def biwi_collections(args):
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
        id, gt = line.strip().split(' ')
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
            id, gt = line.strip().split(' ')
            data_collection['target'][t_p]['ids'].append(id)
            data_collection['target'][t_p]['gt'].append(gt)

        domain_reader.close()

    return data_collection

class biwi(Dataset):
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
        self.bbox = dict(np.load("data/biwi/hpdb/Biwi_plus.npz"))

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image_pth = os.path.join('data/biwi/hpdb', self.images[index % len(self.images)])
        gt_pth = os.path.join('data/biwi/hpdb', self.gt[index % len(self.images)])

        # bounding box
        bbox = self.bbox['bbox'][np.where(self.bbox['images'] == self.images[index])[0]][0]
        x, y, w, h = bbox
        x, y, w, h = (int(x), int(y), int(w), int(h))
        x_min, y_min, x_max, y_max = x, y, x + w - 1, y + h - 1
        img = Image.open(image_pth).convert('RGB').crop((x_min, y_min, x_max, y_max))

        labels = load_gt(gt_pth)
        label = torch.Tensor(labels)

        return dict(feature=self.transform(img), label=label)

class biwi_aug(Dataset):
    def __init__(self, data): 
        self.data = data
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225] 
        self.resize_size=(224, 224)
        
        self.images, self.gt = self.data['ids'], self.data['gt']
        self.bbox = dict(np.load("data/biwi/hpdb/Biwi_plus.npz"))

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
        
        self.normalize = T.Compose([
                                    T.Normalize(mean=self.mean, std=self.std )                                    
                                    ])
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, index):
        image_pth = os.path.join('data/biwi/hpdb', self.images[index % len(self.images)])
        gt_pth = os.path.join('data/biwi/hpdb', self.gt[index % len(self.images)])

        # bounding box
        bbox = self.bbox['bbox'][np.where(self.bbox['images'] == self.images[index])[0]][0]
        x, y, w, h = bbox
        x, y, w, h = (int(x), int(y), int(w), int(h))
        x_min, y_min, x_max, y_max = x, y, x + w - 1, y + h - 1
        img = Image.open(image_pth).convert('RGB').crop((x_min, y_min, x_max, y_max))

        labels = load_gt(gt_pth)
        label = torch.Tensor(labels)

        weak = self.weak_aug(img)
        strong = self.strong_aug(img)
        alpha = np.random.uniform(0, 1)
        mix = (weak * alpha) + (strong * (1 - alpha))

        return dict(weak=self.normalize(weak), strong=self.normalize(strong), mix=self.normalize(mix), label=label)
        # transforms.ToPILImage()(mix).save('img/mix.png')
    