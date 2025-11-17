import os, torch, glob, sys
from tqdm import tqdm
from itertools import product
import torch.nn.functional as F

from dataloaders.dataset.biwi import load_gt

def calculate_statistics(train_l_source=None, train_l_target=None, args=None):

    labels = []

    if args.dataset == 'mpi3d' or args.dataset == 'qmul':
        if train_l_source:
            s_labels = [s_label for s_label in train_l_source.dataset.gt]
            labels.extend(s_labels)
        
        t_labels = [t_label for t_label in train_l_target.dataset.gt]
        labels.extend(t_labels)

    elif args.dataset == 'biwi':
        if train_l_source:
            s_labels = [load_gt('data/biwi/hpdb/'+ s_label) for s_label in train_l_source.dataset.gt]
            labels.extend(s_labels)
        
        t_labels = [load_gt('data/biwi/hpdb/'+ t_label) for t_label in train_l_target.dataset.gt]
        labels.extend(t_labels)

    labels = torch.Tensor(labels)

    y_min = labels.min(dim=0)[0]
    y_max = labels.max(dim=0)[0]

    return y_min, y_max

