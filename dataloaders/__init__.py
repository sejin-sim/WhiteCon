from torch.utils.data import DataLoader
import os, glob
import random
import torch
from torch.utils.data import DataLoader, Sampler
import numpy as np
from dataloaders.dataset.biwi import biwi_collections, biwi, biwi_aug 
from dataloaders.dataset.qmul import qmul_collections, qmul, qmul_aug 
from dataloaders.dataset.mpi3d import mpi3d_collections, mpi3d, mpi3d_aug


def seed_everything(seed: int = 2024):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
class RandomSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def proposed_loader(args):
    seed_everything(args.seed)

    collection_dict = {
        'mpi3d': mpi3d_collections,
        'biwi': biwi_collections,
        'qmul': qmul_collections
        }
    
    data_collection = collection_dict[args.dataset](args)
   
    dataset_dict = {
        'mpi3d': mpi3d,
        'biwi': biwi,
        'qmul': qmul
        }
    dataset_dict_aug = {
        'mpi3d': mpi3d_aug,
        'biwi': biwi_aug,
        'qmul': qmul_aug,
        }   
    
    datasets = {'source': {}, 'target': {}}
    
    datasets['source']['train'] = dataset_dict_aug[args.dataset](data=data_collection['source']['train'])
    datasets['target']['labeled'] = dataset_dict_aug[args.dataset](data=data_collection['target']['labeled'])
    datasets['target']['unlabeled'] = dataset_dict_aug[args.dataset](data=data_collection['target']['unlabeled'])

    datasets['target']['valid'] = dataset_dict[args.dataset](data=data_collection['target']['valid'])
    datasets['target']['test'] = dataset_dict[args.dataset](data=data_collection['target']['test'])
    
    max_step = max(len(datasets['source']['train']), len(datasets['target']['labeled']), len(datasets['target']['unlabeled'])) 

    args.batch_divisor = 5

    # loader 생성 생성
    data_loaders = {
        'train': {
            'source': {
                'labeled': DataLoader(datasets['source']['train'], 
                                      batch_size=int(args.batch_size // args.batch_divisor), 
                                      drop_last=True,
                                      sampler=RandomSampler(len(datasets['source']['train']), max_step)
                                      ),
            },
            'target': {
                'labeled': DataLoader(datasets['target']['labeled'], 
                                      batch_size=int(args.batch_size // args.batch_divisor), 
                                      drop_last=True,
                                      sampler=RandomSampler(len(datasets['target']['labeled']), max_step)
                                      ),
                'unlabeled': DataLoader(datasets['target']['unlabeled'], 
                                        batch_size=int(args.batch_size // args.batch_divisor), 
                                        drop_last=True,
                                        sampler=RandomSampler(len(datasets['target']['unlabeled']), max_step)
                                        ),
            },
        },

        'valid': {
            'target': DataLoader(datasets['target']['valid'], 
                                 batch_size=args.batch_size, 
                                 shuffle=False,
                                 drop_last=False)
        },
     
        'test': {
            'target': DataLoader(datasets['target']['test'], 
                                 batch_size=args.batch_size, 
                                 shuffle=False,
                                 drop_last=False)
        }
     }
    
    
    return data_loaders['train'], data_loaders['valid'], data_loaders['test']