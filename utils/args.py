import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def PROPOSED_PARSER(): 
    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--result-name', type=str, default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)

    parser.add_argument('--scaler', type=bool, default=True, help='Normalize(MinMax) scaler')

    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50']) 
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)') 
    parser.add_argument('--batch-size', type=int, default=48, help='input batch size for training') 
    parser.add_argument('--lr', type=float, default= 0.0001, help='learning rate of optimizer')

    parser.add_argument('--dataset', type=str, default='biwi', choices=['biwi', 'qmul', 'mpi3d'], help='dataset')
    parser.add_argument('--source', type=str, default='M', choices=['F', 'M', 'real', 'realistic', 'toy'], help='source data')
    parser.add_argument('--target', type=str, default='F', choices=['F', 'M', 'real', 'realistic', 'toy'], help='target data')
    parser.add_argument('--label-target-per', type=str, default='5', choices=['5', '10', '20', '30', '100'], help='target data') 


    return parser
