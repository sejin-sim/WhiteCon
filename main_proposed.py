import os, torch, gc, argparse
import numpy as np

from utils.args import PROPOSED_PARSER
from method.VAWReg import Trainer

def main():

    gc.collect()
    torch.cuda.empty_cache()
    
    parser = PROPOSED_PARSER() 
   
    # for semi
    parser.add_argument('--warm-up', type=int, default=20, help='warmup epoch of cr loss')
    parser.add_argument('--lambda-1', type=float, default= 0.1, help='lambda_1 for crp') 
    parser.add_argument('--lambda-2', type=float, default= 0.1, help='lambda_2 for crf') 

    args = parser.parse_args()

    trainer = Trainer(args)

    flag_loss = np.inf     
    for epoch in range(1, args.epochs+1, 1):
        trainer.training(epoch)
        
        s, val_loss = trainer.evaluation(epoch, 'Valid', save=False)        
        if val_loss < flag_loss:
            _ = trainer.evaluation(epoch, 'Test', save=s)
            flag_loss = val_loss
                    
if __name__ == '__main__':
    main()



