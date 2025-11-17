import os, math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.manifold import TSNE
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models.Regressor import base_model
from dataloaders import proposed_loader
from utils.tqdm_config import get_tqdm_config
from utils.calculate_statistics import calculate_statistics
from utils.saver import Saver

import torch.optim as optim
from torch.amp import GradScaler, autocast

def linear_rampup(cur_epoch, warm_up, lambda_u):
    tmp = np.clip((cur_epoch - warm_up) / int(warm_up*0.8), 0.0, 1.0)

    return lambda_u * float(tmp)

class Trainer(object):
    def __init__(self, args):

        self.args = args
        self.saver = Saver(algorithm = f"{self.args.result_name}") # result dir name
        self.csv_dir = os.path.join(self.saver.experiment_dir, 'csvs')
        self.plot_dir = os.path.join(self.saver.experiment_dir, 'plot')
        [os.makedirs(f, exist_ok=True) for f in [self.csv_dir, self.plot_dir]]
        
        self.saver.save_experiment_config(self.args)
        self.writer = SummaryWriter(self.saver.experiment_dir)
        self.device = torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else torch.device('cpu')
        
        self.train_loader, self.valid_loader, self.test_loader  = proposed_loader(args)   
        
        self.train_l_source = self.train_loader['source']['labeled']
        self.train_l_target = self.train_loader['target']['labeled']
        self.train_ul_target = self.train_loader['target']['unlabeled']

        if self.args.scaler:
            self.y_min, self.y_max = calculate_statistics(self.train_l_source, self.train_l_target, args)
            self.y_min = self.y_min.to(self.device)
            self.y_max = self.y_max.to(self.device)

        self.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

        self.model = base_model(args)
        self.model.to(self.device)

        self.main_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        
        self.criterion = nn.MSELoss(reduction='none')

        for s in ['train', 'valid', 'test']:
            setattr(self, f'{s}_losses', [])
        
        self.args.best_val_loss = np.inf
        self.args.best_epoch = 0
        self.cnt_train, self.cnt_val, self.cnt_test = 0, 0, 0 
                
        self.scaler = GradScaler()
         
    def training(self, epoch):                 

        self.model.train()
        losses = 0.0
        loss_sups = 0.0
        loss_cr_ps = 0.0
        loss_cr_fs = 0.0
        
        total_steps = len(self.train_l_target)
        lambda_1 = linear_rampup(epoch, self.args.warm_up, self.args.lambda_1)
        lambda_2 = linear_rampup(epoch, self.args.warm_up, self.args.lambda_2)
    
        print('[Epoch: %d]' % (epoch))
                    
        preds_s, labels_s = [], []
        preds_t, labels_t = [], []
        with tqdm(**get_tqdm_config(total=total_steps, leave=True, color='blue')) as pbar:

            for idx, (l_source, l_target, ul_target) in enumerate(zip(self.train_l_source, self.train_l_target, self.train_ul_target)):
                self.main_optimizer.zero_grad()

                source_x, source_label = l_source['weak'], l_source['label']
                target_l_x_weak, target_label = l_target['weak'], l_target['label']
                target_ul_x_weak, target_ul_x_strong, target_ul_x_mix = ul_target['weak'], ul_target['strong'], ul_target['mix']
                
                source_x = source_x.cuda(self.device).float() 
                source_label = source_label.cuda(self.device).float()
                target_l_x_weak = target_l_x_weak.cuda(self.device).float() 
                target_label = target_label.cuda(self.device).float()
                target_ul_x_weak = target_ul_x_weak.cuda(self.device).float() 
                target_ul_x_strong = target_ul_x_strong.cuda(self.device).float() 
                target_ul_x_mix = target_ul_x_mix.cuda(self.device).float() 

                # scaler
                if self.args.scaler == True:
                    source_label = (source_label - self.y_min) / (self.y_max - self.y_min)
                    target_label = (target_label - self.y_min) / (self.y_max - self.y_min)
                       
                with autocast(device_type="cuda", dtype=torch.float32):
                    x_all = torch.cat([source_x, target_l_x_weak, target_ul_x_weak, target_ul_x_strong, target_ul_x_mix], axis=0)
                    output_first = self.model(x_all)
                    pred_source, pred_l_target, pred_ul_target_weak, pred_ul_target_strong, pred_ul_target_mix = output_first['pred'].chunk(5)
                    _, _, _, f_ul_target_weak, f_ul_target_strong, f_ul_target_mix = output_first['feature'].chunk(5)

                    loss_sup_s = self.criterion(pred_source, source_label).mean() / 2
                    loss_sup_t = self.criterion(pred_l_target, target_label).mean() / 2
                    loss_sup = loss_sup_s + loss_sup_t       

                    loss_cr_p_strong = self.criterion(pred_ul_target_strong, pred_ul_target_weak.detach()).mean() 
                    loss_cr_p_mix = self.criterion(pred_ul_target_mix, pred_ul_target_weak.detach()).mean()
                    loss_cr_p = (loss_cr_p_strong + loss_cr_p_mix) * lambda_1

                    var_weak = f_ul_target_weak.var(dim=0)
                    var_strong = f_ul_target_strong.var(dim=0)
                    var_mix = f_ul_target_mix.var(dim=0)
                    loss_cr_f_strong = nn.L1Loss()(var_strong, var_weak)
                    loss_cr_f_mix = nn.L1Loss()(var_mix, var_weak)
                    loss_cr_f = (loss_cr_f_strong + loss_cr_f_mix) * lambda_2
           
                    loss =  loss_sup + loss_cr_p + loss_cr_f 
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.main_optimizer)
                self.scaler.update()
                        
                losses += loss.item()
                loss_sups += loss_sup.item()
                loss_cr_ps += loss_cr_p.item()
                loss_cr_fs += loss_cr_f.item()

                self.writer.add_scalars(
                        'Training iteration',
                        {'Loss': loss.item(),
                        'loss_sup': loss_sup.item(),
                        'loss_cr_p' : loss_cr_p.item(),  
                        'loss_cr_f' : loss_cr_f.item()
                        },
                        global_step=self.cnt_train
                    )

                self.cnt_train += 1

                preds_s.append(pred_source.detach().cpu())
                labels_s.append(source_label.detach().cpu())
                preds_t.append(pred_l_target.detach().cpu())
                labels_t.append(target_label.detach().cpu())
                                           
                pbar.set_description("Train(%2d/%2d)-Loss: %.4f|Loss_sup: %.4f|Loss_cr_p: %.4f|Loss_cr_t: %.4f"
                                     %(idx+1, total_steps, losses/(idx+1), loss_sups/(idx+1), loss_cr_ps/(idx+1), loss_cr_fs/(idx+1)))
                pbar.update(1)
            
            self.writer.add_scalars(
                    'Training epoch',
                    {'Loss': losses/(idx+1),
                     'loss_sup': loss_sups/(idx+1),
                     'loss_cr_p' : loss_cr_ps/(idx+1),    
                     'loss_cr_f' : loss_cr_fs/(idx+1),              
                    },
                    global_step=epoch
                )
            
            t_mse, t_r2, t_mae  = self.get_regression_measures(preds_t, labels_t, epoch, "train", csv=False)
            s_mse, s_r2, s_mae  = self.get_regression_measures(preds_s, labels_s, epoch, "train", csv=False)

            losses /= (idx+1)     
            self.train_losses.append(losses)  
            pbar.set_description("Train(%2d/%2d)-Loss: %.4f|Loss_t: %.4f|r2_t: %.4f|mae_t: %.4f|Loss_s: %.4f|r2_s: %.4f|mae_s: %.4f"
                                     %(epoch, self.args.epochs, losses, t_mse, t_r2, t_mae, s_mse, s_r2, s_mae))   
                    
        return 3
                
    @torch.no_grad()
    def evaluation(self, epoch: int, phase: str, save: bool=False):
        
        self.model.eval()
        losses = 0.0
        
        if phase == "Valid":
            data_loader, c = self.valid_loader['target'], 'green'
        elif phase == "Test":
            data_loader, c = self.test_loader['target'], 'red'
        
        total_steps = len(data_loader)

        frobs, preds, labels = [], [], []
        with tqdm(**get_tqdm_config(total=total_steps, leave=True, color=c)) as pbar:
            
            for idx, target in enumerate(data_loader):
                labeled_x, label = target['feature'], target['label']
                labeled_x = labeled_x.cuda(self.device).float()
                label = label.cuda(self.device).float()

                if self.args.scaler == True:
                    label = (label - self.y_min) / (self.y_max - self.y_min)

                # predict
                out = self.model(labeled_x)
                pred, feature = out['pred'], out['feature']

                loss = self.criterion(pred, label).mean()
                losses += loss.item()

                frobs.append(torch.norm(feature, dim=1).detach())
                preds.append(pred.detach())
                labels.append(label.detach())
                _, r2, mae  = self.get_regression_measures(preds, labels, epoch=epoch, phase=phase, csv=False)
                      
                pbar.set_description(
                    "%5s(%2d/%2d)-Loss: %.4f|R2:%.3f|MAE:%.3f"%(
                        phase, idx+1, total_steps, losses/(idx+1), r2, mae))
                pbar.update(1)
            
            mses, r2s, maes = self.get_regression_measures(preds, labels, epoch=epoch, phase=phase, csv=True)

            losses = float(mses)
            getattr(self, f"{phase.lower()}_losses").append(losses)


            pbar.set_description(
                "%5s(%2d/%2d)-Loss: %.4f|R2:%.3f|MAE:%.3f"%(
                    phase, epoch, self.args.epochs, mses, r2s, maes
                )
            )   
            
            self.writer.add_scalars(
                    f'{phase} epoch',
                    {'mse': mses,
                     'r2': r2s,
                     'mae': maes
                     },
                    global_step=epoch
                )
            
            if phase == 'Valid':
                if losses < self.args.best_val_loss:
                    self.args.best_val_loss = losses
                    self.args.best_epoch = epoch

                    self.args.valid_r2s = float(r2s)
                    self.args.valid_maes = float(maes)
                    self.args.valid_mses = float(mses)
                                                            
                    torch.save(
                            self.model.state_dict(),
                            os.path.join(self.saver.experiment_dir, 'best_model.pth')
                            )     
                    self.saver.save_experiment_config(self.args)
                    save = True
                                
            if phase == 'Test' and save==True:

                self.args.test_r2s = float(r2s)
                self.args.test_maes = float(maes)
                self.args.test_mses = float(mses)

                self.args.frobenius_norm = float(torch.cat(frobs).mean())
                               
                self.saver.save_experiment_config(self.args)
                
            return save, losses


    def get_regression_measures(self, preds, labels, epoch, phase, csv=False):
        # inverse_transform       
        preds = torch.cat(preds).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

        # inverse standard scaler
        if self.args.scaler == True:
            min, max = self.y_min.cpu().numpy(), self.y_max.cpu().numpy()
            preds = preds * (max - min) + min
            labels = labels * (max - min) + min

        mse, r2, mae = mean_squared_error(labels, preds), r2_score(labels, preds), mean_absolute_error(labels, preds)
        
        # save pred vs real to csv file
        if csv==True:
            preds = pd.DataFrame(preds, columns=[f"pred{i+1}" for i in range(preds.shape[-1])]) 
            labels = pd.DataFrame(labels, columns=[f"labels{i+1}" for i in range(labels.shape[-1])]) 
            
            preds = pd.concat([preds, labels], axis=1)
            preds.to_csv(os.path.join(self.csv_dir, f'preds_labels_{phase}_{str(epoch)}.csv'), index=False)

        return mse, r2, mae
