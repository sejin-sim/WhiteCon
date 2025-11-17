import torch
import torch.nn as nn
import torchvision.models as model
from models.backbone.resnet_whitening import resnet18, resnet50

class ResNet_Backbone(nn.Module):
    def __init__(self, args):
        super(ResNet_Backbone, self).__init__()
        
        model_dict = {
        'resnet18': resnet18,
        'resnet50': resnet50
        }

        self.encoder = model_dict[args.model]()

    def forward(self, x):
        x = self.encoder(x) 
        return x
    
class base_model(nn.Module):
    def __init__(self, args, **kwargs):
        super(base_model, self).__init__()
        self.args = args

        self.backbone = ResNet_Backbone(args)
        
        n_output = {'biwi':3, 'qmul':2, 'mpi3d':2}
        self.regressor = nn.Linear(self.backbone.encoder.fc_out.in_features, n_output[args.dataset])
    
    def forward(self, x):
        feature = self.backbone(x)         
        feature = torch.flatten(feature, 1) 

        pred = self.regressor(feature)
          
        return {
            'feature':feature,  
            'pred': pred
                    }
