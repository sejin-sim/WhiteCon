# https://github.com/roysubhankar/dwt-domain-adaptation

import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
from torchvision import transforms

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class _Whitening(nn.Module):
	def __init__(self, num_features, group_size, running_m=None, running_var=None, momentum=0.1, track_running_stats=True, eps=1e-3, alpha=1):
		super(_Whitening, self).__init__()
		self.num_features = num_features
		self.momentum = momentum
		self.track_running_stats = track_running_stats
		self.eps = eps
		self.alpha = alpha
		self.group_size = min(self.num_features, group_size)
		self.num_groups = self.num_features // self.group_size
		self.running_m = running_m
		self.running_var = running_var

		if self.track_running_stats and self.running_m is not None:
			self.register_buffer('running_mean', self.running_m)
			self.register_buffer('running_variance', self.running_var)
		else:
			self.register_buffer('running_mean', torch.zeros([1, self.num_features, 1, 1], dtype=torch.float32))
			self.register_buffer('running_variance', torch.ones([self.num_groups, self.group_size, self.group_size], dtype=torch.float32))
		
	def _check_input_dim(self, input):
		raise NotImplementedError

	def _check_group_size(self):
		raise NotImplementedError

	def forward(self, x):
		self._check_input_dim(x)
		self._check_group_size()

		m = x.mean(0).view(self.num_features, -1).mean(-1).view(1, -1, 1, 1)
		if not self.training and self.track_running_stats:
			m = self.running_mean
		xn = x - m 
          
		T = xn.permute(1,0,2,3).contiguous().view(self.num_groups, self.group_size,-1)
		f_cov = torch.bmm(T, T.permute(0,2,1)) / T.shape[-1]
		f_cov_shrinked = (1-self.eps) * f_cov + self.eps * torch.eye(self.group_size, dtype=torch.float32, device=f_cov.device).repeat(self.num_groups, 1, 1)

		if not self.training and self.track_running_stats:
			f_cov_shrinked = (1-self.eps) * self.running_variance + self.eps * torch.eye(self.group_size, dtype=torch.float32, device=f_cov.device).repeat(self.num_groups, 1, 1)
		inv_sqrt = torch.inverse(torch.linalg.cholesky(f_cov_shrinked.to(torch.float32))).contiguous().view(self.num_features, self.group_size, 1, 1)
		decorrelated = nn.functional.conv2d(xn, inv_sqrt, groups = self.num_groups)

		if self.training and self.track_running_stats:
			self.running_mean = torch.add(self.momentum * m.detach(), (1 - self.momentum) * self.running_mean, out=self.running_mean) 
			self.running_variance = torch.add(self.momentum * f_cov.detach(), (1 - self.momentum) * self.running_variance, out=self.running_variance)
			
		return decorrelated

class WTransform2d(_Whitening):
	def _check_input_dim(self, input):
		if input.dim() != 4:
			raise ValueError('expected 4D input (got {}D input)'. format(input.dim()))

	def _check_group_size(self):
		if self.num_features % self.group_size != 0:
			raise ValueError('expected number of channels divisible by group_size (got {} group_size\
				for {} number of features'.format(self.group_size, self.num_features))

class whitening_scale_shift(nn.Module):
    def __init__(self, planes, group_size, running_mean=None, running_variance=None, track_running_stats=True, affine=True):
        super().__init__()
        self.wh = WTransform2d(planes, group_size, running_m=running_mean, running_var=running_variance, track_running_stats=track_running_stats)
        if affine:
            self.gamma = nn.Parameter(torch.ones(planes, 1, 1))
            self.beta = nn.Parameter(torch.zeros(planes, 1, 1))

    def forward(self, x):
        out = self.wh(x)
        if hasattr(self, 'gamma') and hasattr(self, 'beta'):
            out = out * self.gamma.to(out.device) + self.beta.to(out.device)
        return out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        layer: int,
        groups: int = 1,
        stride: int = 1,
        downsample = None):
        
        super().__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

        self.gamma1, self.gamma2 = [nn.Parameter(torch.ones(planes, 1, 1)) for _ in range(2)]
        self.beta1, self.beta2 = [nn.Parameter(torch.zeros(planes, 1, 1)) for _ in range(2)]
        self.downsample_gamma = nn.Parameter(torch.ones(planes * self.expansion, 1, 1))
        self.downsample_beta = nn.Parameter(torch.zeros(planes * self.expansion, 1, 1))
        
        if layer == 1: 
            self.bns1, self.bnt1, self.bns2, self.bnt2 = [whitening_scale_shift(planes, groups) for _ in range(4)]
            self.downsample_bns, self.downsample_bnt = [whitening_scale_shift(planes* self.expansion, groups) for _ in range(2)]
        else:
            self.bns1, self.bnt1, self.bns2, self.bnt2 = [nn.BatchNorm2d(planes) for _ in range(4)]
            self.downsample_bns, self.downsample_bnt = [nn.BatchNorm2d(planes* self.expansion, groups) for _ in range(2)]

    def _apply_bn(self, out, bn_layers, gamma, beta):
        if self.training:
            out_s, out_t = torch.split(out, split_size_or_sections=[out.shape[0] // 5, out.shape[0] - out.shape[0] // 5], dim=0)
            out = torch.cat((bn_layers[0](out_s), bn_layers[1](out_t)), dim=0) * gamma + beta
        else:
            out = bn_layers[1](out) * gamma + beta
        return out
    
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self._apply_bn(out, [self.bns1, self.bnt1], self.gamma1, self.beta1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self._apply_bn(out, [self.bns2, self.bnt2], self.gamma2, self.beta2)

        if self.downsample is not None and self.training:
            identity = self.downsample(x)
            identity = self._apply_bn(identity, [self.downsample_bns, self.downsample_bnt], self.downsample_gamma, self.downsample_beta)
        elif self.downsample is not None and self.training == False:
            identity = self.downsample(x)
            identity = self.downsample_bnt(identity) * self.downsample_gamma + self.downsample_beta

        return out
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, layer, group_size=4, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.gamma1, self.gamma2 = [nn.Parameter(torch.ones(planes, 1, 1)) for _ in range(2)]
        self.beta1, self.beta2 = [nn.Parameter(torch.zeros(planes, 1, 1)) for _ in range(2)]
        self.gamma3, self.downsample_gamma = [nn.Parameter(torch.ones(planes * self.expansion, 1, 1)) for _ in range(2)]
        self.beta3, self.downsample_beta = [nn.Parameter(torch.zeros(planes * self.expansion, 1, 1)) for _ in range(2)]
        
        if layer == 1: 
            self.bns1, self.bnt1, self.bns2, self.bnt2 = [whitening_scale_shift(planes, group_size) for _ in range(4)]
            self.bns3, self.bnt3 = [whitening_scale_shift(planes* self.expansion, group_size) for _ in range(2)]
            self.downsample_bns, self.downsample_bnt = [whitening_scale_shift(planes* self.expansion, group_size) for _ in range(2)]
        else:
            self.bns1, self.bnt1, self.bns2, self.bnt2 = [nn.BatchNorm2d(planes) for _ in range(4)]
            self.bns3, self.bnt3 = [nn.BatchNorm2d(planes * self.expansion) for _ in range(2)]
            self.downsample_bns, self.downsample_bnt = [nn.BatchNorm2d(planes* self.expansion, group_size) for _ in range(2)]

    def _apply_bn(self, out, bn_layers, gamma, beta):
        if self.training:
            out_s, out_t = torch.split(out, split_size_or_sections=[out.shape[0] // 5, out.shape[0] - out.shape[0] // 5], dim=0)
            out = torch.cat((bn_layers[0](out_s), bn_layers[1](out_t)), dim=0) * gamma + beta
        else:
            out = bn_layers[1](out) * gamma + beta
        return out
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self._apply_bn(out, [self.bns1, self.bnt1], self.gamma1, self.beta1)
        out = self.relu(out)
        out = self.conv2(out)
        out = self._apply_bn(out, [self.bns2, self.bnt2], self.gamma2, self.beta2)
        out = self.relu(out)
        out = self.conv3(out)
        out = self._apply_bn(out, [self.bns3, self.bnt3], self.gamma3, self.beta3)

        if self.downsample is not None and self.training:
            identity = self.downsample(x)
            identity = self._apply_bn(identity, [self.downsample_bns, self.downsample_bnt], self.downsample_gamma, self.downsample_beta)
        elif self.downsample is not None and self.training == False:
            identity = self.downsample(x)
            identity = self.downsample_bnt(identity) * self.downsample_gamma + self.downsample_beta

        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):        
    def __init__(self, block, layers, num_classes=65, zero_init_residual=False, group_size=4):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bns1, self.bnt1 = [whitening_scale_shift(planes=64, group_size=group_size, affine=False) for _ in range(2)]
        self.gamma1 = nn.Parameter(torch.ones(64, 1, 1))
        self.beta1 = nn.Parameter(torch.zeros(64, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], layer=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, layer=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, layer=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, layer=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights(zero_init_residual, block)

    def _make_layer(self, block, planes, blocks, layer=1, group_size=4, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
            )
        layers = []
        layers.append(block(self.inplanes, planes, layer, group_size, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer, group_size))

        return nn.Sequential(*layers)

    def _apply_bn_layers(self, x, bn_layers, gamma, beta):
        if self.training:
            out_s, out_t = torch.split(x, split_size_or_sections=[x.shape[0] // 5, x.shape[0] - x.shape[0] // 5], dim=0)
            x = torch.cat((bn_layers[0](out_s), bn_layers[1](out_t)), dim=0) * gamma + beta
        else:
            x = bn_layers[1](x) * gamma + beta
        return x
    
    def _initialize_weights(self, zero_init_residual, block):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            if zero_init_residual and isinstance(m, block):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self._apply_bn_layers(x, [self.bns1, self.bnt1], self.gamma1, self.beta1)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    return model

def resnet18():
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model