# Copyright (c) 2024, Ziwen Chen.

import torch
import torch.nn as nn
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self, device):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        #print(vgg.features)
        self.blocks = nn.ModuleList()
        out_idx = [0, 2, 7, 12, 21, 30]
        for i in range(len(out_idx)-1):
            self.blocks.append(nn.Sequential(vgg.features[out_idx[i]:out_idx[i+1]]).to(device).eval())
        for param in self.blocks.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
 
    def forward(self, pred, target):
        """
        pred, target: [B, 3, H, W] in range [0, 1]
        """
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std
        loss = torch.mean(torch.abs(pred - target))
        for block in self.blocks:
            pred = block(pred)
            target = block(target)
            loss += torch.mean(torch.abs(pred - target))
        return loss
        