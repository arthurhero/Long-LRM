# Copyright (c) 2024, Ziwen Chen.

import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class PerceptualLoss(nn.Module):
    def __init__(self, device, config):
        super(PerceptualLoss, self).__init__()
        vgg_weigths = config.training.get("perceptual_vgg_weights", "default")
        if vgg_weigths == "default":
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        else:
            vgg = vgg19()
            vgg.load_state_dict(torch.load(vgg_weigths, map_location="cpu"))
        #print(vgg.features)
        # replace the maxpool layer with avgpool
        for i, layer in enumerate(vgg.features):
            if isinstance(layer, nn.MaxPool2d):
                vgg.features[i] = nn.AvgPool2d(kernel_size=2, stride=2)
        self.blocks = nn.ModuleList()
        out_idx = config.training.perceptual_out_idx
        out_idx = [0] + out_idx
        self.layer_weights = config.training.perceptual_out_weights
        assert len(self.layer_weights) == len(out_idx) - 1
        self.feature_scale = config.training.perceptual_feature_scale
        for i in range(len(out_idx)-1):
            self.blocks.append(nn.Sequential(vgg.features[out_idx[i]:out_idx[i+1]]).to(device).eval())
        for param in self.blocks.parameters():
            param.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        #self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))
 
    def forward(self, pred, target):
        """
        pred, target: [B, 3, H, W] in range [0, 1]
        """
        weights = self.layer_weights
        scale = self.feature_scale
        pred = (pred - self.mean) * scale
        target = (target - self.mean) * scale
        loss = torch.mean(torch.abs(pred - target))
        for i_b, block in enumerate(self.blocks):
            pred = block(pred)
            target = block(target)
            loss += torch.mean(torch.abs(pred - target)) * weights[i_b]
        loss = loss / scale
        return loss

# evaluation metrics
from skimage.metrics import structural_similarity
from lpips import LPIPS

@torch.no_grad()
def compute_psnr(predict, target):
    """
    predict, target: (B, C, H, W) in range [0, 1]
    """
    predict = predict.clamp(0, 1)
    target = target.clamp(0, 1)
    mse = torch.mean((predict - target) ** 2, dim=(1, 2, 3)) # (B,)
    psnr = -10 * torch.log10(mse)
    return psnr

@torch.no_grad()
def compute_ssim(predict, target):
    """
    predict, target: (B, C, H, W) in range [0, 1]
    """
    predict = predict.clamp(0, 1)
    target = target.clamp(0, 1)
    ssim = [
        structural_similarity(
            predict[i].cpu().numpy(),
            target[i].cpu().numpy(),
            multichannel=True,
            channel_axis=0,
            data_range=1.0,
        ) for i in range(predict.size(0))
    ]
    ssim = torch.tensor(ssim, device=predict.device, dtype=predict.dtype)
    return ssim

@torch.no_grad()
def compute_lpips(predict, target):
    """
    predict, target: (B, C, H, W) in range [0, 1]
    """
    predict = predict.clamp(0, 1)
    target = target.clamp(0, 1)
    lpips_fn = LPIPS(net="vgg").to(predict.device)
    batch_size = 10
    values = []
    for i in range(0, predict.size(0), batch_size):
        value = lpips_fn.forward(
            predict[i : i + batch_size],
            target[i : i + batch_size],
            normalize=True,
        )
        values.append(value)
    value = torch.cat(values, dim=0)
    value = value[:, 0, 0, 0] # (B,)
    return value
        