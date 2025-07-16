# Copyright (c) 2024, Ziwen Chen.

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict
from einops import rearrange
from gsplat import rasterization

try:
    from .transformer import TransformerBlock
    from .mamba2 import Mamba2Block
    from .loss import PerceptualLoss
except:
    from transformer import TransformerBlock
    from mamba2 import Mamba2Block
    from loss import PerceptualLoss

def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Processor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_layers = config.model.num_layers
        self.dims = config.model.dim

        if config.model.block_type == "transformer":
            self.block_type = ["t"] * self.num_layers
        elif config.model.block_type == "mamba2":
            self.block_type = ["m"] * self.num_layers
        else:
            self.block_type = config.model.block_type
            if len(self.block_type) > self.num_layers:
                self.block_type = self.block_type[:self.num_layers]
            elif len(self.block_type) < self.num_layers:
                self.block_type = self.block_type * (self.num_layers // len(self.block_type)) + self.block_type[:self.num_layers % len(self.block_type)]
        
        self.merge_at = config.model.get("merge_layers", [])
        if isinstance(self.dims, int):
            self.dims = [self.dims]
        assert len(self.dims) == len(self.merge_at) + 1

        self.blocks = nn.ModuleList()
        if len(self.merge_at) > 0:
            self.resize_blocks = nn.ModuleList()
            self.merge_blocks = nn.ModuleList()
        dim_cur = self.dims[0]
        for i, s in enumerate(self.block_type):
            if i in self.merge_at:
                dim_next = self.dims[self.merge_at.index(i) + 1]
                self.resize_blocks.append(nn.Linear(dim_cur, dim_next))
                self.merge_blocks.append(
                    nn.Conv2d(dim_cur, dim_next, kernel_size=2, stride=2, padding=0, bias=True, groups=dim_cur)
                )
                dim_cur = dim_next
            if s == "t":
                self.blocks.append(TransformerBlock(dim_cur, config.model.transformer.head_dim))
                self.blocks[-1].apply(_init_weights)
            elif s == "m":
                self.blocks.append(Mamba2Block(dim_cur, config.model.mamba2.d_state))
            else:
                raise ValueError(f"Invalid block type {s}")

    def run_one_block(self, i):
        def _run_one_block(x, num_global_tokens, v, h, w):
            if i in self.merge_at:
                if num_global_tokens > 0:
                    global_tokens, image_tokens = x[:, :num_global_tokens], x[:, num_global_tokens:]
                    global_tokens = self.resize_blocks[self.merge_at.index(i)](global_tokens)
                else:
                    image_tokens = x
                image_tokens = rearrange(image_tokens, "b (v h w) d -> (b v) d h w", v=v, h=h, w=w)
                image_tokens = self.merge_blocks[self.merge_at.index(i)](image_tokens)
                h = h // 2
                w = w // 2
                image_tokens = rearrange(image_tokens, "(b v) d h w -> b (v h w) d", v=v, h=h, w=w)
                if num_global_tokens > 0:
                    x = torch.cat([global_tokens, image_tokens], dim=1)
                else:
                    x = image_tokens
            x = self.blocks[i](x)
            return x, h, w
        return _run_one_block

    def forward(self, x, num_global_tokens, v, h, w, use_checkpoint=True):
        """
        x: (B, L, D)
        Returns: B and D remain the same, L might change if there are merge layers
        """
        batch, seq_len, _ = x.shape
        num_image_tokens = seq_len - num_global_tokens
        assert num_image_tokens == v * h * w

        for i in range(self.num_layers):
            if use_checkpoint:
                x, h, w = torch.utils.checkpoint.checkpoint(self.run_one_block(i), x, num_global_tokens, v, h, w, use_reentrant=False)
            else:
                x, h, w = self.run_one_block(i)(x, num_global_tokens, v, h, w)

        return x, h, w

class GaussianRenderer(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_c2w, test_intr, 
               W, H, sh_degree, near_plane, far_plane):
        opacity = opacity.sigmoid().squeeze(-1)
        scale = scale.exp()
        rotation = F.normalize(rotation, p=2, dim=-1)
        test_w2c = test_c2w.float().inverse().unsqueeze(0) # (1, 4, 4)
        test_intr_i = torch.zeros(3, 3).to(test_intr.device)
        test_intr_i[0, 0] = test_intr[0]
        test_intr_i[1, 1] = test_intr[1]
        test_intr_i[0, 2] = test_intr[2]
        test_intr_i[1, 2] = test_intr[3]
        test_intr_i[2, 2] = 1
        test_intr_i = test_intr_i.unsqueeze(0) # (1, 3, 3)
        rendering, _, _ = rasterization(xyz, rotation, scale, opacity, feature,
                                        test_w2c, test_intr_i, W, H, sh_degree=sh_degree, 
                                        near_plane=near_plane, far_plane=far_plane,
                                        render_mode="RGB",
                                        backgrounds=torch.ones(1, 3).to(test_intr.device),
                                        rasterize_mode='classic') # (1, H, W, 3) 
        return rendering # (1, H, W, 3)

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
                W, H, sh_degree, near_plane, far_plane):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        with torch.no_grad():
            B, V, _ = test_intr.shape
            renderings = torch.zeros(B, V, H, W, 3).to(xyz.device)
            for ib in range(B):
                for iv in range(V):
                    renderings[ib, iv:iv+1] = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
                                                                      test_c2ws[ib,iv], test_intr[ib,iv], W, H, sh_degree, near_plane, far_plane)
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        sh_degree = ctx.sh_degree
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            B, V, _ = test_intr.shape
            for ib in range(B):
                for iv in range(V):
                    rendering = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib], 
                                                        test_c2ws[ib,iv], test_intr[ib,iv], W, H, sh_degree, near_plane, far_plane)
                    rendering.backward(grad_output[ib, iv:iv+1])

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None
            
class LongLRM(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        input_dim = 9 # RGB + plucker ray
        self.patch_size = config.model.patch_size
        self.patch_size_out = self.patch_size * 2 ** len(config.model.get("merge_layers", [])) 
        if isinstance(config.model.dim, int):
            self.dim_start = config.model.dim
            self.dim_out = config.model.dim
        else:
            self.dim_start = config.model.dim[0]
            self.dim_out = config.model.dim[-1]
        self.num_global_tokens = config.model.num_global_tokens
        if self.num_global_tokens > 0:
            self.global_token_init = nn.Parameter(torch.randn(1, self.num_global_tokens, self.dim_start))
            nn.init.trunc_normal_(self.global_token_init, std=0.02)
        self.tokenizer = nn.Linear(input_dim * self.patch_size ** 2, self.dim_start, bias=False)
        self.tokenizer.apply(_init_weights)
        self.input_layernorm = nn.LayerNorm(self.dim_start, bias=False)
        self.processor = Processor(config)
        self.tokenDecoder = nn.Sequential(
            nn.LayerNorm(self.dim_out, bias=False),
            nn.Linear(
                self.dim_out, (3 + (config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1) * self.patch_size_out ** 2,
                bias=False,
            )
        )
        self.tokenDecoder.apply(_init_weights)

        if config.training.get("perceptual_loss", 0.0) > 0:
            self.perceptual_loss = PerceptualLoss(device, config)

        # use DepthAnything for depth loss
        if config.training.get("gaussian_depth_loss", 0.0) > 0:
            try:
                from model.depth_anything.dpt import DepthAnything
            except:
                from depth_anything.dpt import DepthAnything
            model_configs = {
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
            }
            encoder = 'vits' # or 'vitb', 'vits'
            self.depth_anything = DepthAnything(model_configs[encoder])
            self.depth_anything.load_state_dict(torch.load(f'model/depth_anything/depth_anything_{encoder}14.pth'))

            for param in self.depth_anything.parameters():
                param.requires_grad = False

    def render(self, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, W, H):
        B, V, _ = test_intr.shape
        renderings = []
        for i in range(B):
            xyz_i = xyz[i]
            feature_i = feature[i]
            scale_i = scale[i]
            scale_i = scale_i.exp()
            rotation_i = rotation[i]
            rotation_i = F.normalize(rotation_i, p=2, dim=-1)
            opacity_i = opacity[i]
            opacity_i = opacity_i.sigmoid().squeeze(-1)
            test_w2c_i = test_c2ws[i].float().inverse() # (V, 4, 4)
            test_intr_i = torch.zeros(V, 3, 3).to(input_intr.device)
            test_intr_i[:, 0, 0] = test_intr[i, :, 0]
            test_intr_i[:, 1, 1] = test_intr[i, :, 1]
            test_intr_i[:, 0, 2] = test_intr[i, :, 2]
            test_intr_i[:, 1, 2] = test_intr[i, :, 3]
            test_intr_i[:, 2, 2] = 1
            rendering, _, _ = rasterization(xyz_i, rotation_i, scale_i, opacity_i, feature_i,
                                            test_w2c_i, test_intr_i, W, H, sh_degree=self.config.model.gaussians.sh_degree, 
                                            near_plane=self.config.model.gaussians.near_plane, far_plane=self.config.model.gaussians.far_plane,
                                            render_mode="RGB",
                                            backgrounds=torch.ones(V, 3).to(input_images.device),
                                            rasterize_mode='classic') # (V, H, W, 3) 
            renderings.append(rendering)
        return torch.stack(renderings, dim=0) # (B, V, H, W, 3)

    def forward(self, input_dict):
        """
        input_images: (B, V, 3, H, W)
        input_intr: (B, V, 4), (fx, fy, cx, cy)
        input_c2ws: (B, V, 4, 4)
        pos_avg_inv: (B, 4, 4)
        scene_scale: (B)
        """
        input_dict = edict(input_dict)
        input_images = input_dict["input_images"]
        input_intr = input_dict["input_intr"]
        input_c2ws = input_dict["input_c2ws"]
        test_images = input_dict.get("test_images", None)
        test_intr = input_dict.get("test_intr", None)
        test_c2ws = input_dict.get("test_c2ws", None)
        pos_avg_inv = input_dict.get("pos_avg_inv", None)
        scene_scale = input_dict.get("scene_scale", None)
        scene_name = input_dict.get("scene_name", None)
        use_checkpoint = input_dict.get("use_checkpoint", True)


        inference_start = time.time()
        B, V, _, H, W = input_images.shape

        # Embed camera info
        ray_o = input_c2ws[:, :, :3, 3].unsqueeze(2).expand(-1, -1, H * W, -1).float() # (B, V, H*W, 3) # camera origin
        x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        x = (x.to(input_intr.dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(input_images.device).contiguous()
        y = (y.to(input_intr.dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(input_images.device).contiguous()
        # unproject to camera space
        x = (x - input_intr[:, :, 2:3]) / input_intr[:, :, 0:1]
        y = (y - input_intr[:, :, 3:4]) / input_intr[:, :, 1:2]
        ray_d = torch.stack([x, y, torch.ones_like(x)], dim=-1).float() # (B, V, H*W, 3)
        ray_d = F.normalize(ray_d, p=2, dim=-1)
        ray_d = ray_d @ input_c2ws[:, :, :3, :3].transpose(-1, -2).contiguous() # (B, V, H*W, 3)

        input_image_cam = torch.cat([input_images.view(B, V, 3, -1).permute(0, 1, 3, 2).contiguous() * 2 - 1, 
                                     torch.cross(ray_o, ray_d, dim=-1),
                                     ray_d], dim=-1) # (B, V, H*W, 9)

        # Pachify
        patch_size = self.patch_size
        hh = H // patch_size
        ww = W // patch_size
        input_image_cam = rearrange(input_image_cam, 
                                    "b v (hh ph ww pw) d -> b (v hh ww) (ph pw d)", 
                                    hh=hh, ww=ww, ph=patch_size, pw=patch_size)

        # Tokenize the input images
        image_tokens = self.tokenizer(input_image_cam) # (B, V*hh*ww, D)
        if self.num_global_tokens > 0:
            global_tokens = self.global_token_init.expand(B, -1, -1)
            tokens = torch.cat([global_tokens, image_tokens], dim=1) # (B, num_global_tokens+V*hh*ww, D)
        else:
            tokens = image_tokens
        tokens = self.input_layernorm(tokens)

        # Process tokens
        tokens, hh, ww = self.processor(tokens, self.num_global_tokens, V, hh, ww, use_checkpoint=use_checkpoint)
        patch_size = self.patch_size_out

        # Decode tokens
        image_tokens = tokens[:, self.num_global_tokens:] # (B, V*hh*ww, D)
        gaussians = self.tokenDecoder(image_tokens) # (B, V*hh*ww, ph*pw*(3 + (sh_degree+1)**2*3 + 3 + 4 + 1))
        gaussians = rearrange(gaussians, "b (v hh ww) (ph pw d) -> b (v hh ph ww pw) d", v=V, hh=hh, ww=ww, ph=patch_size, pw=patch_size)
        xyz, feature, scale, rotation, opacity = torch.split(gaussians, [3, (self.config.model.gaussians.sh_degree + 1) ** 2 * 3, 3, 4, 1], dim=-1)
        feature = feature.view(B, V*H*W, (self.config.model.gaussians.sh_degree + 1) ** 2, 3).contiguous()
        scale = (scale + self.config.model.gaussians.scale_bias).clamp(max = self.config.model.gaussians.scale_max) 
        opacity = opacity + self.config.model.gaussians.opacity_bias
        
        # Align gaussian means to pixel centers
        if self.config.model.gaussians.get("align_to_pixel", True):
            dist = xyz.mean(dim=-1, keepdim=True).sigmoid() * self.config.model.gaussians.max_dist # (B, V*H*W, 1)
            xyz = dist * ray_d.reshape(B, -1, 3) + ray_o.reshape(B, -1, 3) # (B, V*H*W, 3)
            # get pixel-aligned depth before pruning
            if self.config.training.get("gaussian_depth_loss", 0.0) > 0:
                pos = xyz.reshape(B, V, H*W, 3)
                input_w2cs = input_c2ws.float().inverse() # (B, V, 4, 4)
                pos_cam = pos @ input_w2cs[:, :, :3, :3].transpose(-1, -2).contiguous() + input_w2cs[:, :, :3, 3:4].transpose(-1,-2).contiguous() # (B, V, H*W, 3)
                depth_pred = pos_cam[..., 2] # (B, V, H*W)
                disp_pred = 1.0 / depth_pred.clamp(min=1e-2)
                disp_median = torch.median(disp_pred, dim=-1, keepdim=True)[0] # (B, V, 1)
                disp_var = (disp_pred - disp_median).abs().mean(dim=-1, keepdim=True) # (B, V, 1)
                disp_pred = (disp_pred - disp_median) / (disp_var + 1e-6)

        gaussians = {
            "xyz": xyz.float(),
            "feature": feature.float(),
            "scale": scale.float(),
            "rotation": rotation.float(),
            "opacity": opacity.float()
        }
        inference_time = time.time() - inference_start

        # GS Pruning
        num_gaussians = xyz.shape[1]
        prune_ratio = self.config.model.gaussians.get("prune_ratio", 0.0)
        gaussian_usage = (opacity.sigmoid() > self.config.model.gaussians.get("opacity_threshold", 0.001)).float().mean(dim=1).squeeze(-1) # (B,)
        if prune_ratio > 0:
            keep_ratio = 1 - prune_ratio
            random_ratio = self.config.model.gaussians.get("random_ratio", 0.0)
            random_ratio = keep_ratio * random_ratio
            keep_ratio = keep_ratio - random_ratio
            num_keep = int(num_gaussians * keep_ratio)
            num_keep_random = int(num_gaussians * random_ratio)
            # rank by opacity
            idx_sort = opacity.argsort(dim=1, descending=True)
            keep_idx = idx_sort[:, :num_keep]
            if num_keep_random > 0:
                rest_idx = idx_sort[:, num_keep:]
                random_idx = rest_idx[:, torch.randperm(rest_idx.shape[1])[:num_keep_random]]
                keep_idx = torch.cat([keep_idx, random_idx], dim=1)
            for k, v in gaussians.items():
                v_shape = v.shape
                v = v.reshape(v_shape[0], v_shape[1], -1)
                v = v.gather(1, keep_idx.expand(-1, -1, v.shape[-1]))
                gaussians[k] = v.reshape(v_shape[0], -1, *v_shape[2:])

        ret_dict = {
            "gaussians": gaussians,
            "gaussian_usage": gaussian_usage,
            "inference_time": inference_time
        }

        if pos_avg_inv is not None:
            ret_dict["pos_avg_inv"] = pos_avg_inv

        if scene_scale is not None:
            ret_dict["scene_scale"] = scene_scale

        if scene_name is not None:
            ret_dict["scene_name"] = scene_name

        if test_c2ws is None:
            return ret_dict

        # Render images at test views
        xyz = gaussians["xyz"]
        feature = gaussians["feature"]
        scale = gaussians["scale"]
        rotation = gaussians["rotation"]
        opacity = gaussians["opacity"]
        with torch.autocast(enabled=False, device_type="cuda"):
            if use_checkpoint:
                # cannot simply use torch checkpoint as memory reduction relies on the loop through views
                renderings = GaussianRenderer.apply(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, W, H, 
                                                    self.config.model.gaussians.sh_degree, self.config.model.gaussians.near_plane,
                                                    self.config.model.gaussians.far_plane)
            else:
                renderings = self.render(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, W, H) # (B, V, H, W, 3)
        renderings = renderings.permute(0, 1, 4, 2, 3).contiguous() # (B, V, 3, H, W)
        ret_dict["renderings"] = renderings

        if test_images is None:
            return ret_dict

        if not self.training:
            return ret_dict

        # Compute loss
        renderings = renderings.flatten(0, 1) # (B*V, 3, H, W)
        test_images = test_images.flatten(0, 1) # (B*V, 3, H, W)
        l2_loss = F.mse_loss(renderings, test_images)
        psnr = -10 * torch.log10(l2_loss)
        total_loss = l2_loss * self.config.training.get("l2_loss", 1.0)
        loss_dict = {
            "l2_loss": l2_loss,
            "psnr": psnr,
        }
        if self.config.training.get("perceptual_loss", 0.0) > 0:
            perceptual_loss = self.perceptual_loss(renderings, test_images)
            total_loss += perceptual_loss * self.config.training.perceptual_loss
            loss_dict["perceptual_loss"] = perceptual_loss
        if self.config.training.get("opacity_loss", 0.0) > 0:
            opacity_loss = opacity.sigmoid().mean()
            total_loss += opacity_loss * self.config.training.opacity_loss
            loss_dict["opacity_loss"] = opacity_loss
        if self.config.model.gaussians.get("align_to_pixel", True) and self.config.training.get("gaussian_depth_loss", 0.0) > 0:
            H_ = (H // 14) * 14
            W_ = (W // 14) * 14
            input_images_ = nn.functional.interpolate(input_images.flatten(0, 1), (H_, W_), mode='bilinear') # (B*V, 3, H_, W_)
            with torch.no_grad():
                self.depth_anything.eval()
                disp_da = self.depth_anything(input_images_).reshape(B, V, H_, W_) # (B, V, H_, W_)
            disp_da = nn.functional.interpolate(disp_da, (H, W), mode='nearest').to(disp_pred.dtype).reshape(B, V, H*W) # (B, V, H*W)
            disp_median = torch.median(disp_da, dim=-1, keepdim=True)[0] # (B, V, 1)
            disp_var = (disp_da - disp_median).abs().mean(dim=-1, keepdim=True) # (B, V, 1)
            disp_da = (disp_da - disp_median) / (disp_var + 1e-6)

            gaussian_depth_loss = F.smooth_l1_loss(disp_pred, disp_da)
            total_loss += gaussian_depth_loss * self.config.training.gaussian_depth_loss
            loss_dict["gaussian_depth_loss"] = gaussian_depth_loss

            ret_dict["disp_da"] = disp_da.reshape(B, V, H, W)
            ret_dict["disp_pred"] = disp_pred.reshape(B, V, H, W)
        
        loss_dict["total_loss"] = total_loss
        ret_dict["loss"] = loss_dict

        return ret_dict

    def save_gaussian_ply(self, gaussian_dict, save_path, opacity_threshold=None):
        """
        Adapted from the original 3D GS implementation
        https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py
        """
        from plyfile import PlyData, PlyElement
        xyz = gaussian_dict["xyz"].detach().cpu().float() # (N, 3)
        normal = torch.zeros_like(xyz) # (N, 3)
        N = xyz.shape[0]
        feature = gaussian_dict["feature"].detach().cpu().float() # (N, (sh_degree+1)**2, 3)
        f_dc = feature[:, 0].contiguous() # (N, 3)
        f_rest_full = torch.zeros(N, 3*(3+1)**2-3).float()
        if feature.shape[1] > 1:
            f_rest = feature[:, 1:].transpose(1, 2).reshape(N, -1) # (N, 3*(sh_degree+1)**2-3)
            f_rest_full[:, :f_rest.shape[1]] = f_rest
        f_rest_full = f_rest_full.contiguous()
        scale = gaussian_dict["scale"].detach().cpu().float() # (N, 3)
        opacity = gaussian_dict["opacity"].detach().cpu().float() # (N, 1)
        rotation = gaussian_dict["rotation"].detach().cpu().float() # (N, 4)
        attributes = np.concatenate([xyz.numpy(), 
                                     normal.numpy().astype(np.uint8),
                                     f_dc.numpy(),
                                     f_rest_full.numpy(),
                                     opacity.numpy(),
                                     scale.numpy(),
                                     rotation.numpy()
                                    ], axis=1)
        if opacity_threshold is not None:                             
            attributes = attributes[opacity.squeeze(-1).sigmoid().numpy() > opacity_threshold]
        attribute_list = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        attribute_list += ['f_dc_{}'.format(i) for i in range(f_dc.shape[1])]
        attribute_list += ['f_rest_{}'.format(i) for i in range(f_rest_full.shape[1])]
        attribute_list += ['opacity']
        attribute_list += ['scale_{}'.format(i) for i in range(scale.shape[1])]
        attribute_list += ['rot_{}'.format(i) for i in range(rotation.shape[1])]
        dtype_full = [(attribute, 'f4') for attribute in attribute_list]
        dtype_full[3:6] = [(attribute, 'u1') for attribute in attribute_list[3:6]]
        elements = np.empty(attributes.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(save_path)

    def save_input_video(self, input_intr, input_c2ws, gaussian_dict, H, W, save_path, insert_frame_num = 8):
        """
        Interpolate input frames and save rendered video
        input_intr: (V, 4), (fx, fy, cx, cy)
        input_c2ws: (V, 4, 4)
        """
        import cv2
        from .camera_utils import get_interpolated_poses_many
        import subprocess
        V = input_intr.shape[0]
        device = input_intr.device
        input_intr = input_intr.detach().cpu().float()
        input_c2ws = input_c2ws.detach().cpu().float()

        input_intr_mat = torch.zeros((V, 3, 3))
        input_intr_mat[:, 0, 0] = input_intr[:, 0]
        input_intr_mat[:, 1, 1] = input_intr[:, 1]
        input_intr_mat[:, 0, 2] = input_intr[:, 2]
        input_intr_mat[:, 1, 2] = input_intr[:, 3]
        input_c2ws = torch.cat([input_c2ws, input_c2ws[:1]], dim=0) # wrap around
        input_intr_mat = torch.cat([input_intr_mat, input_intr_mat[:1]], dim=0) # wrap around
        c2ws, intr_mat, _ = get_interpolated_poses_many(input_c2ws[:, :3, :4], input_intr_mat, steps_per_transition = insert_frame_num)
        V = c2ws.shape[0]
        c2ws_mat = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)
        c2ws_mat[:, :3, :4] = c2ws
        intr_fxfycxcy = torch.zeros(V, 4)
        intr_fxfycxcy[:, 0] = intr_mat[:, 0, 0]
        intr_fxfycxcy[:, 1] = intr_mat[:, 1, 1]
        intr_fxfycxcy[:, 2] = intr_mat[:, 0, 2]
        intr_fxfycxcy[:, 3] = intr_mat[:, 1, 2]
        c2ws_mat = c2ws_mat.to(device)
        intr_fxfycxcy = intr_fxfycxcy.to(device)

        xyz = gaussian_dict["xyz"].detach().float().to(device) # (N, 3)
        feature = gaussian_dict["feature"].detach().float().to(device) # (N, (sh_degree+1)**2, 3)
        scale = gaussian_dict["scale"].detach().float().to(device) # (N, 3)
        rotation = gaussian_dict["rotation"].detach().float().to(device) # (N, 4)
        opacity = gaussian_dict["opacity"].detach().float().to(device) # (N, 1)

        renderings = []
        with torch.autocast(enabled=False, device_type="cuda"):
            for i in range(V):
                rendering = GaussianRenderer.render(xyz, feature, scale, rotation, opacity,
                                                    c2ws_mat[i], intr_fxfycxcy[i], W, H,
                                                    self.config.model.gaussians.sh_degree,
                                                    self.config.model.gaussians.near_plane,
                                                    self.config.model.gaussians.far_plane)
                rendering = rendering.squeeze(0).clamp(0, 1).cpu().numpy() # (H, W, 3)
                rendering = (rendering * 255).astype(np.uint8)
                rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
                renderings.append(rendering)
        tmp_save_path = save_path.replace(".mp4", "_tmp.mp4")
        video_writer = cv2.VideoWriter(tmp_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
        for r in renderings:
            video_writer.write(r)
        video_writer.release()
        subprocess.run(f"ffmpeg -y -i {tmp_save_path} -vcodec libx264 -f mp4 {save_path} -loglevel quiet", shell=True) 
        os.remove(tmp_save_path)

    def save_visualization(self, input_dict, output_dict, save_dir, save_gaussian=False, save_video=False):
        import torchvision
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)

        input_images = input_dict["input_images"] # (B, V, 3, H, W)
        target_images = input_dict["test_images"] # (B, V, 3, H, W)
        renderings = output_dict["renderings"] # (B, V, 3, H, W)
        B, V, _, H, W = target_images.shape

        # save images of first batch
        input_image_path = os.path.join(save_dir, "input_images.png")
        input_image = input_images[0].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, Vin*W)
        torchvision.utils.save_image(input_image, input_image_path)
        target_rendering_path = os.path.join(save_dir, "target_renderings.png")
        target_renderings = []
        for i in range(B):
            target_image = target_images[i].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, V*W)
            rendering_image = renderings[i].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, V*W)
            target_renderings.append(target_image)
            target_renderings.append(rendering_image)
        target_rendering = torch.cat(target_renderings, dim=1) # (3, 2*B*H, V*W)
        torchvision.utils.save_image(target_rendering, target_rendering_path)

        if "disp_da" in output_dict:
            cmapper = plt.colormaps['magma']
            disp_da = output_dict["disp_da"][0].detach().permute(1, 0, 2).flatten(1, 2) # (V, H, W) -> (H, V*W)
            disp_pred = output_dict["disp_pred"][0].detach().permute(1, 0, 2).flatten(1, 2) # (V, H, W) -> (H, V*W)
            disp_da = (1.0 / (disp_da + 3.0).clamp(min=0.001)) / 1.0
            disp_pred = (1.0 / (disp_pred + 3.0).clamp(min=0.001)) / 1.0
            disp_da = cmapper(disp_da.cpu().numpy())[..., :3]
            disp_pred = cmapper(disp_pred.cpu().numpy())[..., :3]
            disp_image = torch.cat([torch.tensor(disp_da), torch.tensor(disp_pred)], dim=0).permute(2, 0, 1) # (3, 2*H, V*W)
            disp_path = os.path.join(save_dir, "disp.png")
            torchvision.utils.save_image(disp_image, disp_path)

        # save gaussian ply of first batch
        if save_gaussian:
            gaussians = output_dict["gaussians"]
            gaussian_first = {k: v[0] for k, v in gaussians.items()}
            opacity_threshold = self.config.model.gaussians.get("opacity_threshold", 0.001)
            self.save_gaussian_ply(gaussian_first, os.path.join(save_dir, f"gaussians_{str(opacity_threshold).split('.')[-1]}.ply"), opacity_threshold)

        # save input traj video
        if save_video:
            gaussians = output_dict["gaussians"]
            input_intr = input_dict["input_intr"][0]
            input_c2ws = input_dict["input_c2ws"][0]
            gaussian_first = {k: v[0] for k, v in gaussians.items()}
            self.save_input_video(input_intr, input_c2ws, gaussian_first, H, W, os.path.join(save_dir, "input_traj.mp4"),
                                  insert_frame_num=self.config.get("insert_frame_num", 8))

    def save_evaluation_results(self, input_dict, output_dict, save_dir):
        import torchvision
        from .loss import compute_psnr, compute_ssim, compute_lpips

        input_images = input_dict["input_images"] # (B, Vin, 3, H, W)
        target_images = input_dict["test_images"] # (B, V, 3, H, W)
        renderings = output_dict["renderings"] # (B, V, H, W, 3)
        gaussians = output_dict["gaussians"]
        gaussian_usage = output_dict["gaussian_usage"] # (B,)
        inference_time = output_dict["inference_time"] # float
        B, V, _, H, W = target_images.shape

        for i in range(B):
            scene_name = input_dict["scene_name"][i]
            scene_dir = os.path.join(save_dir, scene_name)
            os.makedirs(scene_dir, exist_ok=True)

            # evaluation metrics
            psnr = compute_psnr(renderings[i], target_images[i]) # (V,)
            ssim = compute_ssim(renderings[i], target_images[i]) # (V,)
            lpips = compute_lpips(renderings[i], target_images[i]) # (V,)
            with open(os.path.join(scene_dir, "metrics.csv"), "w") as f:
                f.write("index, psnr, ssim, lpips\n")
                for j in range(V):
                    f.write(f"{j}, {psnr[j].item()}, {ssim[j].item()}, {lpips[j].item()}\n")
                f.write(f"mean, {psnr.mean().item()}, {ssim.mean().item()}, {lpips.mean().item()}\n")
                f.write(f"gaussian_usage, {gaussian_usage[i].item()}\n")
                f.write(f"inference_time, {inference_time}\n") 
                f.close()

            # save images
            input_images_path = os.path.join(scene_dir, "input_images.png")
            input_image = input_images[i].permute(1, 2, 0, 3).flatten(2, 3) # (3, H, Vin*W)
            torchvision.utils.save_image(input_image, input_images_path)
            os.makedirs(os.path.join(scene_dir, "target"), exist_ok=True)
            os.makedirs(os.path.join(scene_dir, "rendering"), exist_ok=True)
            for j in range(V):
                target_path = os.path.join(scene_dir, "target", f"{j}.png")
                rendering_path = os.path.join(scene_dir, "rendering", f"{j}.png")
                torchvision.utils.save_image(target_images[i, j], target_path)
                torchvision.utils.save_image(renderings[i, j], rendering_path)

            # save gaussian ply
            gaussian = {k: v[i] for k, v in gaussians.items()}
            opacity_threshold = self.config.model.gaussians.get("opacity_threshold", 0.001)
            self.save_gaussian_ply(gaussian, os.path.join(scene_dir, f"gaussians_{str(opacity_threshold).split('.')[-1]}.ply"), opacity_threshold)

            # save pose normalization info
            if "pos_avg_inv" in output_dict:
                pos_avg_inv = input_dict["pos_avg_inv"][i] # (4, 4)
                pos_avg_inv_path = os.path.join(scene_dir, "pos_avg_inv.txt")
                np.savetxt(pos_avg_inv_path, pos_avg_inv.cpu().numpy())
            if "scene_scale" in output_dict:
                scene_scale = input_dict["scene_scale"][i]
                scene_scale_path = os.path.join(scene_dir, "scene_scale.txt")
                np.savetxt(scene_scale_path, np.array([scene_scale.item()])) 
            if "input_frame_idx" in input_dict:
                input_frame_idx = input_dict["input_frame_idx"][i] # (V,)
                input_frame_idx_path = os.path.join(scene_dir, "input_frame_idx.txt")
                np.savetxt(input_frame_idx_path, input_frame_idx.cpu().numpy().astype(int), fmt='%i')
            if "test_frame_idx" in input_dict:
                target_frame_idx = input_dict["test_frame_idx"][i]
                target_frame_idx_path = os.path.join(scene_dir, "target_frame_idx.txt")
                np.savetxt(target_frame_idx_path, target_frame_idx.cpu().numpy().astype(int), fmt='%i')

            # save input traj video
            input_intr = input_dict["input_intr"][i]
            input_c2ws = input_dict["input_c2ws"][i]
            self.save_input_video(input_intr, input_c2ws, gaussian, H, W, os.path.join(scene_dir, "input_traj.mp4"),
                                  insert_frame_num=self.config.get("insert_frame_num", 8))

if __name__ == "__main__":

    """
    # Test Processor
    batch_size = 128
    v = 2
    h = 16
    w = 16
    num_global_tokens = 2
    input_dim = 256

    config = edict({
        "model": {
            "dim": [256, 512, 1024],
            "num_layers": 4,
            "block_type": "tmtm",
            "merge_layers": [1,2],
            "transformer": {
                "head_dim": 32
            },
            "mamba2": {
                "d_state": 256
            }
        }
    })

    model = Processor(config).to("cuda").to(torch.float16)
    x = torch.randn(batch_size, num_global_tokens + v * h * w, input_dim).to("cuda").to(torch.float16)
    x = nn.Parameter(x)
    x.retain_grad()
    out, new_h, new_w = model(x, num_global_tokens, v, h, w, use_checkpoint=True)

    print("Input shape:", x.shape, h, w)
    print("Output shape:", out.shape, new_h, new_w)

    out.sum().backward()
    grad = x.grad.clone()
    x.grad.zero_()
    print("Peak memory with checkpoint:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

    out, new_h, new_w = model(x, num_global_tokens, v, h, w, use_checkpoint=False)
    out.sum().backward()
    grad2 = x.grad.clone()
    print("Peak memory without checkpoint:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

    diff = (grad - grad2).abs()
    print("Max grad diff:", diff.max())
    print("Norm grad diff:", diff.norm())
    """

    # Test LongLRM

    batch_size = 2
    view = 4
    h = 256
    w = 256
    patch_size = 8
    num_global_tokens = 2
    input_dim = 256
    config = edict({
        "data": {
            "patch_size": patch_size
        },
        "model": {
            "dim": [256, 512, 1024],
            "patch_size": patch_size,
            "num_layers": 4,
            "block_type": "tmtm",
            "merge_layers": [1,2],
            "transformer": {
                "head_dim": 32
            },
            "mamba2": {
                "d_state": 256
            },
            "num_global_tokens": 2,
            "gaussians": {
                "sh_degree": 2,
                "max_dist": 1,
                "scale_bias": 0,
                "scale_max": 10,
                "opacity_bias": 0,
                "near_plane": 0.1,
                "far_plane": 100,
                "prune_ratio": 0.5,
                "random_ratio": 0.0,
            },
        },
        "training": {
            "l2_loss": 1.0,
            "perceptual_loss": 0.1,
            "opacity_loss": 0.1,
            "gaussian_depth_loss": 0.0,
            "perceptual_out_idx": [4],
            "perceptual_out_weights": [1.0],
            "perceptual_feature_scale": 1.0,
        }
    })
    device = "cuda"
    llrm = LongLRM(config, device).to("cuda")
    input_images = torch.randn(batch_size, view, 3, h, w).to("cuda")
    input_images = nn.Parameter(input_images)
    input_images.retain_grad()
    input_intr = torch.zeros(batch_size, view, 4).to("cuda")
    input_intr[:, :, 0] = 1
    input_intr[:, :, 1] = 1
    input_intr[:, :, 2] = w//2
    input_intr[:, :, 3] = h//2
    input_c2ws = torch.eye(4).unsqueeze(0).expand(batch_size, view, 4, 4).to("cuda")
    pos_avg_inv = torch.eye(4).unsqueeze(0).expand(batch_size, 4, 4).to("cuda")
    scene_scale = torch.ones(batch_size).to("cuda")

    view_test = 256
    test_intr = torch.zeros(batch_size, view_test, 4).to("cuda")
    test_intr[:, :, 0] = 1
    test_intr[:, :, 1] = 1
    test_intr[:, :, 2] = w//2
    test_intr[:, :, 3] = h//2
    test_c2ws = torch.eye(4).unsqueeze(0).expand(batch_size, view_test, 4, 4).to("cuda")
    test_images = torch.ones(batch_size, view_test, 3, h, w).to("cuda")

    input_dict = {
        "input_images": input_images,
        "input_intr": input_intr,
        "input_c2ws": input_c2ws,
        "test_images": test_images,
        "test_intr": test_intr,
        "test_c2ws": test_c2ws,
        "checkpoints": True,
    }

    with torch.autocast(enabled=True, device_type="cuda"):
        ret_dict = llrm(input_dict)
    gaussians = ret_dict["gaussians"]
    renderings = ret_dict["renderings"]
    #renderings.mean().backward()
    render = renderings.clone()
    loss_dict = ret_dict["loss"]
    loss_total = loss_dict["total_loss"]
    loss_total.backward()
    print("Peak memory with checkpoint:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
    print("Renderings shape:", renderings.shape)
    print("input shape:", input_images.shape)
    print("num_gaussians:", gaussians["xyz"].shape[1])
    print("loss:", loss_dict)

    grad = input_images.grad.clone()
    input_images.grad.zero_()
    input_dict = {
        "input_images": input_images,
        "input_intr": input_intr,
        "input_c2ws": input_c2ws,
        "test_images": test_images,
        "test_intr": test_intr,
        "test_c2ws": test_c2ws,
        "checkpoints": False,
    }
    with torch.autocast(enabled=True, device_type="cuda"):
        ret_dict = llrm(input_dict)
    render2 = ret_dict["renderings"].clone()
    loss_dict2 = ret_dict["loss"]
    loss_total2 = loss_dict2["total_loss"]
    loss_total2.backward() 
    grad2 = input_images.grad.clone()
    print("Peak memory without checkpoint:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

    diff_render = (render - render2).abs()
    print("Max render diff:", diff_render.max())
    print("Norm render diff:", diff_render.norm())

    diff = (grad - grad2).abs()
    print("Max grad diff:", diff.max())
    print("Norm grad diff:", diff.norm())
    print("loss2:", loss_dict2)
