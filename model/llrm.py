# Copyright (c) 2024, Ziwen Chen.

import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict
from einops import rearrange
from gsplat import rasterization

try:
    from .transformer import TransformerBlock
except:
    from transformer import TransformerBlock

try:
    from .mamba2 import Mamba2Block
except:
    from mamba2 import Mamba2Block

def _init_weights(self, m):
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
                self.blocks.append(TransformerBlock(dim_cur, config.model.transformer.num_heads))
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
    def render(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, 
               W, H, sh_degree, near_plane, far_plane):
        test_w2c = test_c2ws.float().inverse().unsqueeze(0) # (1, 4, 4)
        test_intr_i = torch.zeros(3, 3).to(input_intr.device)
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
                                        backgrounds=torch.ones(1, 3).to(input_images.device),
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
    def __init__(self, config):
        super().__init__()
        self.config = config
        input_dim = 9 # RGB + plucker ray
        self.patch_size = config.data.patch_size
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
        self.tokenizer = nn.Linear(input_dim * self.patch_size ** 2, self.dim_start)
        self.processor = Processor(config)
        self.tokenDecoder = nn.Sequential(
            nn.LayerNorm(self.dim_out),
            nn.Linear(
                self.dim_out, (3 + (config.model.gaussians.sh_degree + 1) ** 2 * 3 + 3 + 4 + 1) * self.patch_size_out ** 2,
                bias=False,
            )
        )

    def render(self, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, W, H):
        B, V, _ = test_intr.shape
        renderings = []
        for i in range(B):
            xyz_i = xyz[i]
            feature_i = feature[i]
            scale_i = scale[i]
            rotation_i = rotation[i]
            opacity_i = opacity[i]
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

    def forward(self, 
                input_images,
                input_intr,
                input_c2ws,
                test_images=None,
                test_intr=None,
                test_c2ws=None,
                use_checkpoint=True,
               ):
        """
        input_images: (B, V, 3, H, W)
        input_intr: (B, V, 4), (fx, fy, cx, cy)
        input_c2ws: (B, V, 4, 4)
        """
        B, V, _, H, W = input_images.shape

        # Embed camera info
        ray_o = input_c2ws[:, :, :3, 3].unsqueeze(2).expand(-1, -1, H * W, -1) # (B, V, H*W, 3) # camera origin
        x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        x = (x.to(input_intr.dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(input_images.device)
        y = (y.to(input_intr.dtype) + 0.5).view(1, 1, -1).expand(B, V, -1).to(input_images.device)
        # unproject to camera space
        x = (x - input_intr[:, :, 2:3]) / input_intr[:, :, 0:1]
        y = (y - input_intr[:, :, 3:4]) / input_intr[:, :, 1:2]
        ray_d = torch.stack([x, y, torch.ones_like(x)], dim=-1) # (B, V, H*W, 3)
        ray_d = F.normalize(ray_d, p=2, dim=-1)
        ray_d = ray_d @ input_c2ws[:, :, :3, :3].transpose(-1, -2) # (B, V, H*W, 3)

        input_image_cam = torch.cat([input_images.view(B, V, 3, -1).permute(0, 1, 3, 2), 
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

        # Process tokens
        tokens, hh, ww = self.processor(tokens, self.num_global_tokens, V, hh, ww, use_checkpoint=use_checkpoint)
        patch_size = self.patch_size_out

        # Decode tokens
        image_tokens = tokens[:, self.num_global_tokens:] # (B, V*hh*ww, D)
        gaussians = self.tokenDecoder(image_tokens) # (B, V*hh*ww, ph*pw*(3 + (sh_degree+1)**2*3 + 3 + 4 + 1))
        gaussians = rearrange(gaussians, "b (v hh ww) (ph pw d) -> b (v hh ph ww pw) d", v=V, hh=hh, ww=ww, ph=patch_size, pw=patch_size)
        xyz, feature, scale, rotation, opacity = torch.split(gaussians, [3, (self.config.model.gaussians.sh_degree + 1) ** 2 * 3, 3, 4, 1], dim=-1)
        feature = feature.view(B, V*H*W, (self.config.model.gaussians.sh_degree + 1) ** 2, 3)
        scale = (scale + self.config.model.gaussians.scale_bias).clamp(max = self.config.model.gaussians.scale_max) 
        opacity = opacity + self.config.model.gaussians.opacity_bias
        
        # Align gaussian means to pixel centers
        if self.config.model.gaussians.get("align_to_pixel", True):
            dist = xyz.mean(dim=-1, keepdim=True).sigmoid() * self.config.model.gaussians.max_dist # (B, V*H*W, 1)
            xyz = dist * ray_d.reshape(B, -1, 3) + ray_o.reshape(B, -1, 3) # (B, V*H*W, 3)

        gaussians = {
            "xyz": xyz,
            "feature": feature,
            "scale": scale,
            "rotation": rotation,
            "opacity": opacity
        }

        if test_c2ws is None:
            return gaussians

        # Render images at test views
        xyz = xyz.float()
        feature = feature.float()
        scale = scale.exp().float()
        rotation = F.normalize(rotation.float(), p=2, dim=-1)
        opacity = opacity.sigmoid().squeeze(-1).float()
        if use_checkpoint:
            # cannot simply use torch checkpoint as memory reduction relies on the loop through views
            renderings = GaussianRenderer.apply(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, W, H, 
                                                self.config.model.gaussians.sh_degree, self.config.model.gaussians.near_plane,
                                                self.config.model.gaussians.far_plane)
        else:
            renderings = self.render(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr, W, H) # (B, V, H, W, 3)

        if test_images is None:
            return renderings

        #TODO: compute loss, gs pruning, visualize results 





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
                "num_heads": 8
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
            "num_layers": 4,
            "block_type": "tmtm",
            "merge_layers": [1,2],
            "transformer": {
                "num_heads": 8
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
                "far_plane": 100
            }
        }
    })
    llrm = LongLRM(config).to("cuda")
    input_images = torch.randn(batch_size, view, 3, h, w).to("cuda")
    input_images = nn.Parameter(input_images)
    input_images.retain_grad()
    input_intr = torch.zeros(batch_size, view, 4).to("cuda")
    input_intr[:, :, 0] = 1
    input_intr[:, :, 1] = 1
    input_intr[:, :, 2] = w//2
    input_intr[:, :, 3] = h//2
    input_c2ws = torch.eye(4).unsqueeze(0).expand(batch_size, view, 4, 4).to("cuda")

    view_test = 256
    test_intr = torch.zeros(batch_size, view_test, 4).to("cuda")
    test_intr[:, :, 0] = 1
    test_intr[:, :, 1] = 1
    test_intr[:, :, 2] = w//2
    test_intr[:, :, 3] = h//2
    test_c2ws = torch.eye(4).unsqueeze(0).expand(batch_size, view_test, 4, 4).to("cuda")

    with torch.autocast(enabled=True, device_type="cuda"):
        renderings = llrm(input_images, input_intr, input_c2ws,
                        test_images=None, test_intr=test_intr, test_c2ws=test_c2ws, use_checkpoint=True)
    renderings.mean().backward()
    render = renderings.clone()
    print("Peak memory with checkpoint:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
    print("Renderings shape:", renderings.shape)

    grad = input_images.grad.clone()
    input_images.grad.zero_()
    with torch.autocast(enabled=True, device_type="cuda"):
        renderings = llrm(input_images, input_intr, input_c2ws,
                         test_images=None, test_intr=test_intr, test_c2ws=test_c2ws, use_checkpoint=False)
    renderings.mean().backward()
    render2 = renderings.clone()
    grad2 = input_images.grad.clone()
    print("Peak memory without checkpoint:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")

    diff_render = (render - render2).abs()
    print("Max render diff:", diff_render.max())
    print("Norm render diff:", diff_render.norm())

    diff = (grad - grad2).abs()
    print("Max grad diff:", diff.max())
    print("Norm grad diff:", diff.norm())
