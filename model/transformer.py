# Adapted for Long-LRM by Ziwen 2024
# from timm library https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# and DINO https://github.com/facebookresearch/dino/blob/main/vision_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

try:
    import xformers.ops as xops # flashatt v2
except ImportError:
    xops = None

class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4., mlp_bias=False, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=mlp_bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=mlp_bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input 
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim, head_dim=64, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., use_flashatt_v2=True):
        super().__init__()
        assert dim % head_dim == 0, 'dim must be divisible by head_dim'
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop_p = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_flashatt_v2 = use_flashatt_v2

    def forward(self, x):
        """
        x: (B, L, D)
        Returns: same shape as input 
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        if self.use_flashatt_v2:
            qkv = qkv.permute(2, 0, 1, 3, 4)
            q, k, v = qkv[0], qkv[1], qkv[2] # (B, N, H, C)
            x = xops.memory_efficient_attention(q, k, v, op=(xops.fmha.flash.FwOp, xops.fmha.flash.BwOp), p=self.attn_drop_p)
            x = rearrange(x, 'b n h d -> b n (h d)')
        else:
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2] # (B, H, N, C)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, head_dim, mlp_ratio=4., mlp_bias=False, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flashatt_v2=True):
        super().__init__()
        self.norm1 = norm_layer(dim, bias=False)
        self.attn = SelfAttention(
            dim, head_dim=head_dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, use_flashatt_v2=use_flashatt_v2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, bias=False)
        self.mlp = Mlp(in_features=dim, mlp_ratio=mlp_ratio, mlp_bias=mlp_bias, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        """
        x: (B, L, D)
        Returns: same shape as input
        """
        y = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


if __name__ == "__main__":
    # Test TransformerBlock
    batch_size = 4
    seq_len = 128
    input_dim = 256
    head_dim = 32
    mlp_ratio = 4
    
    model = TransformerBlock(dim=input_dim, head_dim=head_dim, mlp_ratio=mlp_ratio, use_flashatt_v2=True).to("cuda").to(torch.float16)
    model_no_flashatt = TransformerBlock(dim=input_dim, head_dim=head_dim, mlp_ratio=mlp_ratio, use_flashatt_v2=False).to("cuda").to(torch.float16)
    # copy param to model_no_flashatt
    model_no_flashatt.load_state_dict(model.state_dict())
    x = torch.randn(batch_size, seq_len, input_dim).to("cuda").to(torch.float16)
    out = model(x)
    out_no_flashatt = model_no_flashatt(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Output shape (no flashatt):", out_no_flashatt.shape)
    print("Output:", out[0,0,:10])
    print("Output (no flashatt):", out_no_flashatt[0,0,:10])

    out_diff = out - out_no_flashatt
    norm_diff = torch.linalg.norm(out_diff)
    print("Norm difference:", norm_diff)
    print("Max difference:", torch.max(out_diff.abs()))
