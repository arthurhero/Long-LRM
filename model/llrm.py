import torch
import torch.nn as nn
import torch.nn.functional as F

from easydict import EasyDict as edict
from einops import rearrange

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
                global_tokens, image_tokens = x[:, :num_global_tokens], x[:, num_global_tokens:]
                global_tokens = self.resize_blocks[self.merge_at.index(i)](global_tokens)
                image_tokens = rearrange(image_tokens, "b (v h w) d -> (b v) d h w", v=v, h=h, w=w)
                image_tokens = self.merge_blocks[self.merge_at.index(i)](image_tokens)
                h = h // 2
                w = w // 2
                image_tokens = rearrange(image_tokens, "(b v) d h w -> b (v h w) d", v=v, h=h, w=w)
                x = torch.cat([global_tokens, image_tokens], dim=1)
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


if __name__ == "__main__":

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
            "block_type": ["t", "m", "t", "m"],
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

