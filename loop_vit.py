"""
LoopViT (Nanbeige-style) for image classification.

Nanbeige4.2-3B applies ONE stack of B distinct transformer blocks K times.
The hidden states that leave block B are fed straight back into block 1
(the "orange arrow"), so the unrolled network has B*K block applications
but stores weights for only B blocks:

    patches -> [CLS] + pos-embed
            -> ( block_1 -> block_2 -> ... -> block_B )   pass 1
            -> ( block_1 -> block_2 -> ... -> block_B )   pass 2   (same weights)
            ...                                           pass K
            -> LayerNorm -> head(CLS)

Things that follow from this (and that the article points out):
  * params  ~ B blocks  (not B*K)
  * compute ~ B*K block applications, forward AND backward
  * the loop count is fixed (Nanbeige uses K=2); you can override it at
    inference with `model(x, num_loops=...)` to see what more/fewer passes do.

Optional, off by default so the default model is the plain Nanbeige recipe:
  * loop_embedding  - a learned vector added at the start of each pass so the
                      shared blocks can tell pass 1 from pass 2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class LoopViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    num_classes: int = 10
    dim: int = 384
    num_blocks: int = 6          # B: distinct blocks in the shared stack
    num_loops: int = 2           # K: times the whole stack is applied (Nanbeige: 2)
    num_heads: int = 6
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout: float = 0.0
    attn_dropout: float = 0.0
    drop_path: float = 0.1       # max stochastic-depth rate, over the B*K applications
    pool: str = "cls"            # "cls" or "mean"
    loop_embedding: bool = False

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def drop_path(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """Stochastic depth. The rate is passed per call because a shared block is
    applied at several depths and each application gets its own rate."""
    if p == 0.0 or not training:
        return x
    keep = 1.0 - p
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
    return x * mask / keep


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, in_chans, dim):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):                      # (N, C, H, W)
        x = self.proj(x)                       # (N, D, H/p, W/p)
        return x.flatten(2).transpose(1, 2)    # (N, P, D)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        N, T, D = x.shape
        q, k, v = (self.qkv(x)
                   .reshape(N, T, 3, self.num_heads, self.head_dim)
                   .permute(2, 0, 3, 1, 4))            # 3 x (N, h, T, d)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0.0)
        x = x.transpose(1, 2).reshape(N, T, D)
        return self.proj_drop(self.proj(x))


class MLP(nn.Module):
    def __init__(self, dim, hidden, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class TransformerBlock(nn.Module):
    """Pre-norm block: attention + MLP, each with a shortcut."""

    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)

    def forward(self, x, dp: float = 0.0):
        x = x + drop_path(self.attn(self.norm1(x)), dp, self.training)
        x = x + drop_path(self.mlp(self.norm2(x)), dp, self.training)
        return x


# --------------------------------------------------------------------------- #
# LoopViT
# --------------------------------------------------------------------------- #
class LoopViT(nn.Module):
    def __init__(self, cfg: LoopViTConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.dim

        self.patch_embed = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.in_chans, D)
        n_tok = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tok, D))
        self.pos_drop = nn.Dropout(cfg.dropout)

        # The shared stack: B distinct blocks, reused on every pass.
        self.blocks = nn.ModuleList([
            TransformerBlock(D, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias,
                             cfg.dropout, cfg.attn_dropout)
            for _ in range(cfg.num_blocks)
        ])

        # One learned vector per pass (optional). Passes beyond the training
        # count reuse the last one.
        self.loop_embed = (nn.Parameter(torch.zeros(cfg.num_loops, 1, 1, D))
                           if cfg.loop_embedding else None)

        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, cfg.num_classes)
        self._init_weights()

    # ---- init --------------------------------------------------------------
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.loop_embed is not None:
            nn.init.trunc_normal_(self.loop_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                w = m.weight.data
                nn.init.xavier_uniform_(w.view(w.shape[0], -1))
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ---- helpers -----------------------------------------------------------
    def _drop_path_rates(self, total: int):
        """Linearly increasing rate over the UNROLLED depth (1 ... B*K)."""
        if total <= 1:
            return [0.0] * total
        return [self.cfg.drop_path * i / (total - 1) for i in range(total)]

    def embed(self, x):
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed
        return self.pos_drop(x)

    def loop(self, h, num_loops: int, return_all: bool = False):
        """Apply the shared stack `num_loops` times. The output of block B on
        pass k is the input of block 1 on pass k+1 (the Nanbeige feedback)."""
        B = len(self.blocks)
        rates = self._drop_path_rates(B * num_loops)
        states = []
        for k in range(num_loops):
            if self.loop_embed is not None:
                h = h + self.loop_embed[min(k, self.loop_embed.shape[0] - 1)]
            for b, blk in enumerate(self.blocks):
                h = blk(h, rates[k * B + b])
            if return_all:
                states.append(h)
        return states if return_all else h

    def pool(self, h):
        h = self.norm(h)
        return h[:, 0] if self.cfg.pool == "cls" else h[:, 1:].mean(dim=1)

    # ---- forward -----------------------------------------------------------
    def forward_features(self, x, num_loops: int | None = None):
        return self.pool(self.loop(self.embed(x), num_loops or self.cfg.num_loops))

    def forward(self, x, num_loops: int | None = None):
        return self.head(self.forward_features(x, num_loops))

    @torch.no_grad()
    def logits_per_loop(self, x, num_loops: int | None = None):
        """Logits read out after every pass: list of (N, num_classes).
        Handy to check how accuracy changes with the number of passes."""
        states = self.loop(self.embed(x), num_loops or self.cfg.num_loops, return_all=True)
        return [self.head(self.pool(s)) for s in states]

    # ---- bookkeeping -------------------------------------------------------
    def param_report(self) -> dict:
        """Parameter count vs a conventional ViT with B*K distinct blocks
        (the article's Figure 7 comparison)."""
        total = sum(p.numel() for p in self.parameters())
        per_block = sum(p.numel() for p in self.blocks[0].parameters())
        stack = per_block * len(self.blocks)
        K = self.cfg.num_loops
        return {
            "total_params": total,
            "params_per_block": per_block,
            "shared_stack_params": stack,
            "other_params (embed/norm/head)": total - stack,
            "block_applications": len(self.blocks) * K,
            "untied_equivalent_params": total + stack * (K - 1),
        }


def build_model(**kwargs) -> LoopViT:
    return LoopViT(LoopViTConfig(**kwargs))


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def _fmt(n: int) -> str:
    return f"{n:,} ({n / 1e6:.2f}M)"


def print_model_summary(model: LoopViT, batch_size: int = 1, device="cpu"):
    cfg = model.cfg
    line = "=" * 78
    print(line)
    print(" LoopViT (Nanbeige-style shared stack)")
    print(line)
    print(f" image {cfg.image_size}x{cfg.image_size}, patch {cfg.patch_size} "
          f"-> {model.patch_embed.num_patches} patches + 1 CLS token")
    print(f" dim {cfg.dim}, heads {cfg.num_heads}, mlp_ratio {cfg.mlp_ratio}, "
          f"classes {cfg.num_classes}, pool '{cfg.pool}'")
    print(f" shared stack: {cfg.num_blocks} blocks x {cfg.num_loops} loops "
          f"= {cfg.num_blocks * cfg.num_loops} block applications")
    unrolled = " | ".join(
        f"pass {k + 1}: apps {k * cfg.num_blocks + 1}-{(k + 1) * cfg.num_blocks}"
        for k in range(cfg.num_loops))
    print(f" unrolled: {unrolled}  (each pass reuses blocks 1-{cfg.num_blocks})")
    print(f" loop_embedding={cfg.loop_embedding}, drop_path={cfg.drop_path}")
    print(line)
    for k, v in model.param_report().items():
        print(f" {k:<32} {_fmt(v) if 'params' in k else v}")
    print(line)

    try:
        from torchinfo import summary
        summary(model,
                input_size=(batch_size, cfg.in_chans, cfg.image_size, cfg.image_size),
                depth=2, device=device,
                col_names=("input_size", "output_size", "num_params"),
                row_settings=("var_names",))
        print(" Note: '(recursive)' rows are the same block run again on a later pass;\n"
              " their parameters are counted once, their compute every time.")
    except ImportError:
        print(" (pip install torchinfo for the layer-by-layer table)")
        print(model)
    print(line)
