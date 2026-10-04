# LoopViT architecture notes

The model is **LoopViT** (`loop_vit.py`). It's a standard ViT with one change,
borrowed from Nanbeige4.2: it stores a single stack of **B** transformer blocks
and runs that stack **K** times. The output of the last block goes straight back
into the first block for the next pass, so you get B×K blocks of depth while only
paying for B blocks of weights.

## Data flow

```
image (N,3,224,224)
  → PatchEmbed: Conv2d(3→384, k=16, s=16) → 196 patch tokens
  → prepend [CLS] token, add learned position embedding (197 tokens), dropout
  → ┌──────── shared stack (B=6 distinct blocks) ────────┐
    │ [+ loop_embed[k] if enabled]                         │
    │ block_1 → block_2 → … → block_6                      │  × K=2 passes
    └──────── output fed back into block_1 ────────────────┘
  → LayerNorm
  → pool: CLS token (default) or mean of the patch tokens
  → Linear head → num_classes logits
```

## Components

- **TransformerBlock**: a pre-norm ViT block, `x + Attn(LN(x))` then
  `x + MLP(LN(x))`, with stochastic depth on each residual branch.
- **Attention**: multi-head self-attention through
  `F.scaled_dot_product_attention`, using a fused QKV projection with bias.
- **MLP**: Linear → GELU → Linear, with a hidden size of `dim × mlp_ratio`
  (384 → 1536 → 384).
- **The loop** (`LoopViT.loop`): applies all blocks in order, then repeats the
  whole stack. The drop-path rate rises linearly over the unrolled depth of
  B×K = 12 applications, so the same block gets a lower rate on pass 1 than on
  pass 2.
- **Optional `loop_embedding`** (off by default): one learned vector per pass,
  added before each pass so the shared blocks can tell pass 1 from pass 2. If you
  run more passes than the model was trained with, the extra passes reuse the
  last vector.

## Default config (`config.yaml`)

Roughly ViT-Small width with half the stored depth.

| Setting | Value |
|---|---|
| image / patch | 224 / 16 → 196 patches + CLS |
| dim / heads | 384 / 6 |
| B (blocks) × K (loops) | 6 × 2 = 12 block applications |
| mlp_ratio | 4.0 |
| drop_path | 0.1 |
| pool | cls |

By hand count from these settings (not a run of the code), that's about
**11.0M parameters**: about 1.77M per block × 6, plus about 0.38M for the
embedding, norm and head. An untied 12-block ViT of the same size would have
about 21.7M. Compute is the same as that 12-block model, in both the forward and
backward pass. `model.param_report()` prints these numbers exactly.

## Inference extras

- `model(x, num_loops=k)` overrides the number of passes at inference, so you can
  test whether more or fewer loops helps.
- `model.logits_per_loop(x)` returns the logits read out after each pass, so you
  can see how accuracy changes from pass to pass.
- The classifier head starts at zero, and the other weights use trunc-normal
  (std 0.02) or Xavier init, the usual ViT setup.
