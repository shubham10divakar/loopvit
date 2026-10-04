"""
Grad-CAM for LoopViT, plus deletion / insertion faithfulness scores.

Grad-CAM on a ViT
-----------------
The usual target for ViT Grad-CAM is the output of the LAST block's first
LayerNorm (`blocks[-1].norm1`), i.e. the tokens right before the final
attention. The class score is back-propagated to those tokens, the gradient is
averaged over the patch tokens to get one weight per channel, and

    cam(patch) = ReLU( sum_d  w_d * A[patch, d] )

is reshaped to the patch grid and upsampled to the image.

Because LoopViT reuses the same blocks on every pass, the hooked LayerNorm runs
once per pass. So one forward + backward gives a Grad-CAM map for EVERY pass
(cams[k] = map at the last block of pass k+1), and you can show how the
evidence changes from pass 1 to pass K.

Faithfulness (deletion / insertion, Petsiuk et al., RISE, BMVC 2018)
--------------------------------------------------------------------
Patches are removed (deletion) or revealed (insertion) in order of heat-map
importance, and the predicted-class probability is tracked. If the heat map
really points at the evidence the model uses, the probability falls fast under
deletion (low AUC) and rises fast under insertion (high AUC), clearly better
than a random patch order.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from data import IMAGENET_MEAN, IMAGENET_STD


class LoopViTGradCAM:
    def __init__(self, model, block_index: int = -1):
        self.model = model
        self.layer = model.blocks[block_index].norm1

    def __call__(self, x, target=None, num_loops=None):
        """x: (N,3,H,W) normalized images. target: None (= predicted class) or
        (N,) class indices. Returns (logits, target, cams) where cams is a list
        with one (N,H,W) tensor in [0,1] per loop pass; cams[-1] is the final pass."""
        model = self.model
        was_training = model.training
        model.eval()
        acts = []

        def hook(_m, _inp, out):
            out.retain_grad()
            acts.append(out)

        handle = self.layer.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                logits = model(x, num_loops=num_loops)
                if target is None:
                    target = logits.argmax(1)
                target = torch.as_tensor(target, device=logits.device).long()
                model.zero_grad(set_to_none=True)
                logits.gather(1, target[:, None]).sum().backward()
        finally:
            handle.remove()
            model.train(was_training)

        grid = model.patch_embed.grid
        N, _, H, W = x.shape
        cams = []
        for a in acts:
            A = a.detach()[:, 1:]                       # drop CLS -> (N, P, D)
            w = a.grad[:, 1:].mean(1, keepdim=True)     # (N, 1, D)
            cam = F.relu((w * A).sum(-1)).reshape(N, 1, grid, grid)
            cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)[:, 0]
            flat = cam.flatten(1)
            lo, hi = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
            cams.append(((cam - lo) / (hi - lo).clamp_min(1e-8)).cpu())
        model.zero_grad(set_to_none=True)
        return logits.detach(), target.detach(), cams


def denormalize(x):
    """(3,H,W) normalized tensor -> (H,W,3) float numpy image in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def overlay(img, cam, alpha=0.45, cmap="jet"):
    import matplotlib.pyplot as plt
    heat = plt.get_cmap(cmap)(np.asarray(cam))[..., :3]
    return np.clip((1 - alpha) * img + alpha * heat, 0, 1)


@torch.no_grad()
def deletion_insertion(model, x, target, cam, patch_size, steps=16, order=None,
                       num_loops=None, batch_size=64):
    """Deletion and insertion curves at patch granularity.

    x: (N,3,H,W) normalized, target: (N,), cam: (N,H,W) importance map, or
    order: (N,P) explicit patch order (e.g. random baseline).
    Deleted patches are set to 0 in normalized space (the dataset mean colour);
    insertion starts from a heavily blurred copy of the image.
    Returns (fractions (S+1,), deletion (N,S+1), insertion (N,S+1)) probabilities.
    """
    from torchvision.transforms.functional import gaussian_blur

    model.eval()
    N, _, H, W = x.shape
    g = H // patch_size
    P = g * g
    if order is None:
        score = F.avg_pool2d(cam[:, None].float().to(x.device), patch_size)[:, 0].flatten(1)
        order = score.argsort(1, descending=True)
    order = order.to(x.device)
    blurred = gaussian_blur(x, kernel_size=[51, 51], sigma=[25.0, 25.0])
    ks = np.unique(np.linspace(0, P, steps + 1).round().astype(int))
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(P, device=x.device).expand(N, P))

    def probs(imgs):
        out = []
        for i in range(0, len(imgs), batch_size):
            out.append(model(imgs[i:i + batch_size], num_loops=num_loops).softmax(-1))
        return torch.cat(out).gather(1, target[:, None].to(x.device))[:, 0]

    dels, ins = [], []
    for k in ks:
        m = (rank < int(k)).float().reshape(N, 1, g, g)
        m = F.interpolate(m, size=(H, W), mode="nearest")
        dels.append(probs(x * (1 - m)).cpu())
        ins.append(probs(blurred * (1 - m) + x * m).cpu())
    fr = ks / P
    return fr, torch.stack(dels, 1).numpy(), torch.stack(ins, 1).numpy()


def curve_auc(fr, curves):
    """Per-sample area under a deletion/insertion curve."""
    return np.trapezoid(curves, fr, axis=1)


def save_gradcam_grid(rows, path_noext, title=None):
    """rows: list of dicts with keys
         image (H,W,3) in [0,1], cams [ (H,W) per pass ], true (str or None),
         pred (str), conf (float)
    Draws one row per image: original | Grad-CAM pass 1 | ... | pass K."""
    import matplotlib.pyplot as plt
    from plots import save, short

    if not rows:
        return
    K = len(rows[0]["cams"])
    fig, axes = plt.subplots(len(rows), K + 1, figsize=(2.4 * (K + 1), 2.5 * len(rows)), squeeze=False)
    for i, r in enumerate(rows):
        ax = axes[i, 0]
        ax.imshow(r["image"])
        ok = r.get("true") is None or r["true"] == r["pred"]
        lab = f"true: {short(r['true'], 26)}\n" if r.get("true") is not None else ""
        ax.set_title(f"{lab}pred: {short(r['pred'], 26)} ({r['conf']:.2f})", fontsize=7,
                     color="black" if ok else "crimson")
        for k, cam in enumerate(r["cams"]):
            axes[i, k + 1].imshow(overlay(r["image"], cam))
            if i == 0:
                last = " (final)" if k == K - 1 else ""
                axes[i, k + 1].set_title(f"Grad-CAM, pass {k + 1}{last}", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save(fig, path_noext)
