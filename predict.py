"""
Classify images with a trained LoopViT checkpoint.

    python predict.py --ckpt runs/loopvit/best.pt --images a.jpg b.png
    python predict.py --ckpt runs/loopvit/best.pt --images some_folder/ --num-loops 3
"""
import argparse
import os

import torch
from PIL import Image

from data import build_transforms
from loop_vit import LoopViT, LoopViTConfig

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", nargs="+", required=True, help="files and/or folders")
    p.add_argument("--num-loops", type=int, default=None, help="override passes at inference")
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = LoopViTConfig(**ckpt["model_cfg"])
    model = LoopViT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    classes = ckpt["classes"]
    _, tf = build_transforms(cfg.image_size)

    paths = []
    for item in args.images:
        if os.path.isdir(item):
            paths += sorted(os.path.join(item, f) for f in os.listdir(item) if f.lower().endswith(EXTS))
        else:
            paths.append(item)

    k = min(args.topk, len(classes))
    with torch.no_grad():
        for path in paths:
            x = tf(Image.open(path).convert("RGB")).unsqueeze(0)
            prob = model(x, num_loops=args.num_loops).softmax(-1)[0]
            top = prob.topk(k)
            preds = ", ".join(f"{classes[i]} {v:.3f}" for v, i in zip(top.values.tolist(), top.indices.tolist()))
            print(f"{path}: {preds}")


if __name__ == "__main__":
    main()
