"""
Grad-CAM heat maps for any images, one column per loop pass.

    python gradcam.py --ckpt runs/loopvit/best.pt --images leaf1.jpg some_folder/
    python gradcam.py --ckpt runs/loopvit/best.pt --images some_folder/ --target Apple___healthy
    python gradcam.py --ckpt runs/loopvit/best.pt --images a.jpg --num-loops 4 --out-dir cams/

Writes one figure per image (original | pass 1 | ... | pass K) and, with
--save-raw, the raw heat maps as .npy. evaluate.py makes the per-class and
misclassified grids for a whole split.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from data import build_transforms
from explain import LoopViTGradCAM, denormalize, save_gradcam_grid
from loop_vit import LoopViT, LoopViTConfig
from predict import EXTS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--images", nargs="+", required=True, help="files and/or folders")
    p.add_argument("--out-dir", default="gradcam_out")
    p.add_argument("--target", default=None, help="class name to explain (default: predicted class)")
    p.add_argument("--num-loops", type=int, default=None, help="override passes at inference")
    p.add_argument("--block", type=int, default=-1, help="which shared block's norm1 to hook (default: last)")
    p.add_argument("--save-raw", action="store_true", help="also save heat maps as .npy")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = LoopViT(LoopViTConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    classes = ckpt["classes"]
    _, tf = build_transforms(model.cfg.image_size)
    cam = LoopViTGradCAM(model, args.block)
    target = None
    if args.target is not None:
        if args.target not in classes:
            raise SystemExit(f"--target {args.target!r} is not one of {classes}")
        target = classes.index(args.target)

    paths = []
    for item in args.images:
        if os.path.isdir(item):
            paths += sorted(os.path.join(item, f) for f in os.listdir(item) if f.lower().endswith(EXTS))
        else:
            paths.append(item)
    os.makedirs(args.out_dir, exist_ok=True)

    for path in paths:
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        t = None if target is None else torch.tensor([target])
        logits, tgt, cams = cam(x, t, num_loops=args.num_loops)
        prob = logits.softmax(-1)[0]
        pred = int(prob.argmax())
        row = {"image": denormalize(x[0]), "cams": [c[0].numpy() for c in cams],
               "true": None, "pred": classes[pred], "conf": float(prob[pred])}
        stem = os.path.splitext(os.path.basename(path))[0]
        title = None if target is None else f"explaining class: {classes[target]}"
        save_gradcam_grid([row], os.path.join(args.out_dir, f"{stem}_gradcam"), title=title)
        if args.save_raw:
            np.save(os.path.join(args.out_dir, f"{stem}_gradcam.npy"), np.stack(row["cams"]))
        print(f"{path}: pred {classes[pred]} ({prob[pred]:.3f}), explained class "
              f"{classes[int(tgt[0])]} -> {args.out_dir}/{stem}_gradcam.png")


if __name__ == "__main__":
    main()
