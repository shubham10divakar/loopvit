"""
Train a Nanbeige-style LoopViT on an image folder.

    # everything from the YAML
    python train.py --config config.yaml

    # YAML + command-line overrides
    python train.py --config config.yaml --train-dir data/train --num-classes 5 \
                    --num-blocks 6 --num-loops 2 --epochs 50

    # just build the model and print the summary
    python train.py --config config.yaml --summary-only --num-classes 10

Command-line flags override the YAML, and the YAML overrides the defaults below.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time

import torch
import torch.nn as nn

from data import build_dataloaders
from loop_vit import LoopViT, LoopViTConfig, print_model_summary


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y")


def int_or_none(v):
    return None if v is None or str(v).lower() in ("none", "null", "") else int(v)


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=str, default=None, help="YAML config file")

    d = p.add_argument_group("data")
    d.add_argument("--train-dir", type=str, default=None)
    d.add_argument("--val-dir", type=str, default=None)
    d.add_argument("--num-classes", type=int_or_none, default=None, help="use N class folders (default: all)")
    d.add_argument("--class-selection", choices=["first", "random"], default="first")
    d.add_argument("--classes", nargs="*", default=None, help="explicit class folder names")
    d.add_argument("--max-per-class", type=int_or_none, default=None)
    d.add_argument("--val-split", type=float, default=0.1)
    d.add_argument("--augment", choices=["none", "basic", "trivial"], default="basic")
    d.add_argument("--num-workers", type=int, default=4)

    m = p.add_argument_group("model")
    m.add_argument("--image-size", type=int, default=224)
    m.add_argument("--patch-size", type=int, default=16)
    m.add_argument("--dim", type=int, default=384)
    m.add_argument("--num-blocks", type=int, default=6, help="B: distinct blocks in the shared stack")
    m.add_argument("--num-loops", type=int, default=2, help="K: passes through the stack")
    m.add_argument("--num-heads", type=int, default=6)
    m.add_argument("--mlp-ratio", type=float, default=4.0)
    m.add_argument("--dropout", type=float, default=0.0)
    m.add_argument("--attn-dropout", type=float, default=0.0)
    m.add_argument("--drop-path", type=float, default=0.1)
    m.add_argument("--pool", choices=["cls", "mean"], default="cls")
    m.add_argument("--loop-embedding", type=str2bool, default=False)

    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--min-lr", type=float, default=1e-5)
    t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--warmup-epochs", type=int, default=5)
    t.add_argument("--label-smoothing", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--amp", type=str2bool, default=True, help="mixed precision on CUDA")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--device", type=str, default="auto")
    t.add_argument("--output-dir", type=str, default="runs/loopvit")
    t.add_argument("--summary-only", action="store_true")
    t.add_argument("--resume", type=str, default=None, help="checkpoint (e.g. runs/loopvit/last.pt) to resume from")

    # YAML -> defaults, then CLI on top
    pre, _ = p.parse_known_args()
    if pre.config:
        import yaml
        with open(pre.config) as f:
            cfg = yaml.safe_load(f) or {}
        known = {a.dest for a in p._actions}
        cfg = {k.replace("-", "_"): v for k, v in cfg.items()}
        unknown = set(cfg) - known
        if unknown:
            raise ValueError(f"Unknown keys in {pre.config}: {sorted(unknown)}")
        p.set_defaults(**cfg)
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_lr(step, total, warmup, base, min_lr):
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * t))


def param_groups(model, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or n.endswith((".bias", "pos_embed", "cls_token", "loop_embed")) else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate(model, loader, device, criterion, per_loop=False):
    model.eval()
    n, loss_sum, correct = 0, 0.0, 0
    loop_correct = None
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if per_loop:
            outs = model.logits_per_loop(x)
            logits = outs[-1]
            hits = [(o.argmax(1) == y).sum().item() for o in outs]
            loop_correct = hits if loop_correct is None else [a + b for a, b in zip(loop_correct, hits)]
        else:
            logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    res = {"loss": loss_sum / max(n, 1), "acc": correct / max(n, 1)}
    if per_loop and loop_correct is not None:
        res["acc_per_loop"] = [c / n for c in loop_correct]
    return res


def main():
    args = get_args()
    seed_all(args.seed)
    device = pick_device(args.device)

    # ---- data ---------------------------------------------------------------
    if args.summary_only and not (args.train_dir and os.path.isdir(args.train_dir)):
        class_names = [f"class_{i}" for i in range(args.num_classes or 10)]
        train_loader = val_loader = None
    else:
        if not args.train_dir:
            raise SystemExit("--train-dir (or train_dir in the YAML) is required")
        train_loader, val_loader, class_names = build_dataloaders(
            args.train_dir, args.val_dir, args.image_size, args.batch_size,
            args.num_workers, args.num_classes, args.class_selection, args.classes,
            args.max_per_class, args.val_split, args.augment, args.seed,
            pin_memory=device.type == "cuda")

    # ---- model --------------------------------------------------------------
    mcfg = LoopViTConfig(
        image_size=args.image_size, patch_size=args.patch_size, num_classes=len(class_names),
        dim=args.dim, num_blocks=args.num_blocks, num_loops=args.num_loops,
        num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, dropout=args.dropout,
        attn_dropout=args.attn_dropout, drop_path=args.drop_path, pool=args.pool,
        loop_embedding=args.loop_embedding)
    model = LoopViT(mcfg)
    print_model_summary(model)
    model.to(device)
    if args.summary_only:
        return
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump({"args": vars(args), "model": mcfg.to_dict(), "classes": class_names}, f, indent=2)

    # ---- optimisation -------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.999))
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    start_epoch, step, best_acc = 1, 0, -1.0
    if args.resume:
        print(f"resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if ckpt.get("classes") != class_names:
            raise SystemExit("--resume checkpoint classes do not match the current dataset/config")
        if ckpt.get("model_cfg") != mcfg.to_dict():
            raise SystemExit("--resume checkpoint model config does not match the current model args")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            opt.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        step = ckpt.get("step", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", -1.0)
        rng = ckpt.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"resumed at epoch {start_epoch}, step {step}, best_acc {best_acc:.4f}")

    log_path = os.path.join(args.output_dir, "log.csv")
    if not (args.resume and os.path.exists(log_path)):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_acc_per_loop", "sec"])

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        n, loss_sum, correct = 0, 0.0, 0
        for x, y in train_loader:
            lr = cosine_lr(step, total_steps, warmup_steps, args.lr, args.min_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            step += 1
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)

        tr = {"loss": loss_sum / n, "acc": correct / n}
        va = evaluate(model, val_loader, device, eval_criterion, per_loop=True) if val_loader else {}
        sec = time.time() - t0
        per_loop = " ".join(f"{a:.3f}" for a in va.get("acc_per_loop", []))
        msg = (f"epoch {epoch:3d}/{args.epochs} | lr {lr:.2e} | train loss {tr['loss']:.4f} "
               f"acc {tr['acc']:.4f}")
        if va:
            msg += f" | val loss {va['loss']:.4f} acc {va['acc']:.4f} | acc after each loop [{per_loop}]"
        print(msg + f" | {sec:.1f}s")
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.3e}", f"{tr['loss']:.4f}", f"{tr['acc']:.4f}",
                                    f"{va.get('loss', float('nan')):.4f}", f"{va.get('acc', float('nan')):.4f}",
                                    per_loop, f"{sec:.1f}"])

        score = va.get("acc", tr["acc"])
        is_best = score > best_acc
        best_acc = max(best_acc, score)
        ckpt = {"model": model.state_dict(), "model_cfg": mcfg.to_dict(),
                "classes": class_names, "epoch": epoch, "val_acc": va.get("acc"),
                "step": step, "best_acc": best_acc,
                "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                "rng_state": {"python": random.getstate(), "torch": torch.get_rng_state(),
                              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))

    print(f"done. best {'val' if val_loader else 'train'} acc {best_acc:.4f} -> {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
