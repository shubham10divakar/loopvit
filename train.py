"""
Train a Nanbeige-style LoopViT on an image folder.

    # everything from the YAML
    python train.py --config config.yaml

    # YAML + command-line overrides
    python train.py --config config.yaml --train-dir data/train --num-classes 5 \
                    --num-blocks 6 --num-loops 2 --epochs 50

    # just build the model and print the summary
    python train.py --config config.yaml --summary-only --num-classes 10

    # resume an interrupted run from <output_dir>/last.pt
    python train.py --config config.yaml --resume auto

    # stop when val accuracy has not improved for 15 epochs
    python train.py --config config.yaml --early-stopping-patience 15

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
    d.add_argument("--test-dir", type=str, default=None, help="optional held-out test folder")
    d.add_argument("--test-split", type=float, default=0.0,
                   help="fraction of train held out as a test set when no --test-dir (0 = none)")
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
    t.add_argument("--resume", type=str, default=None,
                   help="checkpoint to resume from, or 'auto' for <output_dir>/last.pt (fresh start if missing)")
    t.add_argument("--monitor", choices=["val_acc", "val_loss"], default="val_acc",
                   help="metric that picks best.pt and drives early stopping")
    t.add_argument("--early-stopping-patience", type=int, default=0,
                   help="stop after this many epochs without improvement (0 = off)")
    t.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                   help="smallest change in the monitored metric that counts as improvement")

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


def improved(score, best, mode, min_delta):
    if best is None:
        return True
    return score > best + min_delta if mode == "max" else score < best - min_delta


def resolve_resume(resume, output_dir):
    if not resume:
        return None
    if resume == "auto":
        path = os.path.join(output_dir, "last.pt")
        if os.path.exists(path):
            return path
        print(f"--resume auto: no {path} yet, starting fresh")
        return None
    if not os.path.exists(resume):
        raise SystemExit(f"--resume checkpoint not found: {resume}")
    return resume


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
        train_loader, val_loader, _, class_names = build_dataloaders(
            args.train_dir, args.val_dir, args.image_size, args.batch_size,
            args.num_workers, args.num_classes, args.class_selection, args.classes,
            args.max_per_class, args.val_split, args.augment, args.seed,
            pin_memory=device.type == "cuda", test_dir=args.test_dir,
            test_split=args.test_split)

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

    mode = "max" if args.monitor == "val_acc" else "min"
    if not val_loader:
        if args.early_stopping_patience:
            print("[warn] no validation set: early stopping disabled")
        args.early_stopping_patience = 0

    start_epoch, step = 1, 0
    best_score, best_epoch, bad_epochs = None, 0, 0
    resume_path = resolve_resume(args.resume, args.output_dir)
    if resume_path:
        print(f"resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if ckpt.get("classes") != class_names:
            raise SystemExit("--resume checkpoint classes do not match the current dataset/config")
        if ckpt.get("model_cfg") != mcfg.to_dict():
            raise SystemExit("--resume checkpoint model config does not match the current model args")
        model.load_state_dict(ckpt["model"])
        if ckpt.get("optimizer") is not None:
            opt.load_state_dict(ckpt["optimizer"])
        else:
            print("[warn] checkpoint has no optimizer state (older format): AdamW moments start from zero")
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        # older checkpoints did not store the step; rebuild it so the LR schedule continues
        step = ckpt.get("step", (start_epoch - 1) * steps_per_epoch)
        if ckpt.get("monitor", "val_acc") == args.monitor:
            best_score = ckpt.get("best_score", ckpt.get("best_acc", ckpt.get("val_acc")))
            if best_score is not None and best_score < 0:
                best_score = None
            best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", 0))
            bad_epochs = ckpt.get("bad_epochs", 0)
        else:
            print(f"[warn] checkpoint was monitoring {ckpt.get('monitor')}; best score reset")
        rng = ckpt.get("rng_state")
        if rng:
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"].cpu())
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        print(f"resumed at epoch {start_epoch}, step {step}, best {args.monitor} {best_score} "
              f"(epoch {best_epoch}), epochs without improvement {bad_epochs}")

    header = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_acc_per_loop", "sec"]
    log_path = os.path.join(args.output_dir, "log.csv")
    if resume_path and os.path.exists(log_path):
        # drop rows the checkpoint does not cover (e.g. logged just before a crash)
        with open(log_path, newline="") as f:
            rows = list(csv.reader(f))
        kept = [r for r in rows[1:] if r and r[0].isdigit() and int(r[0]) < start_epoch]
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerows([header] + kept)
    else:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    patience = args.early_stopping_patience
    stopped_early = bool(patience) and bad_epochs >= patience
    if stopped_early:
        print(f"early stopping already triggered in this run ({bad_epochs} epochs without improvement)")
    lr = args.lr
    train_start = time.time()
    epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        if stopped_early:
            epoch -= 1
            break
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

        # without a val set fall back to train accuracy for picking best.pt
        score = va["acc" if args.monitor == "val_acc" else "loss"] if va else tr["acc"]
        is_best = improved(score, best_score, mode if va else "max", args.early_stopping_min_delta)
        if is_best:
            best_score, best_epoch, bad_epochs = score, epoch, 0
        else:
            bad_epochs += 1

        msg = (f"epoch {epoch:3d}/{args.epochs} | lr {lr:.2e} | train loss {tr['loss']:.4f} "
               f"acc {tr['acc']:.4f}")
        if va:
            msg += f" | val loss {va['loss']:.4f} acc {va['acc']:.4f} | acc after each loop [{per_loop}]"
        msg += f" | {sec:.1f}s"
        if is_best:
            msg += " | *best*"
        elif patience:
            msg += f" | no improvement {bad_epochs}/{patience}"
        print(msg)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.3e}", f"{tr['loss']:.4f}", f"{tr['acc']:.4f}",
                                    f"{va.get('loss', float('nan')):.4f}", f"{va.get('acc', float('nan')):.4f}",
                                    per_loop, f"{sec:.1f}"])

        stopped_early = bool(patience) and bad_epochs >= patience
        ckpt = {"model": model.state_dict(), "model_cfg": mcfg.to_dict(),
                "classes": class_names, "epoch": epoch, "val_acc": va.get("acc"),
                "val_loss": va.get("loss"), "step": step,
                "monitor": args.monitor, "best_score": best_score, "best_epoch": best_epoch,
                "bad_epochs": bad_epochs, "stopped_early": stopped_early,
                "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                "rng_state": {"python": random.getstate(), "torch": torch.get_rng_state(),
                              "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
        if stopped_early:
            print(f"early stopping: {args.monitor} has not improved for {patience} epochs "
                  f"(best {best_score:.4f} at epoch {best_epoch})")
            break

    summary = {"best_epoch": best_epoch, "monitor": args.monitor if val_loader else "train_acc",
               "best_score": best_score, "last_epoch": epoch, "max_epochs": args.epochs,
               "stopped_early": stopped_early,
               "train_seconds_this_session": round(time.time() - train_start, 1)}
    with open(os.path.join(args.output_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    best_txt = f"{best_score:.4f}" if best_score is not None else "n/a"
    print(f"done. best {summary['monitor']} {best_txt} at epoch {best_epoch} -> {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
