"""
Full evaluation report for a trained LoopViT run: every score, table and figure
a paper usually needs, written to one folder.

    # evaluate best.pt of a run on its held-out split (test if it has one, else val)
    python evaluate.py --run-dir runs/loopvit

    # pick the split / checkpoint / output folder explicitly
    python evaluate.py --run-dir runs/loopvit --split test --ckpt runs/loopvit/last.pt \
                       --out-dir runs/loopvit/report_test

The data split is rebuilt from the run's config.json with the same seed, so the
images are exactly the run's validation/test images.

Output (<out-dir>/):
  metrics.json               all summary scores + 95% bootstrap CIs + per-class + efficiency
  metrics_summary.csv        one row per metric (value, CI low, CI high)
  per_class_metrics.csv      precision / recall / specificity / F1 / NPV / AUC / AP / Brier per class
  classification_report.txt  sklearn text report
  table_main.tex, table_per_class.tex, table_loops.tex   LaTeX (booktabs) tables
  predictions.csv            path, true, predicted, confidence, per-class probabilities
  per_loop_metrics.csv       scores when reading out after 1..max_loops passes
  efficiency.json            params, untied-equivalent params, GFLOPs, latency, throughput
  figures/                   confusion matrices, ROC, PR, reliability, per-class bars,
                             training curves, loop ablation, t-SNE, deletion/insertion
  gradcam/                   Grad-CAM grids (per class, misclassified) for every loop pass
  faithfulness.json          deletion / insertion AUCs of Grad-CAM vs a random patch order
  report.md                  short human-readable summary
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import classification_report

import metrics as M
import plots as P
from data import make_eval_loader, split_samples
from explain import LoopViTGradCAM, curve_auc, deletion_insertion, denormalize, save_gradcam_grid
from loop_vit import LoopViT, LoopViTConfig


def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default="runs/loopvit", help="folder with config.json, best.pt, log.csv")
    p.add_argument("--ckpt", default=None, help="checkpoint (default: <run-dir>/best.pt)")
    p.add_argument("--split", choices=["auto", "val", "test", "train"], default="auto",
                   help="auto = test if the run has a test split, else val")
    p.add_argument("--train-dir", default=None, help="override the run's train_dir (e.g. data moved)")
    p.add_argument("--val-dir", default=None)
    p.add_argument("--test-dir", default=None)
    p.add_argument("--out-dir", default=None, help="default: <run-dir>/report_<split>")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-loops", type=int, default=None,
                   help="read out after 1..max_loops passes for the loop ablation (default: 2 x trained K)")
    p.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples for 95%% CIs (0 = off)")
    p.add_argument("--gradcam-per-class", type=int, default=4, help="correct examples per class in the Grad-CAM grids")
    p.add_argument("--gradcam-misclassified", type=int, default=12, help="misclassified examples to visualise")
    p.add_argument("--faithfulness-samples", type=int, default=100,
                   help="images for the deletion/insertion test (0 = off)")
    p.add_argument("--no-tsne", action="store_true")
    p.add_argument("--no-efficiency", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_run(args, device):
    with open(os.path.join(args.run_dir, "config.json")) as f:
        run_cfg = json.load(f)
    ckpt_path = args.ckpt or os.path.join(args.run_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = LoopViT(LoopViTConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return run_cfg, ckpt, ckpt_path, model


def rebuild_split(run_cfg, args, classes):
    a = run_cfg["args"]
    chosen, tr, va, te = split_samples(
        args.train_dir or a["train_dir"], args.val_dir or a.get("val_dir"),
        args.test_dir or a.get("test_dir"), a.get("num_classes"),
        a.get("class_selection", "first"), a.get("classes"), a.get("max_per_class"),
        a.get("val_split", 0.1), a.get("test_split", 0.0), a.get("seed", 42))
    if chosen != classes:
        raise SystemExit(f"classes rebuilt from config ({chosen}) do not match the checkpoint ({classes})")
    split = args.split
    if split == "auto":
        split = "test" if te else "val"
    samples = {"train": tr, "val": va, "test": te}[split]
    if not samples:
        raise SystemExit(f"the run has no '{split}' images")
    return split, samples


@torch.no_grad()
def run_inference(model, loader, device, max_loops):
    """Logits after every pass 1..max_loops, plus pooled features at the
    trained loop count, for every image in loader order."""
    K = model.cfg.num_loops
    logits = [[] for _ in range(max_loops)]
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        states = model.loop(model.embed(x), max_loops, return_all=True)
        for k, s in enumerate(states):
            f = model.pool(s)
            logits[k].append(model.head(f).float().cpu())
            if k == K - 1:
                feats.append(f.float().cpu())
        labels.append(y)
    return ([torch.cat(l).numpy() for l in logits], torch.cat(feats).numpy(),
            torch.cat(labels).numpy())


def softmax(z):
    z = z.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def efficiency(model, device, batch_size=64):
    cfg = model.cfg
    rep = model.param_report()
    out = {k.split(" ")[0]: v for k, v in rep.items()}
    x1 = torch.randn(1, cfg.in_chans, cfg.image_size, cfg.image_size, device=device)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(display=False) as fc:
            model(x1)
        flops = fc.get_total_flops()
        out["gflops_per_image"] = flops / 1e9
        out["gmacs_per_image"] = flops / 2e9
    except Exception as e:  # pragma: no cover - depends on torch build
        out["flops_error"] = repr(e)

    def bench(x, iters):
        with torch.no_grad():
            for _ in range(5):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(iters):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - t) / iters

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    out["latency_ms_batch1"] = bench(x1, 30) * 1000
    xb = torch.randn(batch_size, cfg.in_chans, cfg.image_size, cfg.image_size, device=device)
    out["throughput_img_per_s"] = batch_size / bench(xb, 10)
    out["throughput_batch_size"] = batch_size
    if device.type == "cuda":
        out["peak_gpu_mem_mb_inference"] = torch.cuda.max_memory_allocated() / 2**20
        out["device"] = torch.cuda.get_device_name(device)
    else:
        out["device"] = str(device)
    out["precision"] = "fp32"
    return out


def write_csv(path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def gradcam_section(model, ds, prob, y, classes, out_dir, args, device):
    """Grad-CAM grids + deletion/insertion faithfulness."""
    gdir = os.path.join(out_dir, "gradcam")
    os.makedirs(gdir, exist_ok=True)
    cam = LoopViTGradCAM(model)
    pred, conf = prob.argmax(1), prob.max(1)

    def cams_for(idx, bs=16):
        rows = []
        for i in range(0, len(idx), bs):
            chunk = idx[i:i + bs]
            x = torch.stack([ds[j][0] for j in chunk]).to(device)
            logits, target, cams = cam(x)
            p = logits.softmax(-1).cpu()
            for n, j in enumerate(chunk):
                t = int(target[n])
                rows.append({"index": int(j), "image": denormalize(x[n]),
                             "cams": [c[n].numpy() for c in cams],
                             "true": classes[int(y[j])], "pred": classes[t], "conf": float(p[n, t]),
                             "path": ds.samples[j][0]})
        return rows

    # correct, most confident examples of every class
    per_class_rows = []
    for c in range(len(classes)):
        idx = np.where((y == c) & (pred == c))[0]
        idx = idx[np.argsort(-conf[idx])][: args.gradcam_per_class]
        rows = cams_for(list(idx))
        per_class_rows += rows
        safe = "".join(ch if ch.isalnum() else "_" for ch in classes[c])
        save_gradcam_grid(rows, os.path.join(gdir, f"class_{c:02d}_{safe}"),
                          title=f"Grad-CAM, correctly classified: {classes[c]}")
    # one representative per class on a single overview figure
    overview = []
    for c in range(len(classes)):
        r = [row for row in per_class_rows if row["true"] == classes[c]]
        overview += r[:1]
    save_gradcam_grid(overview, os.path.join(gdir, "overview_one_per_class"),
                      title="Grad-CAM per loop pass (one correct example per class)")
    # misclassified, most confident mistakes first
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-conf[wrong])][: args.gradcam_misclassified]
    wrong_rows = cams_for(list(wrong))
    save_gradcam_grid(wrong_rows, os.path.join(gdir, "misclassified"),
                      title="Grad-CAM on misclassified images (most confident first)")
    with open(os.path.join(gdir, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["figure", "path", "true", "pred", "confidence"])
        for r in per_class_rows:
            w.writerow(["class_grid", r["path"], r["true"], r["pred"], f"{r['conf']:.4f}"])
        for r in wrong_rows:
            w.writerow(["misclassified", r["path"], r["true"], r["pred"], f"{r['conf']:.4f}"])

    if args.faithfulness_samples <= 0:
        return None
    rng = np.random.default_rng(args.seed)
    n = min(args.faithfulness_samples, len(ds))
    idx = np.sort(rng.choice(len(ds), n, replace=False))
    patch = model.cfg.patch_size
    P_ = model.patch_embed.num_patches
    curves = {}
    for i in range(0, n, 16):
        chunk = idx[i:i + 16]
        x = torch.stack([ds[j][0] for j in chunk]).to(device)
        _, target, cams = cam(x)
        orders = {f"Grad-CAM pass {k + 1}": (cams[k], None) for k in range(len(cams))}
        g = torch.Generator().manual_seed(args.seed + i)
        orders["Random"] = (None, torch.stack([torch.randperm(P_, generator=g) for _ in chunk]))
        for name, (cm_, order) in orders.items():
            fr, d, ins = deletion_insertion(model, x, target, cm_, patch, order=order)
            c = curves.setdefault(name, {"deletion": [], "insertion": []})
            c["deletion"].append(d)
            c["insertion"].append(ins)
    curves = {k: {kk: np.concatenate(vv) for kk, vv in v.items()} for k, v in curves.items()}
    K = model.cfg.num_loops
    final = f"Grad-CAM pass {K}"
    P.plot_deletion_insertion(fr, {"Grad-CAM (final pass)": curves[final], "Random order": curves["Random"]},
                              os.path.join(out_dir, "figures", "faithfulness_deletion_insertion"))
    res = {"n_images": int(n), "fractions": fr.tolist(), "methods": {}}
    for name, c in curves.items():
        da, ia = curve_auc(fr, c["deletion"]), curve_auc(fr, c["insertion"])
        res["methods"][name] = {"deletion_auc": float(da.mean()), "deletion_auc_std": float(da.std()),
                                "insertion_auc": float(ia.mean()), "insertion_auc_std": float(ia.std()),
                                "deletion_curve": c["deletion"].mean(0).tolist(),
                                "insertion_curve": c["insertion"].mean(0).tolist()}
    # paired test: is Grad-CAM better than random on the same images?
    try:
        from scipy.stats import wilcoxon
        dg, dr = curve_auc(fr, curves[final]["deletion"]), curve_auc(fr, curves["Random"]["deletion"])
        ig, ir = curve_auc(fr, curves[final]["insertion"]), curve_auc(fr, curves["Random"]["insertion"])
        res["wilcoxon_p_deletion_gradcam_lt_random"] = float(wilcoxon(dg, dr, alternative="less").pvalue)
        res["wilcoxon_p_insertion_gradcam_gt_random"] = float(wilcoxon(ig, ir, alternative="greater").pvalue)
    except Exception as e:  # scipy missing or degenerate (all-equal) samples
        res["wilcoxon_error"] = repr(e)
    with open(os.path.join(out_dir, "faithfulness.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    args = get_args()
    device = pick_device(args.device)
    run_cfg, ckpt, ckpt_path, model = load_run(args, device)
    classes = ckpt["classes"]
    K = model.cfg.num_loops
    max_loops = args.max_loops or 2 * K
    split, samples = rebuild_split(run_cfg, args, classes)
    out_dir = args.out_dir or os.path.join(args.run_dir, f"report_{split}")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"[eval] {ckpt_path} (epoch {ckpt.get('epoch')}) on {split}: {len(samples)} images, "
          f"{len(classes)} classes -> {out_dir}")

    loader = make_eval_loader(samples, classes, model.cfg.image_size, args.batch_size,
                              args.num_workers, pin_memory=device.type == "cuda")
    t0 = time.time()
    logits_k, feats, y = run_inference(model, loader, device, max_loops)
    print(f"[eval] inference done in {time.time() - t0:.1f}s")
    probs_k = [softmax(z) for z in logits_k]
    prob = probs_k[K - 1]

    # ---- metrics at the trained loop count ---------------------------------
    res = M.compute_metrics(y, prob, classes)
    s = res["summary"]
    print(f"[eval] bootstrap CIs ({args.bootstrap} resamples)...")
    ci = M.bootstrap_ci(y, prob, args.bootstrap, seed=args.seed)
    write_csv(os.path.join(out_dir, "per_class_metrics.csv"), res["per_class"])
    write_csv(os.path.join(out_dir, "metrics_summary.csv"),
              [{"metric": k, "value": v, "ci95_low": ci.get(k, (None, None))[0],
                "ci95_high": ci.get(k, (None, None))[1]} for k, v in s.items()])
    with open(os.path.join(out_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y, prob.argmax(1), labels=list(range(len(classes))),
                                      target_names=classes, digits=4, zero_division=0))
    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), res["confusion_matrix"], fmt="%d",
               delimiter=",", header=",".join(classes), comments="")

    pred = prob.argmax(1)
    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "true", "pred", "confidence", "correct"] + [f"p_{c}" for c in classes])
        for (path, _), t, pr, pv in zip(samples, y, pred, prob):
            w.writerow([path, classes[t], classes[pr], f"{pv[pr]:.6f}", int(t == pr)]
                       + [f"{v:.6f}" for v in pv])

    # ---- loop ablation -----------------------------------------------------
    loop_rows = []
    for k, pk in enumerate(probs_k, 1):
        sk = M.compute_metrics(y, pk, classes)["summary"]
        loop_rows.append({"loops": k, "trained_loops": K, **{m: sk[m] for m in
                          ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "cohen_kappa",
                           "roc_auc_ovr_macro", "brier", "log_loss", "ece"]}})
    write_csv(os.path.join(out_dir, "per_loop_metrics.csv"), loop_rows)

    # ---- figures -----------------------------------------------------------
    print("[eval] figures...")
    cm = res["confusion_matrix"]
    P.plot_confusion_matrix(cm, classes, os.path.join(fig_dir, "confusion_matrix"))
    P.plot_confusion_matrix(cm, classes, os.path.join(fig_dir, "confusion_matrix_normalized"), normalize=True)
    P.plot_roc_curves(y, prob, classes, os.path.join(fig_dir, "roc_curves"))
    P.plot_pr_curves(y, prob, classes, os.path.join(fig_dir, "pr_curves"))
    P.plot_reliability(res["calibration_bins"], s["ece"], os.path.join(fig_dir, "reliability_diagram"))
    P.plot_per_class_bars(res["per_class"], os.path.join(fig_dir, "per_class_metrics"))
    P.plot_loop_ablation(loop_rows, K, os.path.join(fig_dir, "loop_ablation"))
    log_csv = os.path.join(args.run_dir, "log.csv")
    if os.path.exists(log_csv):
        P.plot_training_curves(log_csv, os.path.join(fig_dir, "training_curves"), ckpt.get("best_epoch"))
    if not args.no_tsne and len(y) >= 10:
        sub = np.random.default_rng(args.seed).permutation(len(y))[:3000]
        P.plot_embedding(feats[sub], y[sub], classes, os.path.join(fig_dir, "tsne_features"), seed=args.seed)

    # ---- efficiency --------------------------------------------------------
    eff = None
    if not args.no_efficiency:
        print("[eval] efficiency (params / FLOPs / latency)...")
        eff = efficiency(model, device, args.batch_size)
        with open(os.path.join(out_dir, "efficiency.json"), "w") as f:
            json.dump(eff, f, indent=2)

    # ---- Grad-CAM + faithfulness -------------------------------------------
    print("[eval] Grad-CAM and deletion/insertion faithfulness...")
    faith = gradcam_section(model, loader.dataset, prob, y, classes, out_dir, args, device)

    # ---- tables + json + markdown ------------------------------------------
    cap = f"LoopViT (B={model.cfg.num_blocks}, K={K}) on the {split} split ({len(y)} images)."
    with open(os.path.join(out_dir, "table_main.tex"), "w", encoding="utf-8") as f:
        f.write(M.latex_summary_table(s, ci, cap, f"tab:loopvit_{split}"))
    with open(os.path.join(out_dir, "table_per_class.tex"), "w", encoding="utf-8") as f:
        f.write(M.latex_per_class_table(res["per_class"], "Per-class results. " + cap,
                                        f"tab:loopvit_{split}_per_class"))
    with open(os.path.join(out_dir, "table_loops.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\caption{Read-out after $k$ loop passes "
                f"(trained with $K={K}$).}}\n\\label{{tab:loopvit_loops}}\n"
                "\\begin{tabular}{rcccc}\n\\toprule\n$k$ & Acc. & Macro F1 & MCC & ECE \\\\\n\\midrule\n"
                + "\n".join(f"{r['loops']} & {M.fmt(r['accuracy'])} & {M.fmt(r['f1_macro'])} & "
                            f"{M.fmt(r['mcc'])} & {M.fmt(r['ece'])} \\\\" for r in loop_rows)
                + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    out = {"checkpoint": ckpt_path, "epoch": ckpt.get("epoch"), "split": split,
           "n_images": int(len(y)), "classes": classes, "model_cfg": model.cfg.to_dict(),
           "summary": s, "ci95": ci, "per_class": res["per_class"],
           "confusion_matrix": cm.tolist(), "per_loop": loop_rows, "efficiency": eff,
           "faithfulness": None if faith is None else
           {k: v for k, v in faith.items() if k != "methods"} |
           {"methods": {m: {kk: vv for kk, vv in d.items() if not kk.endswith("_curve")}
                        for m, d in faith["methods"].items()}}}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)

    write_markdown(os.path.join(out_dir, "report.md"), out, ckpt, split)
    print(f"[eval] accuracy {s['accuracy']:.4f} | macro F1 {s['f1_macro']:.4f} | MCC {s['mcc']:.4f} | "
          f"ROC-AUC {s['roc_auc_ovr_macro']:.4f} | Brier {s['brier']:.4f} | ECE {s['ece']:.4f}")
    print(f"[eval] report written to {out_dir}")


def write_markdown(path, out, ckpt, split):
    s, ci = out["summary"], out["ci95"]
    L = [f"# LoopViT evaluation report ({split} split)", "",
         f"- checkpoint: `{out['checkpoint']}` (epoch {out['epoch']})",
         f"- images: {out['n_images']}, classes: {len(out['classes'])}",
         f"- model: B={out['model_cfg']['num_blocks']} blocks x K={out['model_cfg']['num_loops']} loops, "
         f"dim {out['model_cfg']['dim']}", ""]
    if split == "val":
        L += ["> **Note:** the validation split also picked `best.pt`, so these numbers are slightly "
              "optimistic. For the paper, train with `--test-split 0.1` (or `--test-dir`) and report the "
              "test split.", ""]
    L += ["## Main metrics", "", "| metric | value | 95% CI |", "|---|---|---|"]
    for k in M.HEADLINE:
        lo, hi = ci.get(k, (float("nan"), float("nan")))
        L.append(f"| {k} | {M.fmt(s.get(k))} | [{M.fmt(lo)}, {M.fmt(hi)}] |")
    extra = [k for k in s if k not in M.HEADLINE]
    L += ["", "Other scores: " + ", ".join(f"{k} = {M.fmt(s[k]) if isinstance(s[k], float) else s[k]}"
                                          for k in extra), ""]
    L += ["## Per-class", "", "| class | n | precision | recall | specificity | F1 | ROC-AUC |",
          "|---|---|---|---|---|---|---|"]
    for r in out["per_class"]:
        L.append(f"| {r['class']} | {r['support']} | {M.fmt(r['precision'])} | {M.fmt(r['recall'])} | "
                 f"{M.fmt(r['specificity'])} | {M.fmt(r['f1'])} | {M.fmt(r['roc_auc'])} |")
    L += ["", "## Accuracy vs. loop passes at inference", "", "| passes | accuracy | macro F1 | MCC | ECE |",
          "|---|---|---|---|---|"]
    for r in out["per_loop"]:
        mark = " (trained)" if r["loops"] == r["trained_loops"] else ""
        L.append(f"| {r['loops']}{mark} | {M.fmt(r['accuracy'])} | {M.fmt(r['f1_macro'])} | "
                 f"{M.fmt(r['mcc'])} | {M.fmt(r['ece'])} |")
    if out["efficiency"]:
        e = out["efficiency"]
        L += ["", "## Efficiency", ""] + [f"- {k}: {v:,.3f}" if isinstance(v, float) else f"- {k}: {v:,}"
                                          if isinstance(v, int) else f"- {k}: {v}" for k, v in e.items()]
    if out["faithfulness"]:
        f = out["faithfulness"]
        L += ["", f"## Grad-CAM faithfulness ({f['n_images']} images)", "",
              "Deletion AUC: lower is better. Insertion AUC: higher is better.", "",
              "| order | deletion AUC | insertion AUC |", "|---|---|---|"]
        for m, d in f["methods"].items():
            L.append(f"| {m} | {d['deletion_auc']:.4f} ± {d['deletion_auc_std']:.4f} | "
                     f"{d['insertion_auc']:.4f} ± {d['insertion_auc_std']:.4f} |")
        for k in ("wilcoxon_p_deletion_gradcam_lt_random", "wilcoxon_p_insertion_gradcam_gt_random"):
            if k in f:
                L.append(f"\n{k}: {f[k]:.3g}")
    L += ["", "## Files", "", "- `figures/`: confusion matrices, ROC, PR, reliability, per-class, "
          "training curves, loop ablation, t-SNE, deletion/insertion (PNG 300 dpi + PDF)",
          "- `gradcam/`: Grad-CAM grids per class and for misclassified images, one column per loop pass",
          "- `table_*.tex`: LaTeX tables; `*.csv` / `metrics.json`: raw numbers", ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


if __name__ == "__main__":
    main()
