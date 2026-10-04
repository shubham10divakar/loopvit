"""
Figures for the paper. Every function saves a 300-dpi PNG and a vector PDF
next to each other (path given without extension).
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn import metrics as skm  # noqa: E402

from metrics import one_hot  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
})


def save(fig, path_noext):
    os.makedirs(os.path.dirname(path_noext) or ".", exist_ok=True)
    fig.savefig(path_noext + ".png", bbox_inches="tight")
    fig.savefig(path_noext + ".pdf", bbox_inches="tight")
    plt.close(fig)


def short(name, n=22):
    name = str(name).replace("___", ": ").replace("_", " ")
    return name if len(name) <= n else name[: n - 1] + "…"


def _class_colors(n):
    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    return [cmap(i % cmap.N) for i in range(n)]


# --------------------------------------------------------------------------- #
def plot_confusion_matrix(cm, class_names, path_noext, normalize=False, title=None):
    cm = np.asarray(cm, dtype=np.float64)
    n = len(class_names)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = np.nan_to_num(cm / cm.sum(1, keepdims=True))
    size = max(4.5, 0.55 * n + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    ax.grid(False)
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = [short(c) for c in class_names]
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title or ("Normalized confusion matrix (row = recall)" if normalize else "Confusion matrix"))
    if n <= 30:
        thresh = cm.max() / 2 if cm.max() > 0 else 0.5
        for i in range(n):
            for j in range(n):
                txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8 if n <= 15 else 6,
                        color="white" if cm[i, j] > thresh else "black")
    save(fig, path_noext)


def plot_roc_curves(y, prob, class_names, path_noext, title="ROC curves (one-vs-rest)"):
    n = len(class_names)
    Y = one_hot(np.asarray(y), n)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = _class_colors(n)
    grid = np.linspace(0, 1, 501)
    tprs, aucs = [], []
    for c in range(n):
        if not 0 < Y[:, c].sum() < len(Y):
            continue
        fpr, tpr, _ = skm.roc_curve(Y[:, c], prob[:, c])
        tprs.append(np.interp(grid, fpr, tpr))
        aucs.append(skm.auc(fpr, tpr))
        if n <= 15:
            ax.plot(fpr, tpr, lw=1.2, color=colors[c],
                    label=f"{short(class_names[c])} (AUC={skm.auc(fpr, tpr):.3f})")
        else:
            ax.plot(fpr, tpr, lw=0.6, color="0.75")
    fpr, tpr, _ = skm.roc_curve(Y.ravel(), prob.ravel())
    ax.plot(fpr, tpr, lw=2.2, ls="--", color="black", label=f"micro-average (AUC={skm.auc(fpr, tpr):.3f})")
    if tprs:
        mt = np.mean(tprs, 0)
        mt[0] = 0.0
        ax.plot(grid, mt, lw=2.2, ls=":", color="crimson", label=f"macro-average (AUC={np.mean(aucs):.3f})")
    ax.plot([0, 1], [0, 1], lw=0.8, color="0.6", ls="-")
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (sensitivity)", title=title)
    ax.legend(loc="lower right", fontsize=7)
    save(fig, path_noext)


def plot_pr_curves(y, prob, class_names, path_noext, title="Precision-recall curves"):
    n = len(class_names)
    Y = one_hot(np.asarray(y), n)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    colors = _class_colors(n)
    for c in range(n):
        if Y[:, c].sum() == 0:
            continue
        p, r, _ = skm.precision_recall_curve(Y[:, c], prob[:, c])
        ap = skm.average_precision_score(Y[:, c], prob[:, c])
        if n <= 15:
            ax.plot(r, p, lw=1.2, color=colors[c], label=f"{short(class_names[c])} (AP={ap:.3f})")
        else:
            ax.plot(r, p, lw=0.6, color="0.75")
    p, r, _ = skm.precision_recall_curve(Y.ravel(), prob.ravel())
    ap = skm.average_precision_score(Y, prob, average="micro")
    ax.plot(r, p, lw=2.2, ls="--", color="black", label=f"micro-average (AP={ap:.3f})")
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.05), xlabel="Recall", ylabel="Precision", title=title)
    ax.legend(loc="lower left", fontsize=7)
    save(fig, path_noext)


def plot_reliability(bins, ece, path_noext, title="Reliability diagram"):
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(5, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
    width = bins[0]["hi"] - bins[0]["lo"]
    acc = [b["accuracy"] if b["count"] else 0 for b in bins]
    conf = [b["confidence"] if b["count"] else c for b, c in zip(bins, centers)]
    ax.bar(centers, acc, width=width * 0.95, color="#4C72B0", edgecolor="white", label="Accuracy")
    ax.bar(centers, np.array(conf) - np.array(acc), bottom=acc, width=width * 0.95,
           color="#DD8452", alpha=0.45, edgecolor="none", label="Gap")
    ax.plot([0, 1], [0, 1], ls="--", color="0.3", lw=1, label="Perfect calibration")
    ax.set(ylabel="Accuracy", ylim=(0, 1), title=f"{title} (ECE = {ece:.4f})")
    ax.legend(loc="upper left", fontsize=8)
    axh.bar(centers, [b["count"] for b in bins], width=width * 0.95, color="0.5")
    axh.set(xlabel="Confidence", ylabel="Count", xlim=(0, 1))
    save(fig, path_noext)


def plot_per_class_bars(per_class, path_noext, title="Per-class metrics"):
    names = [short(r["class"]) for r in per_class]
    keys = ["precision", "recall", "specificity", "f1"]
    x = np.arange(len(names))
    w = 0.8 / len(keys)
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(names) + 2), 4))
    for i, k in enumerate(keys):
        ax.bar(x + (i - (len(keys) - 1) / 2) * w, [r[k] for r in per_class], w, label=k.capitalize())
    ax.set_xticks(x, names, rotation=45, ha="right")
    lo = min(min(r[k] for k in keys) for r in per_class)
    ax.set(ylim=(max(0, lo - 0.05), 1.0), ylabel="Score", title=title)
    ax.legend(ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    save(fig, path_noext)


def read_log(log_csv):
    with open(log_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    def col(k):
        out = []
        for r in rows:
            try:
                out.append(float(r[k]))
            except (KeyError, TypeError, ValueError):
                out.append(np.nan)
        return np.array(out)
    epochs = col("epoch")
    per_loop = [[float(v) for v in (r.get("val_acc_per_loop") or "").split()] for r in rows]
    return {"epoch": epochs, "lr": col("lr"), "train_loss": col("train_loss"),
            "train_acc": col("train_acc"), "val_loss": col("val_loss"),
            "val_acc": col("val_acc"), "per_loop": per_loop}


def plot_training_curves(log_csv, path_noext, best_epoch=None):
    L = read_log(log_csv)
    e = L["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    ax.plot(e, L["train_loss"], marker="o", ms=3, label="train (label-smoothed)")
    ax.plot(e, L["val_loss"], marker="o", ms=3, label="validation")
    ax.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(e, L["train_acc"], marker="o", ms=3, label="train")
    ax.plot(e, L["val_acc"], marker="o", ms=3, label="validation")
    ax.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax.legend()
    ax = axes[1, 0]
    ax.plot(e, L["lr"], color="tab:green")
    ax.set(title="Learning rate", xlabel="Epoch", ylabel="LR")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax = axes[1, 1]
    K = max((len(p) for p in L["per_loop"]), default=0)
    for k in range(K):
        ax.plot(e, [p[k] if len(p) > k else np.nan for p in L["per_loop"]],
                marker="o", ms=3, label=f"read out after pass {k + 1}")
    ax.set(title="Validation accuracy after each loop pass", xlabel="Epoch", ylabel="Accuracy")
    if K:
        ax.legend()
    if best_epoch:
        for a in axes.ravel():
            a.axvline(best_epoch, color="0.5", ls=":", lw=1)
    fig.tight_layout()
    save(fig, path_noext)


def plot_loop_ablation(rows, train_loops, path_noext):
    """rows: list of {"loops": k, "accuracy":…, "f1_macro":…, "ece":…, ...}"""
    ks = [r["loops"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for key, lab in (("accuracy", "Accuracy"), ("f1_macro", "Macro F1"),
                     ("balanced_accuracy", "Balanced acc."), ("mcc", "MCC")):
        ax[0].plot(ks, [r[key] for r in rows], marker="o", label=lab)
    ax[0].set(xlabel="Loop passes at inference (K)", ylabel="Score",
              title="Inference-time loop count")
    ax[0].legend(fontsize=8)
    for key, lab in (("brier", "Brier"), ("ece", "ECE"), ("log_loss", "Log loss")):
        ax[1].plot(ks, [r[key] for r in rows], marker="o", label=lab)
    ax[1].set(xlabel="Loop passes at inference (K)", ylabel="Lower is better",
              title="Calibration / probabilistic error")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.axvline(train_loops, color="0.5", ls=":", lw=1)
        a.set_xticks(ks)
    fig.tight_layout()
    save(fig, path_noext)


def plot_embedding(feats, y, class_names, path_noext, seed=0, title="t-SNE of pooled features"):
    from sklearn.manifold import TSNE
    feats = np.asarray(feats, dtype=np.float32)
    perp = max(2, min(30, (len(feats) - 1) // 3))
    emb = TSNE(n_components=2, perplexity=perp, init="pca", random_state=seed).fit_transform(feats)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.grid(False)
    colors = _class_colors(len(class_names))
    for c, name in enumerate(class_names):
        m = y == c
        if m.any():
            ax.scatter(emb[m, 0], emb[m, 1], s=8, color=colors[c], label=short(name), alpha=0.8)
    ax.set(title=title, xticks=[], yticks=[])
    if len(class_names) <= 20:
        ax.legend(fontsize=7, markerscale=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, path_noext)


def plot_deletion_insertion(fr, curves, path_noext):
    """curves: {"Grad-CAM": {"deletion": (N,S), "insertion": (N,S)}, "Random": {...}}"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, kind in enumerate(("deletion", "insertion")):
        for name, c in curves.items():
            m = np.asarray(c[kind]).mean(0)
            auc = np.trapezoid(m, fr)
            ax[i].plot(fr, m, lw=2, label=f"{name} (AUC={auc:.3f})")
        better = "lower" if kind == "deletion" else "higher"
        ax[i].set(xlabel=f"Fraction of patches {'removed' if kind == 'deletion' else 'revealed'}",
                  ylabel="Probability of predicted class", ylim=(0, 1.02),
                  title=f"{kind.capitalize()} curve ({better} AUC is better)")
        ax[i].legend()
    fig.tight_layout()
    save(fig, path_noext)
