"""
Classification metrics for the paper tables.

Everything works on plain numpy arrays:
    y     (N,)    integer labels
    prob  (N, C)  predicted class probabilities (softmax output)

compute_metrics() returns a flat dict of summary scores, a per-class table and
the confusion matrix. bootstrap_ci() gives percentile confidence intervals.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn import metrics as skm

# scores that go into the main results table (and get bootstrap CIs)
HEADLINE = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
            "f1_macro", "f1_weighted", "specificity_macro", "mcc", "cohen_kappa",
            "roc_auc_ovr_macro", "pr_auc_macro", "brier", "log_loss", "ece"]

# lower is better for these; everything else is higher-is-better
LOWER_IS_BETTER = {"brier", "log_loss", "ece", "mce", "error_rate"}


def one_hot(y: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(y), n), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def _safe(fn, *a, **k) -> float:
    """sklearn raises when a class is missing from y (e.g. in a bootstrap
    resample or a tiny val set); report NaN instead of crashing."""
    try:
        return float(fn(*a, **k))
    except ValueError:
        return float("nan")


def calibration(prob: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """Expected / maximum calibration error over equal-width confidence bins.
    Returns (ece, mce, bins) where bins is a list of dicts for the reliability
    diagram."""
    conf = prob.max(1)
    correct = (prob.argmax(1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce, bins = 0.0, 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        cnt = int(m.sum())
        acc = float(correct[m].mean()) if cnt else float("nan")
        avg_conf = float(conf[m].mean()) if cnt else float("nan")
        if cnt:
            gap = abs(acc - avg_conf)
            ece += cnt / len(y) * gap
            mce = max(mce, gap)
        bins.append({"lo": float(lo), "hi": float(hi), "count": cnt,
                     "accuracy": acc, "confidence": avg_conf})
    return float(ece), float(mce), bins


def brier_score(prob: np.ndarray, y: np.ndarray) -> float:
    """Multi-class Brier score: mean over samples of sum_c (p_c - y_c)^2.
    Ranges 0 (perfect) to 2."""
    return float(((prob - one_hot(y, prob.shape[1])) ** 2).sum(1).mean())


def roc_auc(y: np.ndarray, prob: np.ndarray, average: str = "macro",
            multi_class: str = "ovr") -> float:
    n = prob.shape[1]
    if n == 2:
        return _safe(skm.roc_auc_score, y, prob[:, 1])
    if average == "micro":
        return _safe(skm.roc_auc_score, one_hot(y, n), prob, average="micro")
    return _safe(skm.roc_auc_score, y, prob, average=average,
                 multi_class=multi_class, labels=list(range(n)))


def compute_metrics(y: np.ndarray, prob: np.ndarray, class_names, n_bins: int = 15):
    y = np.asarray(y, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    n = prob.shape[1]
    labels = list(range(n))
    pred = prob.argmax(1)
    Y = one_hot(y, n)

    cm = skm.confusion_matrix(y, pred, labels=labels)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        specificity = np.where(tn + fp > 0, tn / (tn + fp), np.nan)
        npv = np.where(tn + fn > 0, tn / (tn + fn), np.nan)

    ece, mce, bins = calibration(prob, y, n_bins)
    s = {"n_samples": int(len(y)), "n_classes": int(n)}
    s["accuracy"] = float(skm.accuracy_score(y, pred))
    s["error_rate"] = 1.0 - s["accuracy"]
    s["balanced_accuracy"] = float(skm.balanced_accuracy_score(y, pred))
    for k in (2, 3, 5):
        if k < n:
            s[f"top{k}_accuracy"] = _safe(skm.top_k_accuracy_score, y, prob, k=k, labels=labels)
    for avg in ("macro", "weighted", "micro"):
        p, r, f, _ = skm.precision_recall_fscore_support(
            y, pred, labels=labels, average=avg, zero_division=0)
        s[f"precision_{avg}"], s[f"recall_{avg}"], s[f"f1_{avg}"] = float(p), float(r), float(f)
    s["sensitivity_macro"] = s["recall_macro"]
    s["specificity_macro"] = float(np.nanmean(specificity))
    s["npv_macro"] = float(np.nanmean(npv))
    s["mcc"] = float(skm.matthews_corrcoef(y, pred))
    s["cohen_kappa"] = float(skm.cohen_kappa_score(y, pred, labels=labels))
    s["roc_auc_ovr_macro"] = roc_auc(y, prob, "macro", "ovr")
    s["roc_auc_ovr_weighted"] = roc_auc(y, prob, "weighted", "ovr")
    s["roc_auc_micro"] = roc_auc(y, prob, "micro")
    if n > 2:
        s["roc_auc_ovo_macro"] = roc_auc(y, prob, "macro", "ovo")
    present = Y.sum(0) > 0
    s["pr_auc_macro"] = (_safe(skm.average_precision_score, Y[:, present], prob[:, present], average="macro")
                         if present.any() else float("nan"))
    s["pr_auc_micro"] = _safe(skm.average_precision_score, Y, prob, average="micro")
    s["log_loss"] = _safe(skm.log_loss, y, prob, labels=labels)
    s["brier"] = brier_score(prob, y)
    s["ece"], s["mce"] = ece, mce

    p_c, r_c, f_c, sup = skm.precision_recall_fscore_support(
        y, pred, labels=labels, average=None, zero_division=0)
    per_class = []
    for c in range(n):
        has_both = 0 < Y[:, c].sum() < len(y)
        per_class.append({
            "class": class_names[c],
            "support": int(sup[c]),
            "precision": float(p_c[c]),
            "recall": float(r_c[c]),
            "specificity": float(specificity[c]),
            "f1": float(f_c[c]),
            "npv": float(npv[c]),
            "roc_auc": _safe(skm.roc_auc_score, Y[:, c], prob[:, c]) if has_both else float("nan"),
            "pr_auc": _safe(skm.average_precision_score, Y[:, c], prob[:, c]) if has_both else float("nan"),
            "brier": float(((prob[:, c] - Y[:, c]) ** 2).mean()),
            "tp": int(tp[c]), "fp": int(fp[c]), "fn": int(fn[c]), "tn": int(tn[c]),
        })
    return {"summary": s, "per_class": per_class, "confusion_matrix": cm,
            "calibration_bins": bins}


def _fast_scores(y, prob):
    """The HEADLINE scores only, for bootstrap resampling."""
    n = prob.shape[1]
    labels = list(range(n))
    pred = prob.argmax(1)
    Y = one_hot(y, n)
    cm = skm.confusion_matrix(y, pred, labels=labels)
    tp = np.diag(cm).astype(np.float64)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        spec = np.nanmean(np.where(tn + fp > 0, tn / (tn + fp), np.nan))
    p, r, f, _ = skm.precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    present = Y.sum(0) > 0
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y, pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "f1_weighted": float(skm.f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "specificity_macro": float(spec),
        "mcc": float(skm.matthews_corrcoef(y, pred)),
        "cohen_kappa": float(skm.cohen_kappa_score(y, pred, labels=labels)),
        "roc_auc_ovr_macro": roc_auc(y, prob, "macro", "ovr") if present.all() else float("nan"),
        "pr_auc_macro": _safe(skm.average_precision_score, Y[:, present], prob[:, present], average="macro"),
        "brier": brier_score(prob, y),
        "log_loss": _safe(skm.log_loss, y, prob, labels=labels),
        "ece": calibration(prob, y)[0],
    }


def bootstrap_ci(y, prob, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap (resampling test images with replacement) for the
    HEADLINE scores. Returns {metric: (low, high)}."""
    y = np.asarray(y)
    prob = np.asarray(prob, dtype=np.float64)
    if n_boot <= 0 or len(y) < 2:
        return {}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in HEADLINE}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        for k, v in _fast_scores(y[idx], prob[idx]).items():
            draws[k].append(v)
    out = {}
    for k, v in draws.items():
        v = np.asarray(v, dtype=np.float64)
        v = v[~np.isnan(v)]
        out[k] = ((float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))
                  if len(v) else (float("nan"), float("nan")))
    return out


# --------------------------------------------------------------------------- #
# Table writers
# --------------------------------------------------------------------------- #
PRETTY = {
    "accuracy": "Accuracy", "balanced_accuracy": "Balanced acc.",
    "precision_macro": "Precision (macro)", "recall_macro": "Recall / Sensitivity (macro)",
    "f1_macro": "F1 (macro)", "f1_weighted": "F1 (weighted)",
    "specificity_macro": "Specificity (macro)", "mcc": "MCC", "cohen_kappa": "Cohen's $\\kappa$",
    "roc_auc_ovr_macro": "ROC-AUC (OvR macro)", "pr_auc_macro": "PR-AUC (macro)",
    "brier": "Brier score", "log_loss": "Log loss", "ece": "ECE",
}


def fmt(v, digits=4):
    return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"


def latex_summary_table(summary, ci, caption, label):
    rows = []
    for k in HEADLINE:
        arrow = "$\\downarrow$" if k in LOWER_IS_BETTER else "$\\uparrow$"
        val = fmt(summary.get(k))
        if k in ci:
            lo, hi = ci[k]
            val += f" [{fmt(lo)}, {fmt(hi)}]"
        rows.append(f"{PRETTY[k]} {arrow} & {val} \\\\")
    return ("\\begin{table}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\begin{tabular}{lc}\n\\toprule\n"
            "Metric & Value [95\\% CI] \\\\\n\\midrule\n"
            + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


def latex_per_class_table(per_class, caption, label):
    def esc(t):
        return str(t).replace("_", "\\_").replace("&", "\\&")
    rows = [f"{esc(r['class'])} & {r['support']} & {fmt(r['precision'], 3)} & {fmt(r['recall'], 3)} & "
            f"{fmt(r['specificity'], 3)} & {fmt(r['f1'], 3)} & {fmt(r['roc_auc'], 3)} \\\\"
            for r in per_class]
    return ("\\begin{table}[t]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            "\\begin{tabular}{lrccccc}\n\\toprule\n"
            "Class & $n$ & Precision & Recall & Specificity & F1 & ROC-AUC \\\\\n\\midrule\n"
            + "\n".join(rows) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
