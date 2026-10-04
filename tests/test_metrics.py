import numpy as np
from sklearn import metrics as skm

import metrics as M


def random_probs(n=300, c=4, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    logits = rng.normal(size=(n, c)) + 2.0 * np.eye(c)[y]
    p = np.exp(logits)
    return y, p / p.sum(1, keepdims=True)


def test_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    prob = np.eye(3)[y]
    s = M.compute_metrics(y, prob, ["a", "b", "c"])["summary"]
    for k in ("accuracy", "balanced_accuracy", "f1_macro", "mcc", "cohen_kappa",
              "roc_auc_ovr_macro", "pr_auc_macro", "specificity_macro"):
        assert abs(s[k] - 1.0) < 1e-9, k
    assert s["brier"] < 1e-12 and s["ece"] < 1e-12


def test_matches_sklearn():
    y, prob = random_probs()
    res = M.compute_metrics(y, prob, list("abcd"))
    s = res["summary"]
    pred = prob.argmax(1)
    assert np.isclose(s["accuracy"], skm.accuracy_score(y, pred))
    assert np.isclose(s["f1_macro"], skm.f1_score(y, pred, average="macro"))
    assert np.isclose(s["mcc"], skm.matthews_corrcoef(y, pred))
    assert np.isclose(s["roc_auc_ovr_macro"], skm.roc_auc_score(y, prob, multi_class="ovr"))
    assert np.isclose(s["log_loss"], skm.log_loss(y, prob))
    # multi-class Brier = mean squared distance to the one-hot label, summed over classes
    assert np.isclose(s["brier"], ((prob - np.eye(4)[y]) ** 2).sum(1).mean())
    cm = res["confusion_matrix"]
    assert cm.sum() == len(y)
    # specificity of class c = TN / (TN + FP)
    c = 2
    tn = cm.sum() - cm[c].sum() - cm[:, c].sum() + cm[c, c]
    fp = cm[:, c].sum() - cm[c, c]
    assert np.isclose(res["per_class"][c]["specificity"], tn / (tn + fp))


def test_binary_and_missing_class():
    y, prob = random_probs(c=2)
    s = M.compute_metrics(y, prob, ["neg", "pos"])["summary"]
    assert np.isclose(s["roc_auc_ovr_macro"], skm.roc_auc_score(y, prob[:, 1]))
    # class 2 never appears in y: AUC for it is undefined, the rest still works
    y3 = np.array([0, 1, 0, 1])
    p3 = np.array([[.8, .1, .1], [.2, .7, .1], [.6, .3, .1], [.1, .8, .1]])
    r = M.compute_metrics(y3, p3, ["a", "b", "c"])
    assert r["summary"]["accuracy"] == 1.0
    assert np.isnan(r["per_class"][2]["roc_auc"])


def test_bootstrap_and_latex():
    y, prob = random_probs()
    ci = M.bootstrap_ci(y, prob, n_boot=50)
    s = M.compute_metrics(y, prob, list("abcd"))["summary"]
    for k in M.HEADLINE:
        lo, hi = ci[k]
        assert lo <= hi
    assert ci["accuracy"][0] <= s["accuracy"] <= ci["accuracy"][1]
    tex = M.latex_summary_table(s, ci, "cap", "tab:x")
    assert "\\begin{tabular}" in tex and "MCC" in tex
