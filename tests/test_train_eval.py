"""End-to-end: train / resume / early stopping / evaluate / gradcam CLIs on a tiny synthetic dataset."""
import csv
import json
import os

import torch

from conftest import TINY, run


def log_epochs(out):
    with open(os.path.join(out, "log.csv"), newline="") as f:
        return [int(r["epoch"]) for r in csv.DictReader(f)]


def test_resume_continues_where_it_stopped(image_folder, tmp_path):
    out = str(tmp_path / "run")
    run("train.py", "--train-dir", image_folder, "--output-dir", out, "--epochs", "2", *TINY)
    assert log_epochs(out) == [1, 2]
    p = run("train.py", "--train-dir", image_folder, "--output-dir", out, "--epochs", "4",
            "--resume", "auto", *TINY)
    assert "resumed at epoch 3" in p.stdout
    assert log_epochs(out) == [1, 2, 3, 4]
    ck = torch.load(os.path.join(out, "last.pt"), weights_only=False)
    assert ck["epoch"] == 4 and ck["step"] > 0 and ck["optimizer"] is not None


def test_resume_auto_without_checkpoint_starts_fresh(image_folder, tmp_path):
    out = str(tmp_path / "fresh")
    p = run("train.py", "--train-dir", image_folder, "--output-dir", out, "--epochs", "1",
            "--resume", "auto", *TINY)
    assert "starting fresh" in p.stdout
    assert log_epochs(out) == [1]


def test_resume_legacy_checkpoint(image_folder, tmp_path):
    """Checkpoints saved before --resume existed only had model/cfg/classes/epoch/val_acc."""
    out = str(tmp_path / "legacy")
    run("train.py", "--train-dir", image_folder, "--output-dir", out, "--epochs", "2", *TINY)
    ck = torch.load(os.path.join(out, "last.pt"), weights_only=False)
    legacy = {k: ck[k] for k in ("model", "model_cfg", "classes", "epoch", "val_acc")}
    torch.save(legacy, os.path.join(out, "last.pt"))
    p = run("train.py", "--train-dir", image_folder, "--output-dir", out, "--epochs", "3",
            "--resume", "auto", *TINY)
    assert "no optimizer state" in p.stdout
    assert "resumed at epoch 3" in p.stdout
    assert log_epochs(out) == [1, 2, 3]


def test_early_stopping(image_folder, tmp_path):
    out = str(tmp_path / "es")
    # lr 0 -> the model never changes -> no improvement after epoch 1
    p = run("train.py", "--train-dir", image_folder, "--output-dir", out, "--epochs", "20",
            "--lr", "0", "--min-lr", "0", "--early-stopping-patience", "2", *TINY)
    assert "early stopping" in p.stdout
    assert log_epochs(out) == [1, 2, 3]
    summ = json.load(open(os.path.join(out, "train_summary.json")))
    assert summ["stopped_early"] and summ["best_epoch"] == 1 and summ["last_epoch"] == 3
    # resuming a run that already stopped early does not train further
    p = run("train.py", "--train-dir", image_folder, "--output-dir", out, "--epochs", "20",
            "--lr", "0", "--min-lr", "0", "--early-stopping-patience", "2", "--resume", "auto", *TINY)
    assert "already triggered" in p.stdout
    assert log_epochs(out) == [1, 2, 3]


def test_evaluate_and_gradcam(image_folder, tmp_path):
    out = str(tmp_path / "evalrun")
    run("train.py", "--train-dir", image_folder, "--output-dir", out, "--epochs", "3",
        "--val-split", "0.2", "--test-split", "0.2", *TINY)
    p = run("evaluate.py", "--run-dir", out, "--num-workers", "0", "--device", "cpu",
            "--bootstrap", "20", "--faithfulness-samples", "6", "--gradcam-per-class", "2",
            "--batch-size", "8")
    rep = os.path.join(out, "report_test")          # auto picks the test split
    assert os.path.isdir(rep), p.stdout
    m = json.load(open(os.path.join(rep, "metrics.json")))
    assert m["split"] == "test" and m["n_images"] == 12   # 20% of 20 images x 3 classes
    for k in ("accuracy", "mcc", "brier", "roc_auc_ovr_macro", "ece", "cohen_kappa", "log_loss"):
        assert k in m["summary"]
    assert [r["loops"] for r in m["per_loop"]] == [1, 2, 3, 4]
    assert m["efficiency"]["total_params"] > 0 and m["efficiency"]["gflops_per_image"] > 0
    assert "Random" in m["faithfulness"]["methods"]
    for f in ("confusion_matrix", "confusion_matrix_normalized", "roc_curves", "pr_curves",
              "reliability_diagram", "per_class_metrics", "training_curves", "loop_ablation",
              "tsne_features", "faithfulness_deletion_insertion"):
        for ext in (".png", ".pdf"):
            assert os.path.exists(os.path.join(rep, "figures", f + ext)), f + ext
    for f in ("per_class_metrics.csv", "metrics_summary.csv", "predictions.csv", "per_loop_metrics.csv",
              "classification_report.txt", "table_main.tex", "table_per_class.tex", "table_loops.tex",
              "report.md", "efficiency.json", "faithfulness.json", "confusion_matrix.csv"):
        assert os.path.exists(os.path.join(rep, f)), f
    assert os.path.exists(os.path.join(rep, "gradcam", "overview_one_per_class.png")) or \
        m["summary"]["accuracy"] == 0
    with open(os.path.join(rep, "predictions.csv"), newline="") as f:
        assert len(list(csv.DictReader(f))) == 12

    # the test images never appear in train or val
    from data import split_samples
    _, tr, va, te = split_samples(image_folder, val_split=0.2, test_split=0.2, seed=42)
    assert not ({p for p, _ in te} & ({p for p, _ in tr} | {p for p, _ in va}))

    # standalone Grad-CAM CLI on a folder
    cams = str(tmp_path / "cams")
    folder = os.path.join(image_folder, "red_class")
    run("gradcam.py", "--ckpt", os.path.join(out, "best.pt"), "--images", folder,
        "--out-dir", cams, "--device", "cpu", "--save-raw")
    assert os.path.exists(os.path.join(cams, "000_gradcam.png"))
    assert os.path.exists(os.path.join(cams, "000_gradcam.npy"))
