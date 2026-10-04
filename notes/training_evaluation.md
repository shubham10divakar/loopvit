# Training, evaluation and results notes

See `architecture.md` for the model itself.

## Features

### Training (`train.py`)
- **Early stopping:** `--early-stopping-patience N` stops after N epochs with no
  improvement (0 = off). `--early-stopping-min-delta` sets the smallest change
  that counts. `--monitor val_acc|val_loss` picks the metric, which also decides
  which epoch is saved as `best.pt`.
- **Resume:** `--resume auto` continues from `<output_dir>/last.pt`, or starts
  fresh if there is none. You can also pass a checkpoint path. It restores the
  optimizer, LR schedule, AMP scaler, random state and early-stopping counter.
  Log rows past the resumed epoch are dropped.
- **Older checkpoints** (saved before `--resume` existed, with only model, epoch
  and val_acc) can still be resumed. The LR schedule position is rebuilt from the
  epoch number, but the optimizer's momentum starts from zero (a warning is
  printed). The current `runs/loopvit/last.pt` (epoch 7) is in this format.
- **Test split:** `--test-split 0.1` holds out 10% of each class as a test set,
  or use `--test-dir`. With the default of 0, the train/val split is identical
  to earlier versions.
- `train_summary.json` records the best epoch, best score, last epoch and
  whether training stopped early.

### Evaluation report (`evaluate.py`)
The data split is rebuilt from the run's `config.json` with the same seed, so the
report uses exactly the run's validation/test images. Output goes to
`<run-dir>/report_<split>/`:
- **Scores**, each with a 95% bootstrap confidence interval: accuracy, balanced
  accuracy, top-k, precision / recall (sensitivity) / F1 (macro, weighted and
  micro), specificity, NPV, MCC, Cohen's kappa, ROC-AUC (OvR macro/weighted,
  micro, OvO), PR-AUC, log loss, Brier score, ECE/MCE.
  Files: `metrics.json`, `metrics_summary.csv`, `report.md`.
- **Tables:** `per_class_metrics.csv`, `classification_report.txt`,
  `confusion_matrix.csv`, `predictions.csv` (one row per image), and LaTeX
  tables `table_main.tex`, `table_per_class.tex`, `table_loops.tex`.
- **Figures** (`figures/`, PNG at 300 dpi plus PDF): confusion matrix (raw and
  normalized), ROC and PR curves, reliability (calibration) diagram, per-class
  bars, training curves, t-SNE of features, and the loop ablation (scores when
  reading out after 1…2K passes).
- **Efficiency** (`efficiency.json`): parameters, untied-equivalent parameters,
  GFLOPs, latency, throughput, peak GPU memory.
- **Grad-CAM** (`gradcam/`): grids per class, misclassified images, and a
  one-per-class overview. Each row has one heatmap per loop pass, because the
  hooked shared block runs once per pass.
- **Grad-CAM check** (`faithfulness.json` plus a figure): the deletion/insertion
  test. Patches are removed or revealed in order of heatmap importance and
  compared with a random order, with a Wilcoxon test. Lower deletion AUC and
  higher insertion AUC than random means the highlighted regions really drive
  the prediction.

### Grad-CAM for any image (`gradcam.py`)
One figure per image: original | pass 1 | … | pass K. `--target <class>` explains
a chosen class instead of the predicted one. `--save-raw` also saves the maps as `.npy`.

### Code layout
| file | what |
|---|---|
| `metrics.py` | all metric computations, bootstrap CIs, LaTeX writers |
| `plots.py` | all figures |
| `explain.py` | `LoopViTGradCAM`, deletion/insertion test, Grad-CAM grid renderer |
| `tests/` | pytest suite on a tiny synthetic dataset (12 tests, about 1 minute on CPU) |

## Commands

```bash
pip install -r requirements.txt

# model summary only
python train.py --config config.yaml --summary-only --num-classes 5

# train (current PlantVillage 5-class setup)
python train.py --config config.yaml --train-dir ./plantvillage --num-classes 5

# train for the paper: held-out test split + early stopping
python train.py --config config.yaml --train-dir ./plantvillage --num-classes 5 \
                --test-split 0.1 --early-stopping-patience 15

# continue the interrupted run (epoch 7 of 100)
python train.py --config config.yaml --train-dir ./plantvillage --num-classes 5 \
                --resume auto --early-stopping-patience 15

# resume from a specific checkpoint
python train.py --config config.yaml --train-dir ./plantvillage --num-classes 5 \
                --resume runs/loopvit/last.pt

# full paper report (test split if the run has one, else val)
python evaluate.py --run-dir runs/loopvit

# choose the split, checkpoint and output folder explicitly
python evaluate.py --run-dir runs/loopvit --split test --ckpt runs/loopvit/best.pt \
                   --out-dir runs/loopvit/report_test

# faster report (no bootstrap, no deletion/insertion test, no t-SNE)
python evaluate.py --run-dir runs/loopvit --bootstrap 0 --faithfulness-samples 0 --no-tsne

# Grad-CAM for any images or folders
python gradcam.py --ckpt runs/loopvit/best.pt --images some_folder/ --out-dir cams/
python gradcam.py --ckpt runs/loopvit/best.pt --images leaf.jpg --target Apple___healthy --save-raw

# predict, optionally with a different loop count
python predict.py --ckpt runs/loopvit/best.pt --images some_folder/ --num-loops 2

# run the tests
python -m pytest tests -q
```

## Results so far (2026-10-04)

Run `runs/loopvit`: default LoopViT (B=6, K=2, dim 384, 11.02M params) trained
from scratch on 5 PlantVillage classes (Apple scab, Apple black rot, Apple cedar
rust, Apple healthy, Blueberry healthy). The run stopped after **7 of 100
epochs**. Evaluated `best.pt` (epoch 7) on the val split (467 images). The
rebuilt split reproduces the training log's 92.93% exactly.

| Metric | Value [95% CI] |
|---|---|
| Accuracy | 0.9293 [0.9036, 0.9507] |
| Balanced accuracy | 0.9235 [0.8880, 0.9518] |
| Macro F1 | 0.9344 [0.9071, 0.9571] |
| Specificity (macro) | 0.9796 [0.9720, 0.9861] |
| MCC | 0.9035 [0.8684, 0.9340] |
| Cohen's kappa | 0.9033 [0.8675, 0.9338] |
| ROC-AUC (OvR macro) | 0.9921 [0.9869, 0.9963] |
| PR-AUC (macro) | 0.9785 [0.9653, 0.9892] |
| Brier score | 0.1147 [0.0862, 0.1435] |
| Log loss | 0.2713 [0.2228, 0.3189] |
| ECE | 0.0820 [0.0743, 0.1057] |

Per class (F1): Apple scab 0.942, Black rot 0.911, Cedar rust 0.963,
Apple healthy 0.915, Blueberry healthy 0.941.

**Loop count at inference:**

| Passes | Accuracy | Macro F1 | ECE |
|---|---|---|---|
| 1 | 0.9251 | 0.9291 | 0.1038 |
| 2 (trained) | 0.9293 | 0.9344 | 0.0820 |
| 3 | 0.9208 | 0.9217 | 0.0780 |
| 4 | 0.9079 | 0.9088 | 0.0736 |

**Efficiency** (RTX 3060, fp32): 9.2 GFLOPs per image, 7.1 ms per image at
batch size 1, 560 images/s at batch size 64, 291 MB peak inference memory.

**Grad-CAM check** (100 images):

| Order | Deletion AUC (lower is better) | Insertion AUC (higher is better) |
|---|---|---|
| Grad-CAM, pass 1 | 0.411 | 0.690 |
| Grad-CAM, pass 2 | 0.412 | 0.690 |
| Random | 0.453 | 0.677 |

Wilcoxon: deletion p = 2.3e-7 (Grad-CAM beats random), insertion p = 0.70 (no difference).

## Open issues before the paper
1. **Background attention.** On both healthy classes the Grad-CAM maps mostly
   highlight the background, not the leaf. The disease classes partly highlight
   lesions and leaf edges. This fits PlantVillage's known background shortcut and
   the short training. Re-check after full training; if it persists, try
   `--augment trivial` or background-cropped images.
2. **Val numbers are slightly optimistic,** because the same val split chose
   `best.pt`. Retrain with `--test-split 0.1` and report the test split.
3. **Undertrained:** 7 of 100 epochs, about 6 minutes per epoch.
