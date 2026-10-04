# LoopViT: a Nanbeige-style looped Vision Transformer

This is a ViT for image classification that uses the Nanbeige4.2 looping idea. The model has one stack of **B distinct blocks** and runs it **K times**. The output of block B feeds straight back into block 1, so the weights are shared across passes.

```
patches + CLS + pos ─► [blk1 … blkB] ─► [blk1 … blkB] ─► … (K passes) ─► LN ─► head(CLS)
                         pass 1          pass 2 (same weights)
```

Default: B = 6, K = 2, dim 384. That is 12 block applications with only 6 blocks of weights, giving **11.0M params**. An untied ViT with 12 distinct blocks would have 21.7M params.

## Files
| file | what |
|---|---|
| `loop_vit.py` | model (`LoopViT`, `LoopViTConfig`), `print_model_summary` |
| `data.py` | loads an ImageFolder, with class selection and train/val split |
| `train.py` | training loop, driven by YAML plus command-line flags (resume, early stopping) |
| `evaluate.py` | full paper report for a run: metrics with 95% CIs, tables, figures, Grad-CAM, efficiency |
| `gradcam.py` | Grad-CAM heat maps (one per loop pass) for any images |
| `metrics.py`, `plots.py`, `explain.py` | metric computations, figures, Grad-CAM + deletion/insertion |
| `notes/` | architecture, training/evaluation notes, results so far |
| `tests/` | pytest suite on a tiny synthetic dataset (`python -m pytest tests`) |
| `predict.py` | runs a trained checkpoint on image files or folders |
| `downloads.py` | fetches and organizes public benchmark datasets into `datasets/` |
| `config.yaml` | every setting in one place |

## Data layout
```
data/train/<class_name>/*.jpg      # required
data/val/<class_name>/*.jpg        # optional; otherwise val_split of train is held out
```

## Benchmark datasets

`downloads.py` fetches five standard fine-grained classification datasets and
lays each one out as `datasets/<name>/{train,test}/<class>/*.jpg`, ready to
hand straight to `--train-dir` / `--val-dir`:

| `--dataset` | classes | source |
|---|---|---|
| `aircraft` | 100 (FGVC-Aircraft variants) | torchvision, automatic |
| `cub200` | 200 (Caltech-UCSD Birds-200-2011) | Caltech tarball, automatic |
| `flowers102` | 102 (Oxford Flowers-102) | torchvision, automatic |
| `food101` | 101 (Food-101) | torchvision, automatic |
| `cars` | 196 (Stanford Cars) | Kaggle mirror (see below) |

```bash
pip install torch torchvision torchinfo pyyaml scipy

# everything (several GB total; safe to Ctrl-C and re-run, already-done ones are skipped)
python downloads.py --dataset all

# just one or two
python downloads.py --dataset cub200 flowers102

# force real file copies instead of hardlinks (e.g. --root and the raw
# download cache end up on different drives)
python downloads.py --dataset food101 --copy
```

Stanford Cars' original ai.stanford.edu hosting is dead, so torchvision's own
downloader for it refuses to run. This repo falls back to the Kaggle CLI:

```bash
pip install kaggle
# create an API token at https://www.kaggle.com/settings ("Create New Token")
# and save the downloaded file to ~/.kaggle/kaggle.json
python downloads.py --dataset cars
# or point at a different mirror:
python downloads.py --dataset cars --kaggle-dataset <owner>/<dataset-slug>
```

Then train on any of them:
```bash
python train.py --config config.yaml \
    --train-dir datasets/cub200/train --val-dir datasets/cub200/test --num-classes 200

python train.py --config config.yaml \
    --train-dir datasets/food101/train --val-dir datasets/food101/test --num-classes 101
```

## Usage
```bash
pip install -r requirements.txt

# print the model summary only
python train.py --config config.yaml --summary-only --num-classes 10

# train
python train.py --config config.yaml --train-dir data/train --num-classes 5

# pick specific classes, cap the images per class, change the loop shape
python train.py --config config.yaml --classes cat dog horse --max-per-class 500 \
                --num-blocks 4 --num-loops 3

# resume an interrupted run (restores optimizer/scheduler/RNG/early-stopping state)
python train.py --config config.yaml --resume auto            # = <output_dir>/last.pt
python train.py --config config.yaml --resume runs/loopvit/last.pt

# early stopping + a held-out test split for reporting
python train.py --config config.yaml --early-stopping-patience 15 --test-split 0.1

# predict (you can also try a different loop count at inference)
python predict.py --ckpt runs/loopvit/best.pt --images some_folder/ --num-loops 2
```

Class options: `num_classes` (null means all folders), `class_selection` (first or random), `classes` (an explicit list), and `max_per_class`.

Each epoch logs the **validation accuracy read out after every pass** (`acc after each loop`). This shows whether the second pass helps.

## Paper reports (`evaluate.py`)

```bash
python evaluate.py --run-dir runs/loopvit                 # test split if the run has one, else val
python evaluate.py --run-dir runs/loopvit --split test --out-dir runs/loopvit/report_test
python gradcam.py --ckpt runs/loopvit/best.pt --images some_folder/ --out-dir cams/
```

The split is rebuilt from the run's `config.json` with the same seed, so the report uses exactly the
run's validation/test images. Written to `<run-dir>/report_<split>/`:

- **Scores** (`metrics.json`, `metrics_summary.csv`, `report.md`): accuracy, balanced accuracy,
  top-k, precision / recall (sensitivity) / F1 (macro, weighted, micro), specificity, NPV, MCC,
  Cohen's kappa, ROC-AUC (OvR macro/weighted, micro, OvO), PR-AUC, log loss, Brier score,
  ECE / MCE, all with 95% bootstrap confidence intervals.
- **Per class** (`per_class_metrics.csv`), `classification_report.txt`, `confusion_matrix.csv`, `predictions.csv`.
- **LaTeX tables**: `table_main.tex`, `table_per_class.tex`, `table_loops.tex`.
- **Figures** (`figures/`, PNG at 300 dpi plus PDF): confusion matrix (raw and normalized), ROC and PR
  curves, reliability diagram, per-class bars, training curves, t-SNE of features, and the
  **loop ablation** (scores when reading out after 1…2K passes).
- **Efficiency** (`efficiency.json`): parameters, untied-equivalent parameters, GFLOPs, latency, throughput, peak memory.
- **Grad-CAM** (`gradcam/`): per-class grids, misclassified images, and an overview. Each grid has one
  column per loop pass, because the hooked shared block runs once per pass.
- **Faithfulness** (`faithfulness.json`, figure): deletion/insertion AUC of Grad-CAM vs a random patch
  order, with a Wilcoxon test. This checks whether the highlighted regions really drive the prediction.

If you report the val split, remember it also picked `best.pt`. For a paper, train with `--test-split`
(or `--test-dir`) and report the test split.

## Command reference

```bash
pip install -r requirements.txt

# model summary only
python train.py --config config.yaml --summary-only --num-classes 5

# train
python train.py --config config.yaml --train-dir ./plantvillage --num-classes 5

# train for a paper: held-out test split + early stopping
python train.py --config config.yaml --train-dir ./plantvillage --num-classes 5                 --test-split 0.1 --early-stopping-patience 15

# continue an interrupted run (also works with older checkpoints)
python train.py --config config.yaml --train-dir ./plantvillage --num-classes 5                 --resume auto --early-stopping-patience 15
python train.py --config config.yaml --train-dir ./plantvillage --num-classes 5                 --resume runs/loopvit/last.pt

# full paper report (test split if the run has one, else val)
python evaluate.py --run-dir runs/loopvit
python evaluate.py --run-dir runs/loopvit --split test --ckpt runs/loopvit/best.pt                    --out-dir runs/loopvit/report_test
# faster: no bootstrap, no deletion/insertion test, no t-SNE
python evaluate.py --run-dir runs/loopvit --bootstrap 0 --faithfulness-samples 0 --no-tsne

# Grad-CAM for any images or folders (one heatmap per loop pass)
python gradcam.py --ckpt runs/loopvit/best.pt --images some_folder/ --out-dir cams/
python gradcam.py --ckpt runs/loopvit/best.pt --images leaf.jpg --target Apple___healthy --save-raw

# predict, optionally with a different loop count
python predict.py --ckpt runs/loopvit/best.pt --images some_folder/ --num-loops 2

# tests (synthetic data, about 1 minute on CPU)
python -m pytest tests -q
```

## Current results

PlantVillage, 5 classes, default model (11.02M params), `best.pt` at **epoch 7 of 100**
(the run was interrupted), val split of 467 images:

| Accuracy | Macro F1 | MCC | Cohen's κ | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---|---|---|---|---|---|
| 0.929 [0.904, 0.951] | 0.934 | 0.904 | 0.903 | 0.992 | 0.979 | 0.115 | 0.082 |

With 2 loop passes (as trained), accuracy is 92.9%. One pass gives 92.5%, 3 passes
92.1%, and 4 passes 90.8%. Cost: 9.2 GFLOPs per image, 7 ms per image on an RTX 3060.
The Grad-CAM maps for the healthy classes mostly highlight the background, and the val
split also picked `best.pt`. Train fully with `--test-split` before reporting.
Details, per-class numbers and open issues are in
[`notes/training_evaluation.md`](notes/training_evaluation.md). The architecture is
described in [`notes/architecture.md`](notes/architecture.md).

## Notes
- The default is the plain Nanbeige recipe. `loop_embedding: true` adds a learned vector per pass, which is not in Nanbeige.
- Stochastic depth rates increase over the *unrolled* depth (1 … B·K). Each application of a shared block therefore gets its own rate.
- In the torchinfo table, `(recursive)` rows mark a block running again on a later pass. Its parameters are counted once, but its compute is counted on every pass.
- Training a ViT from scratch on small datasets is hard. Use `augment: trivial`, more epochs, or smaller `image_size` / `dim` to help.
