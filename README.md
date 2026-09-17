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
| `train.py` | training loop, driven by YAML plus command-line flags |
| `predict.py` | runs a trained checkpoint on image files or folders |
| `config.yaml` | every setting in one place |

## Data layout
```
data/train/<class_name>/*.jpg      # required
data/val/<class_name>/*.jpg        # optional; otherwise val_split of train is held out
```

## Usage
```bash
pip install torch torchvision torchinfo pyyaml

# print the model summary only
python train.py --config config.yaml --summary-only --num-classes 10

# train
python train.py --config config.yaml --train-dir data/train --num-classes 5

# pick specific classes, cap the images per class, change the loop shape
python train.py --config config.yaml --classes cat dog horse --max-per-class 500 \
                --num-blocks 4 --num-loops 3

# predict (you can also try a different loop count at inference)
python predict.py --ckpt runs/loopvit/best.pt --images some_folder/ --num-loops 2
```

Class options: `num_classes` (null means all folders), `class_selection` (first or random), `classes` (an explicit list), and `max_per_class`.

Each epoch logs the **validation accuracy read out after every pass** (`acc after each loop`). This shows whether the second pass helps.

## Notes
- The default is the plain Nanbeige recipe. `loop_embedding: true` adds a learned vector per pass, which is not in Nanbeige.
- Stochastic depth rates increase over the *unrolled* depth (1 … B·K). Each application of a shared block therefore gets its own rate.
- In the torchinfo table, `(recursive)` rows mark a block running again on a later pass. Its parameters are counted once, but its compute is counted on every pass.
- Training a ViT from scratch on small datasets is hard. Use `augment: trivial`, more epochs, or smaller `image_size` / `dim` to help.
