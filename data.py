"""
Image-folder data pipeline.

Expected layout (standard torchvision ImageFolder):

    train_dir/
        class_a/  img1.jpg  img2.png ...
        class_b/  ...
    val_dir/      (optional, same class sub-folders)

Configurable:
  * num_classes      - use only N of the class folders (None = all)
  * class_selection  - "first" (alphabetical) or "random" (seeded) when N < total
  * classes          - explicit list of folder names (overrides the two above)
  * max_per_class    - cap images per class (None = all)
  * val_split        - fraction of train held out per class when no val_dir
  * test_split       - fraction of train held out per class as a test set when
                       no test_dir (default 0 = no test set; with 0 the
                       train/val split is identical to earlier versions)
"""
from __future__ import annotations

import random
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SampleListDataset(Dataset):
    def __init__(self, samples, classes, transform=None):
        self.samples = samples          # list of (path, label)
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = default_loader(path)      # PIL RGB
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transforms(image_size: int, augment: str = "basic"):
    resize = int(round(image_size / 0.875))
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = [transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
             transforms.RandomHorizontalFlip()]
    if augment == "trivial":
        train.append(transforms.TrivialAugmentWide())
    elif augment == "none":
        train = [transforms.Resize((image_size, image_size))]
    train += [transforms.ToTensor(), norm]
    evalt = [transforms.Resize(resize), transforms.CenterCrop(image_size),
             transforms.ToTensor(), norm]
    return transforms.Compose(train), transforms.Compose(evalt)


def _select_classes(all_classes, num_classes, class_selection, classes, seed):
    if classes:
        missing = [c for c in classes if c not in all_classes]
        if missing:
            raise ValueError(f"Classes not found in train folder: {missing}")
        return list(classes)
    if num_classes is None or num_classes >= len(all_classes):
        if num_classes is not None and num_classes > len(all_classes):
            print(f"[data] asked for {num_classes} classes, folder has "
                  f"{len(all_classes)} - using all.")
        return list(all_classes)
    if class_selection == "random":
        return sorted(random.Random(seed).sample(list(all_classes), num_classes))
    return list(all_classes)[:num_classes]


def _group(samples, idx_to_name, keep, max_per_class, rng):
    by_class = defaultdict(list)
    for path, idx in samples:
        name = idx_to_name[idx]
        if name in keep:
            by_class[name].append(path)
    for name in by_class:
        rng.shuffle(by_class[name])
        if max_per_class:
            by_class[name] = by_class[name][:max_per_class]
    return by_class


def split_samples(train_dir, val_dir=None, test_dir=None, num_classes=None,
                  class_selection="first", classes=None, max_per_class=None,
                  val_split=0.1, test_split=0.0, seed=42):
    """Deterministic class selection + train/val/test split.

    Returns (classes, train_samples, val_samples, test_samples), each sample a
    (path, label) pair. Re-running with the same arguments gives the same split,
    which is how evaluate.py recovers the exact validation/test images of a run.
    """
    rng = random.Random(seed)
    base = datasets.ImageFolder(train_dir)
    chosen = _select_classes(base.classes, num_classes, class_selection, classes, seed)
    name_to_label = {c: i for i, c in enumerate(chosen)}
    idx_to_name = {i: c for c, i in base.class_to_idx.items()}

    train_by_class = _group(base.samples, idx_to_name, set(chosen), max_per_class, rng)
    empty = [c for c in chosen if not train_by_class.get(c)]
    if empty:
        raise ValueError(f"No images found for classes: {empty}")

    def from_dir(d):
        vbase = datasets.ImageFolder(d)
        vidx_to_name = {i: c for c, i in vbase.class_to_idx.items()}
        by_class = _group(vbase.samples, vidx_to_name, set(chosen), None, rng)
        return [(p, name_to_label[c]) for c in chosen for p in by_class.get(c, [])]

    val_samples = from_dir(val_dir) if val_dir else []
    test_samples = from_dir(test_dir) if test_dir else []
    train_samples = []
    # stratified hold-out: at least one image per class per held-out split when possible
    for c in chosen:
        paths = train_by_class[c]
        n_val = 0
        if not val_dir and val_split > 0:
            n_val = int(round(len(paths) * val_split))
            if len(paths) > 1:
                n_val = max(1, n_val)
        n_test = 0
        if not test_dir and test_split > 0:
            n_test = int(round(len(paths) * test_split))
            if len(paths) - n_val > 1:
                n_test = max(1, n_test)
        val_samples += [(p, name_to_label[c]) for p in paths[:n_val]]
        test_samples += [(p, name_to_label[c]) for p in paths[n_val:n_val + n_test]]
        train_samples += [(p, name_to_label[c]) for p in paths[n_val + n_test:]]
    return chosen, train_samples, val_samples, test_samples


def make_eval_loader(samples, classes, image_size, batch_size=64, num_workers=4,
                     pin_memory=True):
    _, eval_tf = build_transforms(image_size)
    ds = SampleListDataset(samples, classes, eval_tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=pin_memory, persistent_workers=num_workers > 0)


def build_dataloaders(train_dir, val_dir=None, image_size=224, batch_size=64,
                      num_workers=4, num_classes=None, class_selection="first",
                      classes=None, max_per_class=None, val_split=0.1,
                      augment="basic", seed=42, pin_memory=True,
                      test_dir=None, test_split=0.0):
    """Returns (train_loader, val_loader, test_loader, classes); the val/test
    loaders are None when that split is empty."""
    train_tf, _ = build_transforms(image_size, augment)
    chosen, train_samples, val_samples, test_samples = split_samples(
        train_dir, val_dir, test_dir, num_classes, class_selection, classes,
        max_per_class, val_split, test_split, seed)

    train_ds = SampleListDataset(train_samples, chosen, train_tf)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=len(train_ds) > batch_size, generator=g,
                              persistent_workers=num_workers > 0)
    val_loader = (make_eval_loader(val_samples, chosen, image_size, batch_size,
                                   num_workers, pin_memory) if val_samples else None)
    test_loader = (make_eval_loader(test_samples, chosen, image_size, batch_size,
                                    num_workers, pin_memory) if test_samples else None)

    counts = defaultdict(int)
    for _, y in train_samples:
        counts[y] += 1
    print(f"[data] {len(chosen)} classes from {train_dir}")
    print(f"[data] train images: {len(train_ds)} | val images: {len(val_samples)}"
          f" ({'from ' + val_dir if val_dir else f'{val_split:.0%} split of train'})"
          f" | test images: {len(test_samples)}"
          f"{' (from ' + test_dir + ')' if test_dir else (f' ({test_split:.0%} split of train)' if test_samples else '')}")
    preview = ", ".join(f"{c}={counts[i]}" for i, c in enumerate(chosen[:10]))
    print(f"[data] per-class train counts: {preview}{' ...' if len(chosen) > 10 else ''}")
    return train_loader, val_loader, test_loader, chosen
