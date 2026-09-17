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


def build_dataloaders(train_dir, val_dir=None, image_size=224, batch_size=64,
                      num_workers=4, num_classes=None, class_selection="first",
                      classes=None, max_per_class=None, val_split=0.1,
                      augment="basic", seed=42, pin_memory=True):
    rng = random.Random(seed)
    train_tf, eval_tf = build_transforms(image_size, augment)

    base = datasets.ImageFolder(train_dir)
    chosen = _select_classes(base.classes, num_classes, class_selection, classes, seed)
    name_to_label = {c: i for i, c in enumerate(chosen)}
    idx_to_name = {i: c for c, i in base.class_to_idx.items()}

    train_by_class = _group(base.samples, idx_to_name, set(chosen), max_per_class, rng)
    empty = [c for c in chosen if not train_by_class.get(c)]
    if empty:
        raise ValueError(f"No images found for classes: {empty}")

    train_samples, val_samples = [], []
    if val_dir:
        vbase = datasets.ImageFolder(val_dir)
        vidx_to_name = {i: c for c, i in vbase.class_to_idx.items()}
        val_by_class = _group(vbase.samples, vidx_to_name, set(chosen), None, rng)
        for c in chosen:
            train_samples += [(p, name_to_label[c]) for p in train_by_class[c]]
            val_samples += [(p, name_to_label[c]) for p in val_by_class.get(c, [])]
    else:
        # stratified hold-out: at least one val image per class when possible
        for c in chosen:
            paths = train_by_class[c]
            n_val = int(round(len(paths) * val_split)) if val_split > 0 else 0
            if val_split > 0 and len(paths) > 1:
                n_val = max(1, n_val)
            val_samples += [(p, name_to_label[c]) for p in paths[:n_val]]
            train_samples += [(p, name_to_label[c]) for p in paths[n_val:]]

    train_ds = SampleListDataset(train_samples, chosen, train_tf)
    val_ds = SampleListDataset(val_samples, chosen, eval_tf) if val_samples else None

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              drop_last=len(train_ds) > batch_size, generator=g,
                              persistent_workers=num_workers > 0)
    val_loader = (DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=num_workers > 0)
                  if val_ds else None)

    counts = defaultdict(int)
    for _, y in train_samples:
        counts[y] += 1
    print(f"[data] {len(base.classes)} class folders in {train_dir}; using {len(chosen)}")
    print(f"[data] train images: {len(train_ds)} | val images: {len(val_samples)}"
          f" ({'from ' + val_dir if val_dir else f'{val_split:.0%} split of train'})")
    preview = ", ".join(f"{c}={counts[i]}" for i, c in enumerate(chosen[:10]))
    print(f"[data] per-class train counts: {preview}{' ...' if len(chosen) > 10 else ''}")
    return train_loader, val_loader, chosen
