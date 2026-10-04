import os
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# a tiny model so every test runs on CPU in seconds
TINY = ["--image-size", "32", "--patch-size", "8", "--dim", "32", "--num-blocks", "2",
        "--num-loops", "2", "--num-heads", "2", "--batch-size", "8", "--num-workers", "0",
        "--amp", "false", "--device", "cpu", "--warmup-epochs", "1", "--drop-path", "0.0"]


@pytest.fixture(scope="session")
def image_folder(tmp_path_factory):
    """3 classes told apart by their dominant colour, 20 images each."""
    root = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    for c, name in enumerate(["red_class", "green_class", "blue_class"]):
        d = root / name
        d.mkdir()
        for i in range(20):
            img = rng.integers(0, 90, (40, 40, 3)).astype(np.uint8)
            img[..., c] = rng.integers(160, 255, (40, 40))
            Image.fromarray(img).save(d / f"{i:03d}.png")
    return str(root)


def run(script, *args, check=True):
    proc = subprocess.run([sys.executable, os.path.join(ROOT, script), *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(f"{script} failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return proc
