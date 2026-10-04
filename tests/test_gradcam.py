import numpy as np
import torch

from explain import LoopViTGradCAM, deletion_insertion
from loop_vit import build_model


def tiny(num_loops=3):
    torch.manual_seed(0)
    m = build_model(image_size=32, patch_size=8, dim=32, num_blocks=2, num_loops=num_loops,
                    num_heads=2, num_classes=3, drop_path=0.0)
    # the head is zero-initialised; give it weights so gradients are non-trivial
    torch.nn.init.normal_(m.head.weight, std=0.5)
    return m.eval()


def test_one_cam_per_pass():
    m = tiny(num_loops=3)
    x = torch.randn(4, 3, 32, 32)
    logits, target, cams = LoopViTGradCAM(m)(x)
    assert len(cams) == 3                        # one map per loop pass
    assert torch.equal(target, logits.argmax(1))
    for c in cams:
        assert c.shape == (4, 32, 32)
        assert c.min() >= 0 and c.max() <= 1 + 1e-6
        assert (c.flatten(1).max(1)[0] > 0).all()  # not all-zero
    # the hook must be removed and the grads cleared afterwards
    assert not m.blocks[-1].norm1._forward_hooks
    assert all(p.grad is None for p in m.parameters())


def test_num_loops_override_and_target():
    m = tiny(num_loops=2)
    x = torch.randn(2, 3, 32, 32)
    _, target, cams = LoopViTGradCAM(m)(x, target=torch.tensor([1, 2]), num_loops=4)
    assert len(cams) == 4
    assert target.tolist() == [1, 2]


def test_deletion_insertion_curves():
    m = tiny()
    x = torch.randn(3, 3, 32, 32)
    _, target, cams = LoopViTGradCAM(m)(x)
    fr, d, ins = deletion_insertion(m, x, target, cams[-1], patch_size=8, steps=4)
    assert fr[0] == 0 and fr[-1] == 1
    assert d.shape == ins.shape == (3, len(fr))
    with torch.no_grad():
        p = m(x).softmax(-1).gather(1, target[:, None])[:, 0].numpy()
    # nothing removed = original prediction; everything revealed = original prediction
    assert np.allclose(d[:, 0], p, atol=1e-5)
    assert np.allclose(ins[:, -1], p, atol=1e-5)
    # everything deleted = the blank (all-zero, i.e. dataset-mean) image
    with torch.no_grad():
        p_blank = m(torch.zeros_like(x)).softmax(-1).gather(1, target[:, None])[:, 0].numpy()
    assert np.allclose(d[:, -1], p_blank, atol=1e-5)
