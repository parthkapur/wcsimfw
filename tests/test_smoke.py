import torch

import wcsimfw


def test_import_and_device():
    assert wcsimfw.__version__
    assert torch.zeros(2, device="cpu").sum().item() == 0
