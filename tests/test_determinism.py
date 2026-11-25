import pytest
import torch

from wcsimfw import determinism as det


def test_skill_share_ends():
    y = torch.tensor([1.0, 0.0, 1.0, 0.0])
    assert det.skill_share(y, y)[0] == 1  # known before the match
    assert det.skill_share(torch.full((4,), 0.5), y)[0] == 0  # a coin
    g = torch.Generator().manual_seed(0)
    p = torch.rand(200_000, generator=g).double()
    share, se = det.skill_share(p, torch.bernoulli(p, generator=g))
    assert abs(share - 1 / 3) < 4 * se  # uniform p: Var(p) / Var(y) = (1/12) / (1/4)


def test_series():
    assert det.series(0.6, 1) == pytest.approx(0.6)
    assert det.series(0.6, 3) == pytest.approx(0.6**2 * (1 + 2 * 0.4))
    assert det.series(torch.tensor(0.5), 7).item() == pytest.approx(0.5)


def test_openness_of_a_cup():
    p = torch.full((16,), 1 / 16, dtype=torch.float64)
    assert det.contenders(p).item() == pytest.approx(16) and det.tournament_share(p) == 0
    sure = torch.eye(16, dtype=torch.float64)[0]
    assert det.contenders(sure) == 1 and det.tournament_share(sure) == 1
