import polars as pl
import pytest
import torch

from wcsimfw import chain, run
from wcsimfw.pipelines import possessions


def test_goal_per_possession_by_hand():
    # all logits 0: from each zone 1/5 to B, M, F, shot, loss; goal per shot 1/2; first touch
    # 1/4 each. P(goal | zone) = h solves h = 3/5 h + 1/5 * 1/2, so h = 1/4.
    p = chain.goal_per_possession(torch.zeros(chain.SIZE, dtype=torch.float64), torch.tensor(0.0))
    assert p.item() == pytest.approx(3 / 4 * 1 / 4 + 1 / 4 * 1 / 2)


def expected_counts(theta, gaps, n=1e6):
    """Moves and first touches in expectation (weights) for a known chain."""
    moves = {"gap": [], "from": [], "to": [], "n": []}
    firsts = {"gap": [], "state": [], "n": []}
    for g in gaps:
        p = chain.probabilities(theta, torch.tensor(g))
        for z in range(3):
            for k in range(5):
                for key, v in zip(moves, (g, z, k, n * p["move"][z, k].item()), strict=True):
                    moves[key].append(v)
        for k, v in ((5, p["goal"].item()), (6, 1 - p["goal"].item())):
            for key, x in zip(moves, (g, 3, k, n * v), strict=True):
                moves[key].append(x)
        for s in range(4):
            for key, x in zip(firsts, (g, s, n * p["first"][s].item()), strict=True):
                firsts[key].append(x)
    as_t = {"gap": torch.float64, "n": torch.float64}
    return ({k: torch.tensor(v, dtype=as_t.get(k, torch.long)) for k, v in moves.items()},
            {k: torch.tensor(v, dtype=as_t.get(k, torch.long)) for k, v in firsts.items()})


def test_fit_recovers_a_known_chain():
    g = torch.Generator().manual_seed(3)
    theta = torch.randn(chain.SIZE, generator=g, dtype=torch.float64) * 0.5
    est, _ = chain.fit(*expected_counts(theta, [-300.0, 0.0, 300.0]))
    assert torch.allclose(est, theta, atol=1e-3)


def test_pairs_equal_sides_and_symmetry():
    theta = torch.zeros(chain.SIZE, dtype=torch.float64)
    theta[25] = 0.5  # stronger side finishes better
    full, adv = chain.pairs(torch.tensor([1800.0, 2000.0], dtype=torch.float64), theta, n=4.0)
    assert torch.allclose(full[0, 1], full[1, 0].T)
    assert adv[1, 0] > 0.5 > adv[0, 1] and adv[0, 1] + adv[1, 0] == pytest.approx(1)


def test_touches_split_possessions_and_stop_at_shot():
    e = pl.DataFrame({
        "match_id": [1] * 7, "period": ["FirstHalf"] * 7,
        "team": ["A", "A", "A", "B", "B", "B", "A"],
        "x": [10.0, 50.0, 90.0, 20.0, 80.0, 85.0, 30.0],
        "is_touch": [True] * 7, "is_shot": [False, False, True, False, False, False, False],
        "is_goal": [False, False, True, False, False, False, False]})
    t = possessions.touches(e)
    assert t["state"].to_list() == ["B", "M", "S", "B", "F", "F", "B"]
    assert t["to"].to_list() == ["M", "S", "goal", "F", "F", "loss", "loss"]
    assert t["poss"].n_unique() == 3


def test_chain_config_rules():
    fit = {"kind": "fitted", "source": "x"}
    run.check({"model": {"name": "possession_chain", "elo": "history", "params": {"chain": fit}}})
    with pytest.raises(ValueError, match="history"):
        run.check({"model": {"name": "possession_chain", "params": {"chain": fit}}})
    with pytest.raises(ValueError, match="possession_chain"):
        run.check({"model": {"name": "elo_poisson", "params": {"chain": fit}}})
