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


def test_touches_split_possessions():
    # A builds and shoots; A keeps the rebound with two touches, B has one touch (a block) and A
    # goes on; B takes the ball with two touches; A's own goal (marked as a shot) is no touch.
    e = pl.DataFrame({
        "match_id": [1] * 11, "period": ["FirstHalf"] * 11, "team": list("AAAAABAABBA"),
        "x": [10.0, 50.0, 90.0, 70.0, 60.0, 30.0, 75.0, 80.0, 20.0, 40.0, 5.0],
        "is_touch": [True] * 11, "is_shot": [False, False, True] + [False] * 7 + [True],
        "is_goal": [False] * 10 + [True], "own_goal": [None] * 10 + [True]})
    t = possessions.touches(e)
    assert "".join(t["team"]) == "AAAAAAABB"
    assert t["state"].to_list() == ["B", "M", "S", "F", "M", "F", "F", "B", "M"]
    assert t["to"].to_list() == ["M", "S", "miss", "M", "F", "F", "loss", "M", "loss"]
    assert t["poss"].n_unique() == 3


def test_style_pulls_toward_the_rating():
    theta = torch.zeros(chain.SIZE, dtype=torch.float64)
    matches = pl.DataFrame({"match_id": [1], "home": ["A"], "away": ["B"], "gap": [0.0]})
    moves = pl.DataFrame({"match_id": [1, 1], "team": ["A", "B"], "from": ["M", "M"],
                          "to": ["F", "loss"], "n": [100, 100]})
    starts = pl.DataFrame([{"match_id": 1, "team": t, "state": z, "n": 25}
                           for t in "AB" for z in chain.FIRST])
    s = chain.style(theta, moves, starts, matches, ["A", "B", "C"], prior=1.0)
    m = chain.ZONES.index("M")
    # expected 100 / 5 = 20 moves to F, seen 100, prior worth one match (100): (100+100)/(100+20)
    assert s["move_for"][0, m, 2].item() == pytest.approx(200 / 120)
    assert s["move_against"][1, m, 2].item() == pytest.approx(200 / 120)
    assert (s["move_for"][2] == 1).all() and (s["first_against"][2] == 1).all()
    elo = torch.tensor([1800.0, 1800.0, 1800.0], dtype=torch.float64)
    r = chain.rates(elo, theta, 4.0, style=s)
    plain = chain.rates(elo, theta, 4.0)
    assert r[0, 2] > plain[0, 2] and r[2, 2] == pytest.approx(plain[2, 2].item())


def test_chain_config_rules():
    fit = {"kind": "fitted", "source": "x"}
    run.check({"model": {"name": "possession_chain", "elo": "history", "params": {"chain": fit}}})
    with pytest.raises(ValueError, match="history"):
        run.check({"model": {"name": "possession_chain", "params": {"chain": fit}}})
    with pytest.raises(ValueError, match="no param"):
        run.check({"model": {"name": "elo_poisson", "params": {"chain": fit}}})
    with pytest.raises(ValueError, match="cannot be fitted"):
        run.check({"model": {"name": "possession_chain", "elo": "history",
                             "params": {"chain": fit, "style": fit}}})
