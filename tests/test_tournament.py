import pytest
import torch

from wcsimfw import match, tournament
from wcsimfw.reference import load

T = tournament.Tournament.load()
TEAMS = load("teams")["teams"]
ELO = torch.tensor([load("elo")["ratings"][t] for t in T.teams], dtype=torch.float64)


def real_goals():
    held, _ = T.held("md3")
    return torch.tensor([held[i] for i in range(len(T.fixtures))])[None]


def test_load():
    assert len(T.teams) == 48 and T.fixtures.shape == (72, 2)
    assert (T.fixtures // 4)[:, 0].eq((T.fixtures // 4)[:, 1]).all()  # matches inside groups
    assert (T.annex >= 0).sum() == 495 * 8


def test_ranking_of_played_groups():
    pos, mask = tournament.rank_groups(T, real_goals())
    for g, row in enumerate(pos[0].tolist()):
        assert [TEAMS[T.teams[i]]["group_position"] for i in row] == [1, 2, 3, 4], T.groups[g]
    qualified = {TEAMS[t]["group"] for t in TEAMS if (TEAMS[t]["third_rank"] or 99) <= 8}
    assert {T.groups[g] for g in range(12) if mask[0] >> g & 1} == qualified


def test_head_to_head_before_goal_difference():
    # Group A: team 0 beats 1 but has the worse goal difference; both end on 6 points.
    score = {(0, 1): (1, 0), (1, 2): (5, 0), (1, 3): (5, 0), (0, 3): (1, 0), (0, 2): (0, 1),
             (2, 3): (0, 0)}
    goals = torch.zeros(1, 72, 2, dtype=torch.long)
    for k, (h, a) in enumerate(T.fixtures.tolist()):
        if (h, a) in score:
            goals[0, k] = torch.tensor(score[h, a])
        elif (a, h) in score:
            goals[0, k] = torch.tensor(score[a, h][::-1])
    pos, _ = tournament.rank_groups(T, goals)
    assert pos[0, 0].tolist() == [0, 1, 2, 3]


def test_line_up_of_played_tournament():
    lineup = tournament.line_up(T, *tournament.rank_groups(T, real_goals()))[0].tolist()
    played = {m["match"]: {m["home"], m["away"]} for m in T.results if m["stage"] == "r32"}
    for k, m in enumerate(range(73, 89)):
        assert {T.teams[lineup[2 * k]], T.teams[lineup[2 * k + 1]]} == played[m]


def test_everything_held_gives_the_real_champion():
    full, adv = match.elo_pairs(ELO)
    out = tournament.simulate(T, full, adv, n=50, after="final")
    champ = out["exact"][:, 5].argmax().item()
    assert T.teams[champ] == "Spain" and out["mc"][champ, 5] == out["exact"][champ, 5] == 1


@pytest.mark.parametrize("after", [None, "md2", "r16"])
def test_monte_carlo_agrees_with_exact(after):
    full, adv = match.elo_pairs(ELO)
    out = tournament.simulate(T, full, adv, n=4000, seed=1, after=after, chunk=1500)
    for k in ("mc", "exact"):
        assert torch.allclose(out[k].sum(0), torch.tensor([32, 16, 8, 4, 2, 1.0],
                                                          dtype=torch.float64))
    e = out["exact"]
    z = (out["mc"] - e).abs() / (e * (1 - e) / 4000).sqrt().clamp_min(1e-12)
    assert z.max() < 4.5, z.max()
    assert out["exact_se"][:, 1:].sum() < out["mc_se"][:, 1:].sum()  # recursion: less noise


def test_hosts_play_at_home():
    for k, (h, a) in enumerate(T.fixtures.tolist()):
        for team in (T.teams[h], T.teams[a]):
            if team in T.hosts:  # every group match of a host nation is in its own country
                assert T.fixture_place[k] == 1 + T.hosts.index(team)
    assert T.place[79] == 1 + T.hosts.index("Mexico") and T.place[104] == 1 + T.hosts.index("USA")
    zero, _ = match.elo_pairs(ELO, host=T.host_bonus(0.0)), None
    plain = tournament.simulate(T, *match.elo_pairs(ELO), n=3000)["exact"]
    assert torch.allclose(tournament.simulate(T, *zero, n=3000)["exact"], plain)
    home = tournament.simulate(T, *match.elo_pairs(ELO, host=T.host_bonus(0.25)), n=3000)["exact"]
    for h in T.hosts:
        i = T.teams.index(h)
        assert home[i, 0] > plain[i, 0] and home[i, 1] > plain[i, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_runs():
    full, adv = match.elo_pairs(ELO.cuda())
    out = tournament.simulate(T, full, adv, n=4000)
    assert torch.allclose(out["exact"].sum(0).cpu(),
                          torch.tensor([32, 16, 8, 4, 2, 1.0], dtype=torch.float64))
