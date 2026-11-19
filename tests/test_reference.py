"""The WC2026 reference files obey the competition's own rules."""

import itertools
from collections import Counter

import pytest

from wcsimfw.reference import load

FMT, TEAMS, RES, ELO = (load(n) for n in ("format", "teams", "results", "elo"))
TEAM = TEAMS["teams"]
MATCHES = RES["matches"]
GROUP = [m for m in MATCHES if m["stage"] == "group"]
KO = {m["match"]: m for m in MATCHES if m["stage"] != "group"}
WINNERS = "ABDEGIKL"  # the eight group winners that can meet a third (art. 12.6)


def winner(m):
    for s in (m["goals_90"], m["goals_120"], m["penalties"]):
        if s and s[0] != s[1]:
            return (m["home"], m["away"]) if s[0] > s[1] else (m["away"], m["home"])
    raise AssertionError(m)


def test_twelve_groups_of_four_with_hosts_apart():
    assert len(TEAM) == 48
    assert Counter(t["group"] for t in TEAM.values()) == {g: 4 for g in FMT["groups"]}
    assert sorted(TEAM[h]["group"] for h in TEAMS["hosts"]) == ["A", "B", "D"]


def test_group_stage_is_a_round_robin_over_three_matchdays():
    assert len(GROUP) == 72
    pairs = Counter(frozenset((m["home"], m["away"])) for m in GROUP)
    assert set(pairs.values()) == {1}
    for g in FMT["groups"]:
        ts = [t for t, v in TEAM.items() if v["group"] == g]
        assert {frozenset(p) for p in itertools.combinations(ts, 2)} <= set(pairs)
    for md in (1, 2, 3):
        played = Counter(t for m in GROUP if m["matchday"] == md for t in (m["home"], m["away"]))
        assert set(played) == set(TEAM) and set(played.values()) == {1}


def test_group_positions_and_best_thirds_give_the_round_of_32():
    pos = Counter((t["group"], t["group_position"]) for t in TEAM.values())
    assert set(pos.values()) == {1} and len(pos) == 48
    best = {t for t, v in TEAM.items() if v["third_rank"] and v["third_rank"] <= 8}
    assert len(best) == 8
    through = {t for t, v in TEAM.items() if v["group_position"] <= 2} | best
    r32 = {t for n in range(73, 89) for t in (KO[n]["home"], KO[n]["away"])}
    assert through == r32


def test_knockout_tree_uses_every_slot_once():
    ko = FMT["knockout"]
    assert sorted(map(int, ko)) == list(range(73, 105))
    codes = Counter(c for n in ko for c in ko[n])
    assert set(codes.values()) == {1}
    assert {c for c in codes if c[0] in "12"} == {f"{p}{g}" for p in "12" for g in FMT["groups"]}
    assert {c for c in codes if c[0] == "W"} == {f"W{n}" for n in range(73, 103)}
    assert {c for c in codes if c[0] == "L"} == {"L101", "L102"}
    thirds = {ko[n][0][1]: ko[n][1][1:] for n in ko if ko[n][1][0] == "3"}
    assert sorted(thirds) == list(WINNERS)


def test_annex_c_covers_every_set_of_eight_thirds():
    ko = FMT["knockout"]
    pool = {ko[n][0][1]: set(ko[n][1][1:]) for n in ko if ko[n][1][0] == "3"}
    annex = FMT["annex_c"]
    assert set(annex) == {"".join(c) for c in itertools.combinations(FMT["groups"], 8)}
    for key, row in annex.items():
        assert sorted(row) == list(WINNERS)
        assert sorted(row.values()) == list(key)  # each qualified third placed once
        for w, g in row.items():
            assert g != w and g in pool[w]  # never the winner of its own group


def test_played_knockout_follows_the_bracket_and_annex_c():
    ko = FMT["knockout"]
    best = sorted(v["group"] for v in TEAM.values() if v["third_rank"] and v["third_rank"] <= 8)
    row = FMT["annex_c"]["".join(best)]
    slot = {(v["group_position"], v["group"]): t for t, v in TEAM.items()}
    won = {}
    for n in range(73, 105):
        need = set()
        for c in ko[str(n)]:
            if c[0] in "12":
                need.add(slot[(int(c[0]), c[1])])
            elif c[0] == "3":
                winner_group = ko[str(n)][0][1]
                need.add(slot[(3, row[winner_group])])
            else:
                need.add(won[int(c[1:])][0 if c[0] == "W" else 1])
        assert {KO[n]["home"], KO[n]["away"]} == need, n
        won[n] = winner(KO[n])


@pytest.mark.parametrize("m", [m for m in MATCHES if m["stage"] != "group"],
                         ids=lambda m: str(m["match"]))
def test_extra_time_and_penalties_only_when_level(m):
    level_90 = m["goals_90"][0] == m["goals_90"][1]
    assert (m["goals_120"] is not None) == level_90
    if m["penalties"]:
        assert m["goals_120"][0] == m["goals_120"][1]
    assert all(m["goals_120"] is None and m["penalties"] is None for m in GROUP)


def test_dates_are_before_kickoff():
    kickoff = min(m["date"] for m in MATCHES)
    assert kickoff == "2026-06-11"
    assert TEAMS["fifa_ranking"]["draw"] < "2025-12-05" <= TEAMS["fifa_ranking"]["tiebreak"]
    assert TEAMS["fifa_ranking"]["tiebreak"] <= kickoff
    assert ELO["date"] < kickoff and set(ELO["ratings"]) == set(TEAM)
    for release in TEAMS["fifa_ranking"]["releases"].values():
        assert set(release) == set(TEAM)
