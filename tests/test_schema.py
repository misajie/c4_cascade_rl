import pytest
from c4_cascade_rl.schema import Hop, Stop, parse_line, parse_trajectory


def test_parse_hop_signs():
    h = parse_line("HOP=A|reg|B|+1")
    assert isinstance(h, Hop) and h.sign == 1
    h = parse_line("HOP=A|reg|B|-1")
    assert h.sign == -1
    h = parse_line("HOP=A|reg|B|0")
    assert h.sign == 0


def test_parse_stop():
    s = parse_line("STOP|DE=1|DIR=+1")
    assert isinstance(s, Stop) and s.de == 1 and s.direction == 1
    s = parse_line("STOP|DE=0|DIR=-1")
    assert s.de == 0 and s.direction == -1


def test_invalid_drops_whole_trajectory():
    text = "HOP=A|r|B|+1\nHOP=BAD\nSTOP|DE=1|DIR=+1"
    traj = parse_trajectory(text)
    assert traj.valid is False
    assert traj.hops == []
    assert traj.drop_reason and "invalid_line" in traj.drop_reason


def test_missing_stop_invalid():
    traj = parse_trajectory("HOP=A|r|B|+1")
    assert not traj.valid
    assert traj.drop_reason == "missing_stop"


def test_valid_trajectory_and_mute():
    text = "\n".join([
        "HOP=A|r|B|+1",
        "HOP=B|r|C|-1",
        "HOP=C|r|D|0",
        "STOP|DE=1|DIR=+1",
    ])
    traj = parse_trajectory(text)
    assert traj.valid and traj.length == 3
    muted = traj.muted(2)
    assert muted.length == 1
    assert muted.hops[0].src == "C"
    assert muted.stop.direction == 1


def test_roundtrip_lines():
    traj = parse_trajectory("HOP=X|y|Z|+1\nSTOP|DE=0|DIR=-1")
    assert parse_trajectory(traj.to_text()).valid
