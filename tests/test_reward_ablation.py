"""Path-order ablation k=2."""
import pytest
from c4_cascade_rl.reward import (
    RewardConfig,
    ablation_reward,
    compute_reward_with_muted_preds,
    path_order_mute_flips,
)
from c4_cascade_rl.schema import parse_trajectory


def _traj():
    return parse_trajectory(
        "HOP=A|r|B|+1\nHOP=B|r|C|-1\nHOP=C|r|D|0\nSTOP|DE=1|DIR=+1"
    )


def test_path_order_mute_k2_keeps_tail():
    traj = _traj()
    muted = traj.muted(2)
    assert [h.src for h in muted.hops] == ["C"]
    assert muted.stop.de == 1


def test_r_abl_relu_with_muted_preds():
    traj = _traj()
    rb = compute_reward_with_muted_preds(
        traj, gold_dir=1, gold_de=1,
        muted_pred_dir=-1, muted_pred_de=0,
        cfg=RewardConfig(alpha=0.5, k_hops=2),
    )
    assert rb.r_task == pytest.approx(1.5)
    assert rb.r_muted == pytest.approx(0.0)
    assert rb.r_abl == pytest.approx(1.5)
    assert path_order_mute_flips(rb.r_task, rb.r_muted)


def test_r_abl_zero_when_no_flip():
    traj = _traj()
    rb = compute_reward_with_muted_preds(
        traj, gold_dir=1, gold_de=1,
        muted_pred_dir=1, muted_pred_de=1,
        cfg=RewardConfig(k_hops=2),
    )
    assert rb.r_abl == 0.0
    assert not path_order_mute_flips(rb.r_task, rb.r_muted)


def test_ablation_reward_relu():
    assert ablation_reward(1.5, 0.5) == 1.0
    assert ablation_reward(0.5, 1.5) == 0.0
