from rl_cross_context.eval_curves import (
    risk_at_budgets,
    cumulative_reward_curve,
    area_under_budget_curve,
    summarize_method_curves,
    paired_compare,
)


def test_risk_budget_helpers():
    losses = [1.0, 0.8, 0.5, 0.4]
    rb = risk_at_budgets(losses, [1, 2, 4, 10])
    assert rb[1] == 1.0
    assert rb[2] == 0.8
    assert rb[4] == 0.4
    assert rb[10] == 0.4
    cr = cumulative_reward_curve([0.1, 0.2, -0.05])
    assert abs(cr[-1] - 0.25) < 1e-9
    aubc = area_under_budget_curve([1, 2, 4], [1.0, 0.8, 0.4])
    assert isinstance(aubc, float)


def test_summarize():
    hist = {
        "random": [{"loss": 1.0, "reward": 0.0}, {"loss": 0.5, "reward": 0.5}],
        "open_loop": [{"loss": 0.9, "reward": 0.0}, {"loss": 0.6, "reward": 0.3}],
    }
    s = summarize_method_curves(hist, [1, 2])
    ranked = paired_compare(s)
    assert len(ranked) == 2
