import numpy as np

from rl_cross_context.predictor import DeltaPredictor, build_xy_from_revealed, deltas_from_control


def test_ridge_fit_predict_loss():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 6))
    tgt = src * 0.5 + rng.normal(scale=0.05, size=src.shape)
    src_d = deltas_from_control(np.vstack([np.zeros((1, 6)), src]))
    tgt_d = deltas_from_control(np.vstack([np.zeros((1, 6)), tgt]))
    # use conditions 1..10
    ids = list(range(1, 11))
    X, Y = build_xy_from_revealed(src_d, tgt_d, ids)
    model = DeltaPredictor(alpha=1.0).fit(X, Y)
    pred = model.predict(X)
    assert pred.shape == Y.shape
    loss = model.loss(X, Y)
    assert loss >= 0
