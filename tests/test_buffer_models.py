"""Buffer drug-wise split + tiny model forward."""
import numpy as np
import pandas as pd
import torch

from c4_cascade_rl.buffer import drug_wise_split, filter_l1, write_buffer, read_buffer
from c4_cascade_rl.models_hop import build_default_model


def test_drug_wise_91(tmp_path):
    rows = [{"drug": f"D{i//5}", "r_abl": 1.0, "dir_ok": True, "valid": True} for i in range(50)]
    df = pd.DataFrame(rows)
    train, val = drug_wise_split(df, train_frac=0.9, seed=0)
    assert set(train["drug"]).isdisjoint(set(val["drug"]))
    assert len(train) + len(val) == len(df)
    path = tmp_path / "b.parquet"
    write_buffer(df.assign(traj_id=range(len(df)), cell="A", gene="G", text="t", ctx="true",
                           temp=0.2, gold_dir=1, gold_de=1, r_task=1.0, r_muted=0.0,
                           r_total=1.0, hall=False, length=2, n_flips=1), path)
    assert len(read_buffer(path)) == 50


def test_filter_l1():
    df = pd.DataFrame({"dir_ok": [True, True, False], "r_abl": [1.0, 0.0, 2.0]})
    assert len(filter_l1(df)) == 1


def test_model_legal_mask_forward():
    m = build_default_model(n_nodes=20, n_actions=8, d_model=32)
    node_ids = torch.randint(0, 20, (4, 3))
    mask = torch.ones(4, 8)
    mask[:, 5:] = 0
    logits = m.forward_pi(node_ids, mask)
    assert logits.shape == (4, 8)
    # illegal slots heavily negative
    assert (logits[:, 5:] < -1e8).all()
    act, logp = m.sample(node_ids, mask, greedy=True)
    assert act.shape == (4,)
    assert (act < 5).all()
