"""Tiny synthetic DE/DIR CSVs under tmp GeneTak/C32/."""
from pathlib import Path

import pandas as pd

from c4_cascade_rl.splits import (
    OFFICIAL_TRAIN_COUNTS,
    build_official_pert_from_genetak,
    canonicalize_cell,
    count_de_train_pairs,
    discover_genetak_cells,
    gate_a1_counts_from_df,
    genetak_load_diagnostics,
    load_genetak_cell_csvs,
    write_genetak_parquets,
)


def _write_tiny_genetak(root: Path) -> None:
    """DE has extra non-DE rows that DIR does not include (real VCWorld shape)."""
    cell_dir = root / "GeneTak" / "C32"
    cell_dir.mkdir(parents=True)
    # DE: DE-positive + non-DE negatives (Table 5 counts include non-DE)
    de = pd.DataFrame(
        {
            "pert": ["d1", "d1", "d1", "d2", "d2", "d2"],
            "gene": ["g1", "g2", "g3", "g1", "g2", "g3"],
            "label": [1, 0, 0, 1, 0, 0],  # g2/g3 non-DE
            "split": ["train", "train", "train", "test", "test", "test"],
        }
    )
    # DIR: only DE-positive genes (as VCWorld prepare.py)
    di = pd.DataFrame(
        {
            "pert": ["d1", "d2"],
            "gene": ["g1", "g1"],
            "label": [1, 0],  # up/down → +1/-1
            "split": ["train", "test"],
        }
    )
    de.to_csv(cell_dir / "C32_DE.csv", index=False)
    di.to_csv(cell_dir / "C32_DIR.csv", index=False)


def test_canonicalize_aliases():
    assert canonicalize_cell("HepG2_C3A") == "HepG2C3A"
    assert canonicalize_cell("Hs_766T") == "Hs766T"
    assert canonicalize_cell("PANC-1") == "PANC1"
    assert canonicalize_cell("C32") == "C32"
    assert set(OFFICIAL_TRAIN_COUNTS) == {"C32", "HepG2C3A", "HOP62", "Hs766T", "PANC1"}


def test_load_genetak_cell_csvs(tmp_path):
    _write_tiny_genetak(tmp_path)
    assert discover_genetak_cells(tmp_path) == ["C32"]
    df = load_genetak_cell_csvs(tmp_path, "C32")
    assert set(["cell", "drug", "gene", "y_de", "y_dir", "logfc", "fdr", "split"]).issubset(df.columns)
    assert (df["cell"] == "C32").all()
    assert set(df["y_dir"].unique()).issubset({1, -1, 0})

    train = df[df["split"] == "train"]
    # Left merge keeps all DE train rows (incl. non-DE not in DIR)
    assert len(train) == 3
    assert count_de_train_pairs(df) == 3
    assert len(train) == count_de_train_pairs(
        pd.read_csv(tmp_path / "GeneTak" / "C32" / "C32_DE.csv").rename(
            columns={"pert": "drug", "label": "y_de"}
        )
    )

    # DE-positive has DIR label; non-DE extras have y_dir=0
    assert train.loc[train["gene"] == "g1", "y_dir"].iloc[0] == 1
    assert train.loc[train["gene"] == "g2", "y_dir"].iloc[0] == 0
    assert train.loc[train["gene"] == "g3", "y_dir"].iloc[0] == 0
    assert set(train.loc[train["y_de"] == 0, "gene"]) == {"g2", "g3"}

    written = write_genetak_parquets(df, tmp_path / "out")
    assert (tmp_path / "out" / "C32_train.parquet").exists()
    assert (tmp_path / "out" / "C32_test.parquet").exists()
    pert = build_official_pert_from_genetak(df)
    assert "C32" in pert
    assert "d1" in pert["C32"]["train_drugs"]
    counts = gate_a1_counts_from_df(df)
    assert counts["C32"] == 3

    diag = genetak_load_diagnostics(tmp_path, df)
    assert diag["C32"]["n_de_train"] == 3
    assert diag["C32"]["n_dir_train"] == 1
    assert diag["C32"]["n_merged_train"] == 3
