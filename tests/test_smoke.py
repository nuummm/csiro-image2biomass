"""End to end on a synthetic dataset: extract -> CV -> submission -> report (CPU, no torch)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biomass.data import load_train
from biomass.synthetic import make_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str, cwd: Path, env: dict) -> str:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            "--config",
            str(ROOT / "configs/base.yaml"),
            *args,
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("smoke")
    data = make_synthetic_dataset(tmp / "data", n_train=60, n_test=6, seed=0)
    env = {**os.environ, "BIOMASS_DATA_DIR": str(data)}
    common = [
        "--set", "backbone=color_stats",
        "--set", "cv.seeds=[0, 1]",
        "--set", "model.pca_components=8",
    ]  # fmt: skip
    return tmp, data, env, common


def test_long_to_wide_on_synthetic(workspace):
    _, data, _, _ = workspace
    train = load_train(data)
    assert len(train) == 60
    assert train["image_id"].is_unique


def test_three_commands_produce_a_valid_submission(workspace):
    tmp, data, env, common = workspace
    _run("extract_embeddings.py", *common, cwd=tmp, env=env)
    assert (tmp / "emb/color_stats/train.npy").exists()

    out = _run("run_cv.py", *common, "--name", "smoke", cwd=tmp, env=env)
    assert "OOF weighted R2" in out
    summary = json.loads((tmp / "runs/smoke/summary.json").read_text())
    assert np.isfinite(summary["oof_score_mean"])
    assert summary["oof_score_mean"] > 0.0  # colour features carry signal by construction

    _run(
        "make_submission.py",
        *common,
        "--runs",
        "runs/smoke",
        "--out",
        "submission.csv",
        cwd=tmp,
        env=env,
    )
    sub = pd.read_csv(tmp / "submission.csv")
    test = pd.read_csv(data / "test.csv")
    assert sub["sample_id"].tolist() == test["sample_id"].tolist()
    assert (sub["target"] >= 0).all()


def test_run_all_writes_report(workspace):
    tmp, _, env, common = workspace
    _run("extract_embeddings.py", *common, cwd=tmp, env=env)
    _run("run_all.py", *common, "--report-dir", "report", cwd=tmp, env=env)
    text = (tmp / "report/results.md").read_text()
    assert "## Output formulation" in text
    assert "## Ensembling" in text
    assert (tmp / "report/figures/cv_scheme.png").exists()
