"""Complete runs on synthetic images (tests/conftest.py): every run writes one CSV with one row per image and a
final 'mean' row; the experiment script writes the per-seed CSVs; the sensitivity reproduces the main run."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from spmvfcm import runner as R

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiments as N  # noqa: E402

PARAMS = N.CFG["params"]


def only_csv(folder):
    files = sorted(Path(folder).rglob("*.csv"))
    assert len(files) == 1, files
    return files[0]


@pytest.mark.parametrize("method", ["SPMVFCM", "SPMVKFCM", "SPWPMVFCM", "SPWPMVKFCM", "kmeans", "fcm"])
def test_dermatology_run(method, isic, tmp_path):
    info = R.run_derma(method, isic["list"], isic["images"], isic["masks"], tmp_path, seed=41, params=PARAMS.get(method))
    df = pd.read_csv(only_csv(tmp_path / "derma" / method), index_col=0)
    assert len(df) == 4 and df["id"].iloc[-1] == "mean" and info["seed"] == 41


@pytest.mark.parametrize("method", ["SPMVFCM", "SPMVKFCM", "SPWPMVFCM", "SPWPMVKFCM", "kmeans", "fcm"])
def test_xray_run(method, rsua, tmp_path):
    R.run_xray(method, "noncovid", rsua, tmp_path, seed=43, params=PARAMS.get(method))
    assert len(pd.read_csv(only_csv(tmp_path / "noncovid" / method), index_col=0)) == 33


def test_example_masks_are_saved(isic, tmp_path):
    ids = isic["list"].read_text().split()
    info = R.run_derma("SPMVFCM", isic["list"], isic["images"], isic["masks"], tmp_path / "run", seed=42,
                       params=PARAMS["SPMVFCM"], record_masks=(ids[:2], tmp_path / "masks"))
    assert info["masks_saved"] == ids[:2]
    assert sorted(p.name for p in (tmp_path / "masks" / "SPMVFCM").iterdir()) == \
        sorted([f"{i}_{s}.png" for i in ids[:2] for s in ("segmented", "inverted")])


def test_sensitivity_at_the_main_parameters_reproduces_the_main_run(isic, tmp_path):
    common = (isic["list"], isic["images"], isic["masks"])
    R.run_derma("SPMVFCM", *common, tmp_path / "main", seed=42, params=PARAMS["SPMVFCM"])
    main = pd.read_csv(only_csv(tmp_path / "main" / "derma" / "SPMVFCM"), index_col=0)
    R.run_sensitivity("SPMVFCM_lambda", *common, tmp_path, seed=42, grid=[PARAMS["SPMVFCM"]["lambda"]], target_name="sens")
    sens = pd.read_csv(tmp_path / "sens" / "MVJFCM" / "Lamda" / "FINAL_RESULT_no_kernal_MVJFC_parameter_100000000.0.csv",
                       index_col=0)
    pd.testing.assert_frame_equal(main, sens)


@pytest.mark.parametrize("method", ["SPWPMVFCM", "kmeans"])
def test_experiment_job_writes_the_per_seed_csv(method, rsua, tmp_path, monkeypatch):
    monkeypatch.setattr(N, "OUT", tmp_path)
    monkeypatch.setattr(N, "RSUA", rsua)
    N.run_job({"kind": "main", "seed": 41, "dataset": "noncovid", "method": method})
    df = pd.read_csv(N.final_csv(41, "noncovid", method), index_col=0)
    assert len(df) == 33 and df["id"].iloc[0] == "CXR_Images_208" and df["id"].iloc[-1] == "mean"
    info = json.loads((tmp_path / "seed_41" / "notebook_output" / "noncovid" / method / "run_info.json").read_text())
    assert info["seed"] == 41 and info["params"] == PARAMS.get(method)


def test_one_step_rejects_unknown_dataset_or_method():
    for args in (["one", "derma"], ["one", "skin", "SPMVFCM"], ["one", "derma", "SPMVXYZ"]):
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_experiments.py"), *args],
                             capture_output=True, text=True, timeout=120)
        assert out.returncode == 2 and "one: DATASET METHOD" in out.stderr


@pytest.mark.slow
def test_xray_deep_learning_run(rsua, tmp_path):
    R.run_xray("dec", "noncovid", rsua, tmp_path, seed=41)
    assert len(pd.read_csv(only_csv(tmp_path / "noncovid" / "dec"), index_col=0)) == 33
