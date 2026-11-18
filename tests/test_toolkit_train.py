"""End-to-end checks of `multitask train` / `multitask report`: every method,
frozen stage 2, single-task and every ablation, each trained for one epoch on
a small synthetic dataset (with and without a merged output).

Downloads google/bert_uncased_L-2_H-128_A-2 and the distilbert-base-uncased
tokenizer, so it needs network access the first time. About 5 minutes on a CPU.

    python -m pytest tests/test_toolkit_train.py -o addopts=
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pytest

from multitask.cli import main

WORDS = ["apple", "river", "stone", "cloud", "music", "paper", "glass", "tiger"]
FAST = ["--max-length", "16", "--batch-size", "16", "--epochs", "1", "--patience", "1",
        "--verbose", "0"]


def make_split(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        words = list(rng.choice(WORDS, size=5))
        src = "A" if i % 2 == 0 else "B"
        rows.append({
            "text": " ".join(words) + f" item {i}",
            "dataset": src,
            "t1": float("apple" in words),
            "t2": float("river" in words),
            "t3": float("stone" in words) if src == "A" else -1,
            "p1": float("cloud" in words) if src == "B" else np.nan,
            "p2": float("music" in words) if src == "B" else np.nan,
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("toy")
    for split, n, seed in (("train", 96, 0), ("val", 48, 1), ("test", 48, 2)):
        make_split(n, seed).to_csv(d / f"toy_{split}.csv", index=False)
    return str(d)


def run(*argv):
    main(list(argv))


def results(out):
    paths = sorted(glob.glob(os.path.join(out, "*", "result.json")))
    return {os.path.basename(os.path.dirname(p)): json.load(open(p)) for p in paths}


def scored_everywhere(r):
    for split in ("val", "test"):
        for o in r["scored_outputs"]:
            assert "f1" in r[split][o] and "auc" in r[split][o], (split, o)


@pytest.mark.parametrize("method", ["plain", "famo", "dbmtl", "famo_mlm", "famo_consf1_m5"])
def test_methods_without_pool(data, tmp_path, method):
    out = str(tmp_path / "runs")
    run("train", "--data", data, "--method", method, "--out", out, *FAST)
    (name, r), = results(out).items()
    assert name == f"{method}_s42"
    assert r["outputs"] == ["t1", "t2", "t3", "p1", "p2"]
    scored_everywhere(r)
    assert os.path.exists(os.path.join(out, name, "dataset.yaml"))
    assert os.path.exists(os.path.join(out, name, "predictions.npz"))


def test_pool_famo_consistency_stage2_and_report(data, tmp_path, capsys):
    out = str(tmp_path / "runs")
    run("train", "--data", data, "--pool", "any_p=p1,p2", "--method", "famo_consf1_m5",
        "--stage2", "--set", "stage2_epochs=1", "--out", out, *FAST)
    (name, r), = results(out).items()
    assert name == "famo_consf1_m5_stage2_s42"
    assert r["pool"]["tasks"] == ["p1", "p2"] and "any_p" in r["scored_outputs"]
    scored_everywhere(r)
    s2 = r["stage2"]
    assert s2["frozen_unchanged"] and set(s2["runs"]) == {"1e-4", "3e-5"}
    capsys.readouterr()
    run("report", out, "--csv", str(tmp_path / "scores.csv"))
    text = capsys.readouterr().out
    assert "famo_consf1_m5_stage2" in text and "S2 test F1" in text
    assert len(pd.read_csv(tmp_path / "scores.csv")) == len(r["scored_outputs"])


def test_single_task(data, tmp_path, capsys):
    out = str(tmp_path / "runs")
    run("train", "--data", data, "--method", "single_task", "--tasks", "t1,t2", "--out", out,
        *FAST)
    rs = results(out)
    assert sorted(rs) == ["single_task_t1_s42", "single_task_t2_s42"]
    assert all(len(r["scored_outputs"]) == 1 for r in rs.values())
    capsys.readouterr()
    run("report", out, "--brief")
    assert "single_task" in capsys.readouterr().out


def test_hide_each_label_with_probes(data, tmp_path):
    out = str(tmp_path / "runs")
    run("train", "--data", data, "--tasks", "t1,t2,t3", "--method", "plain",
        "--hide-label", "all", "--out", out, *FAST)
    rs = results(out)
    assert sorted(rs) == ["plain_full_s42", "plain_hide-t1_s42", "plain_hide-t2_s42",
                          "plain_hide-t3_s42"]
    assert set(rs["plain_full_s42"]["probes"]) == {"t1", "t2", "t3"}
    hid = rs["plain_hide-t2_s42"]
    assert hid["hidden_output"] == "t2" and "t2" not in hid["scored_outputs"]
    assert set(hid["probes"]) == {"t2"} and "test" in hid["probes"]["t2"]["shared"]
    from multitask.train.report import hide_label_table, load_results
    table = hide_label_table(load_results(out))
    assert table.count("plain") == 3 and "Hidden labels" in table


def test_hide_share_scores_hidden_cells(data, tmp_path):
    out = str(tmp_path / "runs")
    run("train", "--data", data, "--method", "famo", "--hide-share", "0.5", "--out", out, *FAST)
    (name, r), = results(out).items()
    assert name == "famo_hide50pct_s42"
    cells = r["hidden_cells"]["scores"]
    labelled = {"t1": 96, "t2": 96, "t3": 48, "p1": 48, "p2": 48}
    assert {t: v["n"] for t, v in cells.items()} == {t: n // 2 for t, n in labelled.items()}


@pytest.mark.parametrize("extra,name", [
    (["--label-budget", "t1=0.25"], "plain_budget-t1-25pct_s42"),
    (["--hide-source", "A:t1,t2"], "plain_nosrc-A_s42"),
    (["--keep-sources", "B", "--tasks", "t1,t2,p1"], "plain_only-B_s42"),
    (["--unweighted-loss"], "plain_unweighted_s42"),
    (["--no-grouping"], "plain_nogroups_s42"),
    (["--arch", "32/16/8"], "plain_arch-32-16-8_s42"),
    (["--probes", "all", "--probe-untrained"], "plain_untrained-probe_s42"),
])
def test_ablations(data, tmp_path, extra, name):
    out = str(tmp_path / "runs")
    run("train", "--data", data, "--method", "plain", "--out", out, *extra, *FAST)
    rs = results(out)
    assert list(rs) == [name]
    r = rs[name]
    if "--probe-untrained" in extra:
        assert r["untrained"] and set(r["probes"]) == set(r["outputs"])
    else:
        scored_everywhere(r)
    if "--unweighted-loss" in extra:
        assert all(w == {"0": 1.0, "1": 1.0} for w in r["class_weights"].values())
    if "--hide-source" in extra:
        prepared = glob.glob(os.path.join(out, ".prepared", "*", "test.csv"))[0]
        df = pd.read_csv(prepared)
        assert df.loc[df["dataset"] == "A", ["t1", "t2"]].isna().all().all()


def test_dry_run_and_resume(data, tmp_path, capsys):
    out = str(tmp_path / "runs")
    run("train", "--data", data, "--method", "plain", "--seeds", "42,1", "--out", out,
        "--dry-run", *FAST)
    assert results(out) == {}
    assert "plain_s1: to run" in capsys.readouterr().out
    run("train", "--data", data, "--method", "plain", "--out", out, *FAST)
    capsys.readouterr()
    run("train", "--data", data, "--method", "plain", "--seeds", "42,1", "--out", out,
        "--dry-run", *FAST)
    text = capsys.readouterr().out
    assert "plain_s42: done, skipped" in text and "plain_s1: to run" in text
