"""Checks for the toolkit's data side: detection, the dataset spec, the
standard files (materialize), ablation bookkeeping, cleaning and run
planning. No model is built, so these run in seconds.

    python -m pytest tests/test_toolkit_dataset.py -o addopts=
"""
import os

import numpy as np
import pandas as pd
import pytest

from multitask.train.data import random_hidden_cells
from multitask.train.dataset import (
    TaskSpec, column_labels, detect, find_split_files, join_specs, load_spec, materialize,
    parse_cut, save_spec, split_prefix,
)
from multitask.train.methods import (
    PAPER_ARCHITECTURES, RunSettings, architecture, parse_set, run_name, settings_for,
)
from multitask.train.runner import check_settings, parse_hide_source, plan_runs
from multitask.prepare import clean_text, prepare

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    src = np.where(np.arange(n) % 2 == 0, "A", "B")
    yes = rng.integers(0, 2, n)
    coded = np.where(src == "A", rng.integers(0, 2, n), -1)      # -1 = no label on B rows
    word = np.where(rng.integers(0, 2, n) == 1, "Yes", "no")
    score = rng.integers(0, 5, n)
    return pd.DataFrame({
        "id": np.arange(n) + 1000,
        "comment": [f"this is sentence number {i} with some words in it" for i in range(n)],
        "dataset": src,
        "toxic": yes.astype(float),
        "coded": coded,
        "answer": word,
        "rating": score,
        "constant": 1,
        "maybe": np.where(src == "B", rng.integers(0, 2, n).astype(float), np.nan),
    })


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "mydata"
    d.mkdir()
    for split, n, seed in (("train", 80, 0), ("dev", 40, 1), ("test", 40, 2)):
        make_frame(n, seed).to_csv(d / f"mine_{split}.csv", index=False)
    return str(d)


# -- detection ---------------------------------------------------------------

def test_find_split_files_and_prefix(folder, tmp_path):
    assert find_split_files(folder) == {"train": "mine_train.csv", "val": "mine_dev.csv",
                                        "test": "mine_test.csv"}
    assert split_prefix(os.path.join(folder, "mine_train.csv")) == (folder, "mine_")
    make_frame(10, 3).to_csv(os.path.join(folder, "other_train.csv"), index=False)
    with pytest.raises(ValueError, match="more than one file"):
        find_split_files(folder)
    assert find_split_files(folder, "mine_")["train"] == "mine_train.csv"
    with pytest.raises(ValueError, match="no train / val / test file"):
        find_split_files(str(tmp_path))


def test_detect_roles(folder):
    spec, report = detect(folder)
    roles = {r.column: r.role for r in report}
    assert roles["comment"] == "text" and roles["dataset"] == "source"
    assert roles["toxic"] == roles["coded"] == roles["answer"] == roles["maybe"] == "task"
    for c in ("id", "rating", "constant"):
        assert roles[c] == "not used"
    assert spec.task("coded").missing == [-1.0]
    assert spec.task("answer").binarize == "==yes"
    # groups follow the source that labels most of each task's rows
    assert spec.task("coded").group == "A" and spec.task("maybe").group == "B"
    single, _ = detect(folder, group_by="single")
    assert {t.group for t in single.tasks} == {"all"}


def test_detect_binarize_and_selection(folder):
    spec, report = detect(folder, binarize=["rating>=3"])
    assert spec.task("rating").binarize == ">=3"
    with pytest.raises(ValueError, match="matched no column"):
        detect(folder, binarize=["nothing>1"])
    sub = spec.select(["toxic", "rat*"])
    assert sub.task_names() == ["toxic", "rating"]
    assert spec.select(ignore=["group:B"]).task_names() == [
        t for t in spec.task_names() if spec.task(t).group != "B"]
    with pytest.raises(ValueError, match="matches no task"):
        spec.select(["missing_task"])


def test_pool_and_outputs(folder):
    spec, _ = detect(folder, group_by="single")
    pooled = spec.with_pool("any", ["toxic", "answer"])
    assert pooled.scored_outputs()[0] == "any"
    assert "toxic" not in pooled.scored_outputs()
    assert pooled.output_tasks("any") == ["toxic", "answer"]
    with pytest.raises(ValueError, match="only one merged output"):
        pooled.with_pool("again", ["coded", "maybe"])


def test_column_labels_rules():
    s = pd.Series([0.0, 1.0, -1.0, np.nan, 3.0])
    out = column_labels(s, TaskSpec("x", "x", missing=[-1.0]), [])
    assert np.isnan(out[2]) and np.isnan(out[3]) and out[1] == 1
    cut = column_labels(pd.Series([0, 2, 3, np.nan]), TaskSpec("x", "x", binarize=">2"), [])
    assert list(cut[:3]) == [0, 0, 1] and np.isnan(cut[3])
    words = column_labels(pd.Series(["Yes", "no", "NA", " yes "]),
                          TaskSpec("x", "x", binarize="==yes"), ["NA"])
    assert list(words[[0, 1, 3]]) == [1, 0, 1] and np.isnan(words[2])
    assert parse_cut(">=1.5") == (">=", 1.5) and parse_cut("==Toxic") == ("==", "toxic")
    with pytest.raises(ValueError):
        parse_cut("1")


def test_spec_yaml_roundtrip(folder, tmp_path):
    spec, _ = detect(folder, binarize=["rating>=3"])
    path = str(tmp_path / "spec" / "dataset.yaml")
    save_spec(spec, path, header="test")
    again = load_spec(path)
    assert again.task_names() == spec.task_names()
    assert again.task("rating").binarize == ">=3"
    assert os.path.abspath(again.data_dir) == os.path.abspath(folder)


def test_bundled_specs_load():
    for name in ("iac", "hatespeech"):
        spec = load_spec(os.path.join(REPO, "datasets", name, "dataset.yaml"))
        assert len(spec.scored_outputs()) == 10
    iac = load_spec(os.path.join(REPO, "datasets", "iac", "dataset.yaml"))
    assert iac.pool.name == "propaganda_max_pool" and len(iac.pool.tasks) == 18
    assert [g for g, _ in iac.groups()] == ["propaganda_binary", "wa_bi", "discourse"]
    skip = iac.skip_map()
    assert set(skip) == set(iac.expand(["group:discourse"])) and skip["nasty_nice_bi"] == iac.pool.tasks


# -- materialize -------------------------------------------------------------

def test_materialize_values_and_cache(folder, tmp_path):
    spec, _ = detect(folder)
    out = materialize([spec], str(tmp_path / "cache"), verbose=False)
    train = pd.read_csv(out.path("train"))
    raw = pd.read_csv(os.path.join(folder, "mine_train.csv"))
    assert list(train.columns[:2]) == ["text", "dataset"]
    assert (train["text"] == raw["comment"]).all()
    assert train["coded"].isna().sum() == (raw["coded"] == -1).sum()
    assert (train["answer"].dropna() == (raw["answer"].str.lower() == "yes").astype(float)).all()
    mtime = os.path.getmtime(out.path("train"))
    again = materialize([spec], str(tmp_path / "cache"), verbose=False)
    assert again.data_dir == out.data_dir and os.path.getmtime(out.path("train")) == mtime


def test_materialize_hide_and_keep_sources(folder, tmp_path):
    spec, _ = detect(folder)
    hide = parse_hide_source(spec, "A:toxic,coded")
    assert hide == ("A", ["toxic", "coded"])
    out = materialize([spec], str(tmp_path / "c"), hide_source=hide, verbose=False)
    for split in ("train", "val", "test"):
        df = pd.read_csv(out.path(split))
        assert df.loc[df["dataset"] == "A", ["toxic", "coded"]].isna().all().all()
        assert df.loc[df["dataset"] == "B", "toxic"].notna().all()
    only_b = materialize([spec], str(tmp_path / "c"), keep_sources=["B"], verbose=False)
    assert set(pd.read_csv(only_b.path("train"))["dataset"]) == {"B"}


def test_joint_datasets(folder, tmp_path):
    a, _ = detect(folder, group_by="single")
    b = a.select(["toxic"])
    b.name, b.tasks[0].name = "second", "toxic2"
    joined = join_specs([a, b])
    assert "toxic2" in joined.task_names()
    assert joined.task("toxic2").group == "second_all"  # same group name -> kept apart
    out = materialize([a, b], str(tmp_path / "j"), verbose=False)
    df = pd.read_csv(out.path("train"))
    assert len(df) == 160 and df["toxic2"].notna().sum() == 80 and df["coded"].iloc[80:].isna().all()
    with pytest.raises(ValueError, match="share task names"):
        join_specs([a, a])


# -- ablation bookkeeping ----------------------------------------------------

def test_random_hidden_cells_matches_hate_speech_rule():
    labels = {"a": np.zeros(100, np.float32), "b": np.r_[np.zeros(50), -np.ones(50)].astype(np.float32)}
    cells = random_hidden_cells(labels, ["a", "b"], 0.3, seed=7)
    assert cells["a"].size == 30 and cells["b"].size == 15 and cells["b"].max() < 50
    # hatespeech/run_hatespeech.py hidden_cells: rng.choice(n_rows, k) per item, in order
    rng = np.random.default_rng(7)
    assert np.array_equal(np.sort(rng.choice(100, size=30, replace=False)), cells["a"])
    assert np.array_equal(random_hidden_cells(labels, ["a", "b"], 0.3, seed=7)["a"], cells["a"])


def test_settings_and_names():
    s = settings_for("famo_consf1_m5")
    assert (s.loss_balancer, s.mlm_lambda, s.mlm_mask_prob, s.consistency_weight,
            s.consistency_keep) == ("famo", 0.0, 0.05, 1.0, 0.1)
    assert (s.learning_rate, s.patience, s.arch, s.dropout) == (1e-4, 3, "wide", 0.0)
    assert run_name(s) == "famo_consf1_m5_s42"
    assert run_name(settings_for("plain", hide_label="x", stage2=True, seed=1)) == \
        "plain_hide-x_stage2_s1"
    assert parse_set(["consistency_keep=0.2", "stage2=true", "mlm_lambda=none"]) == \
        {"consistency_keep": 0.2, "stage2": True, "mlm_lambda": None}
    with pytest.raises(ValueError, match="unknown setting"):
        parse_set(["nope=1"])
    assert architecture("wide") == PAPER_ARCHITECTURES["wide"]
    assert architecture("256,128/128,64/64,32") == PAPER_ARCHITECTURES["wide"]
    with pytest.raises(ValueError):
        architecture("big")


def test_check_settings_errors(folder):
    spec, _ = detect(folder)
    check_settings(spec, RunSettings())
    bad = [
        dict(hide_label="nope"), dict(hide_share=0.5, hide_label="toxic"),
        dict(probes="hidden"), dict(label_budget={"toxic": 1.5}), dict(hide_share=1.0),
        dict(method="plain", consistency_weight=1.0), dict(probe_untrained=True),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            check_settings(spec, RunSettings(**{"method": "plain", **kw}))
    with pytest.raises(ValueError, match="at least two outputs"):
        check_settings(spec.select(["toxic"]), RunSettings())
    check_settings(spec.select(["toxic"]), settings_for("single_task"))


def test_plan_runs(folder):
    spec, _ = detect(folder)
    n = len(spec.scored_outputs())
    single = plan_runs(spec, settings_for("single_task"), [42, 1])
    assert len(single) == 2 * n and all(len(r[1].scored_outputs()) == 1 for r in single)
    hide = plan_runs(spec, settings_for("famo_consf1_m5"), [42], hide_each=True)
    assert hide[0][0] == "famo_consf1_m5_full_s42" and hide[0][2].probes == "all"
    assert [r[2].hide_label for r in hide[1:]] == spec.scored_outputs()
    assert all(r[2].probes == "hidden" for r in hide[1:])


# -- cleaning ----------------------------------------------------------------

def test_clean_text_cmp_style():
    steps = ("ascii", "urls", "handles", "punct")
    assert clean_text("@user Don’t   go! https://x.co/a", steps) == "Dont go"
    assert clean_text("you f@ggot and d*ck", steps) == "you fggot and dck"
    assert clean_text("café — naïve 😀", steps) == "cafe naive"
    assert clean_text("see URL here", steps, remove_tokens=["URL"]) == "see here"
    assert clean_text("  Keep  Case ") == "Keep Case"


def test_prepare_split_and_dedupe(tmp_path):
    n = 300
    df = pd.DataFrame({"text": [f"text number {i % 280} here" for i in range(n)],
                       "label": [i % 3 == 0 for i in range(n)], "votes": np.arange(n)})
    raw = tmp_path / "raw.csv"
    df.to_csv(raw, index=False)
    report = prepare(str(raw), str(tmp_path / "out"), name="d", stratify=["label"],
                     dedupe_keep_max="votes")
    parts = {s: pd.read_csv(tmp_path / "out" / f"d_{s}.csv") for s in ("train", "val", "test")}
    assert sum(len(p) for p in parts.values()) == 280
    texts = pd.concat([p["text"] for p in parts.values()])
    assert texts.is_unique
    # the kept copy of a duplicate is the one with the most votes
    assert pd.concat(parts.values()).set_index("text").loc["text number 0 here", "votes"] == 280
    assert abs(len(parts["train"]) - 224) <= 2
    assert "What `multitask train` will detect" in report
