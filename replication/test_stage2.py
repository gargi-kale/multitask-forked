"""Checks for the frozen stage 2 (Stage2, run_stage2, table3_scores) in
test_no_prop_overlap.py.

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_stage2.py

Downloads google/bert_uncased_L-2_H-128_A-2, so it needs network access.
"""
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_no_prop_overlap import (  # noqa: E402
    ENCODER_CONFIGS, PooledOutputLossTrainer, Stage2, run_stage2)
from multitask import MultiTaskModel  # noqa: E402
from multitask.config import ModelConfig, TrainingConfig  # noqa: E402

M = -1.0
NAMES = ["p1", "p2", "wa", "d1", "d2"]
OUTPUTS = ["propaganda_max_pool", "wa", "d1", "d2"]
N = 96


def data(seed, shuffle=False):
    """Rows 0-47 propaganda, 48-71 WA, 72-95 discourse. Only train is
    shuffled, as in load_splits."""
    rng = np.random.default_rng(seed)
    ids = rng.integers(999, 30522, size=(N, 16)).astype(np.int32)
    ids[:, 0], ids[:, -1] = 101, 102
    x = {"input_ids": ids, "attention_mask": np.ones_like(ids)}
    y = {k: np.full((N, 1), M, np.float32) for k in NAMES}
    for k, rows in (("p1", range(48)), ("p2", range(48)), ("wa", range(48, 72)),
                    ("d1", range(72, 96)), ("d2", range(72, 96))):
        y[k][list(rows), 0] = rng.integers(0, 2, size=len(rows))
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    return (ds.shuffle(N, seed=seed) if shuffle else ds).batch(16)


def make(encoder="small-bert"):
    tf.keras.utils.set_random_seed(0)
    cfg = ModelConfig(
        task_structure=[[3], [2, 1, 2]], task_names=NAMES, task_types=["binary"] * 5,
        num_classes_per_task=[2] * 5, shared_layer_sizes=[16], default_layer_sizes=[8],
        output_pools={"propaganda_max_pool": {"tasks": ["p1", "p2"], "strategy": "max",
                                              "task_type": "binary", "num_classes": 2}},
        encoder_config=ENCODER_CONFIGS[encoder],
    )
    model = MultiTaskModel(cfg, dropout_rate=0.0)
    trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=["p1", "p2"],
                                      pooled_class_weights={0: 1.0, 1: 2.0})
    model._mtl_loss_fn = trainer._build_loss_function(
        5, [{0: 1.0, 1: 2.0}] * 5, [1.0] * 5,
        task_type_mapping={"p1": "p", "p2": "p", "wa": "w", "d1": "d", "d2": "d"})
    train, val, test = data(0, shuffle=True), data(1), data(2)
    model(next(iter(val))[0], training=False)
    return model, trainer, train, val, test


def cached(s2, train, val, test):
    return s2.cache(train), s2.cache(val), s2.cache(test)


def test_frozen_weights_are_identical_before_and_after_stage2():
    model, trainer, train, val, test = make()
    frozen = ([v for v in model.encoder.variables]
              + [v for l in model.shared_layers for v in l.variables]
              + [v for b in model.branch_layers[0] for l in b for v in l.variables])
    assert len(frozen) > 10
    before = [v.numpy() for v in frozen]
    out = run_stage2(trainer, train, val, test, lrs=["0.05"], epochs=2, outputs=OUTPUTS,
                     batch_size=16, seed=0)
    assert out["frozen_unchanged"] is True
    for a, v in zip(before, frozen):
        np.testing.assert_array_equal(a, v.numpy())
    assert not model.encoder.trainable
    # ... while the task blocks did train: scores moved after epoch 1
    recs = out["runs"]["0.05"]["epochs"]
    assert any(recs[1]["val"][k]["auc"] != recs[0]["val"][k]["auc"] for k in OUTPUTS)


def test_learning_rate_zero_reproduces_stage1_exactly():
    model, trainer, train, val, test = make()
    out = run_stage2(trainer, train, val, test, lrs=["0"], epochs=2, outputs=OUTPUTS,
                     batch_size=16, seed=0)
    stage1, run = out["stage1"], out["runs"]["0"]
    assert run["chosen_epoch"] == {k: 0 for k in OUTPUTS}
    assert run["assembly_max_f1_gap"] == 0.0
    for rec in run["epochs"]:
        for split in ("val", "test"):
            assert rec[split] == stage1[split]
    for split in ("val", "test"):
        assert run["assembled"][split] == stage1[split]
    assert run["assembled"]["thresholds"] == stage1["thresholds"]


def test_updating_one_tasks_block_leaves_every_other_output_unchanged():
    model, trainer, train, val, test = make()
    s2 = Stage2(trainer, OUTPUTS, batch_size=16)
    s2.freeze()
    tr, va, te = cached(s2, train, val, test)
    before = s2.predict(va[0])
    blocks = s2.get_blocks()
    real_score, seen = s2.score, []

    def score(val_, test_):
        seen.append(s2.get_blocks())  # the blocks as trained, before the best is put back
        return real_score(val_, test_)

    s2.score = score
    run = s2.fit(tr, va, te, lr=0.05, epochs=1, only=["wa"])
    s2.score = real_score
    first, last = run["epochs"][0], run["epochs"][1]
    assert any(not np.array_equal(a, b) for a, b in zip(blocks["wa"], seen[-1]["wa"]))
    for k in ("propaganda_max_pool", "d1", "d2"):
        for a, b in zip(blocks[k], seen[-1][k]):
            np.testing.assert_array_equal(a, b)
    for k in ("propaganda_max_pool", "d1", "d2"):
        assert last["val"][k] == first["val"][k] and last["test"][k] == first["test"][k]
        for a, b in zip(blocks[k], s2.get_blocks()[k]):
            np.testing.assert_array_equal(a, b)
    after = s2.predict(va[0])
    for k in ("p1", "p2", "propaganda_max_pool", "d1", "d2"):
        np.testing.assert_array_equal(before[k], after[k])

    # and a task trains the same alone as alongside the others
    s2.set_blocks(blocks)
    alone = s2.fit(tr, va, te, lr=0.05, epochs=2, seed=3, only=["d1"])
    s2.set_blocks(blocks)
    together = s2.fit(tr, va, te, lr=0.05, epochs=2, seed=3)
    for a, b in zip(alone["epochs"], together["epochs"]):
        np.testing.assert_allclose(a["val"]["d1"]["auc"], b["val"]["d1"]["auc"], atol=1e-6)


def test_a_task_that_never_improves_keeps_its_stage1_block():
    model, trainer, train, val, test = make()
    s2 = Stage2(trainer, OUTPUTS, batch_size=16)
    s2.freeze()
    tr, va, te = cached(s2, train, val, test)
    stage1 = s2.get_blocks()
    real_score, seen = s2.score, []

    def score(val_, test_):
        v, t, thr = real_score(val_, test_)
        seen.append(s2.get_blocks())
        if len(seen) > 1:  # after any training, d1 looks worse and wa better on val
            v["d1"]["f1"] = -1.0
            v["wa"]["f1"] = 1.0 + len(seen)
        return v, t, thr

    s2.score = score
    run = s2.fit(tr, va, te, lr=0.05, epochs=3)
    s2.score = real_score
    assert run["chosen_epoch"]["d1"] == 0 and run["chosen_epoch"]["wa"] == 3
    # d1's block moved during training and was put back; wa kept its last epoch
    assert any(not np.array_equal(a, b) for a, b in zip(stage1["d1"], seen[-1]["d1"]))
    final = s2.get_blocks()
    for a, b in zip(stage1["d1"], final["d1"]):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(seen[-1]["wa"], final["wa"]):
        np.testing.assert_array_equal(a, b)


def test_each_output_keeps_its_own_best_epoch_and_the_assembly_scores_it():
    model, trainer, train, val, test = make()
    out = run_stage2(trainer, train, val, test, lrs=["0.05", "0.01"], epochs=4,
                     outputs=OUTPUTS, batch_size=16, seed=0)
    for lr in ("0.05", "0.01"):
        run = out["runs"][lr]
        assert [r["epoch"] for r in run["epochs"]] == [0, 1, 2, 3, 4]
        for k in OUTPUTS:
            f1s = [r["val"][k]["f1"] for r in run["epochs"]]
            e = run["chosen_epoch"][k]
            assert e == int(np.argmax(f1s))  # first best; epoch 0 wins ties
            # the assembled model scores each output as its chosen epoch did
            for split in ("val", "test"):
                assert run["assembled"][split][k]["f1"] == run["epochs"][e][split][k]["f1"]
        assert run["assembly_max_f1_gap"] == 0.0
        assert run["assembled"]["val_f1"] >= out["stage1"]["val_f1"]
    # both rates start from the same stage 1
    assert out["runs"]["0.05"]["epochs"][0]["val"] == out["runs"]["0.01"]["epochs"][0]["val"]
    assert out["runs"]["0.01"]["epochs"][0]["val"] == out["stage1"]["val"]


def test_cached_features_give_the_same_outputs_as_the_full_model():
    model, trainer, train, val, test = make()
    s2 = Stage2(trainer, OUTPUTS, batch_size=16)
    for ds in (val, test):
        feats, labels = s2.cache(ds)
        full = model.predict(ds, verbose=0)
        mine = s2.predict(feats)
        assert set(mine) == set(full) == set(NAMES) | {"propaganda_max_pool"}
        for k in full:
            np.testing.assert_allclose(mine[k], full[k], atol=1e-6)
        want = np.concatenate([y["wa"].numpy() for _, y in ds])
        np.testing.assert_array_equal(labels["wa"], want)
        assert labels["propaganda_max_pool"].shape == (N,)
