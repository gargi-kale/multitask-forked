"""Checks for the fine-tuning options in test_no_prop_overlap.py:

- ``--length-chunk-batches``: batches of similar length, padded to their own
  longest text (``length_grouped_batches``, ``trim_padding``);
- ``--layer-lr-decay`` / ``--clipnorm`` / ``--no-decay-bias-norm``
  (``LayerwiseAdamW``, ``layerwise_lr_groups``, ``build_finetune_optimizer``);
- ``--evals-per-epoch``: several scorings per epoch (``EpochScorer``).

    TF_USE_LEGACY_KERAS=1 python -m pytest replication/test_finetune_recipe.py

Downloads google/electra-small-discriminator and
google/bert_uncased_L-2_H-128_A-2, so it needs network access.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from test_electra_encoder import make_trainer  # noqa: E402
from test_gradient_probe import data, make_model  # noqa: E402
from test_no_prop_overlap import (  # noqa: E402
    DISCOURSE_BINARY_COLUMNS, ENCODER_CONFIGS, PROPAGANDA_TECHNIQUE_COLUMNS, EpochScorer,
    LayerwiseAdamW, PooledOutputLossTrainer, build_finetune_optimizer, build_parser,
    layerwise_lr_groups, length_grouped_batches, load_splits, run_paper_replication,
    trim_padding,
)
from multitask.config import TrainingConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Batches of similar length
# ---------------------------------------------------------------------------

def toy_lengths(n=1000, seed=0):
    """Three sources; source 2 has much longer texts, like the forum posts."""
    rng = np.random.default_rng(seed)
    sources = rng.choice(3, size=n, p=[0.45, 0.3, 0.25])
    lengths = np.where(sources == 2, rng.integers(10, 200, n), rng.integers(5, 40, n))
    return lengths, sources


def test_every_row_once_in_full_batches():
    lengths, sources = toy_lengths()
    batches = length_grouped_batches(lengths, sources, 32, 8, np.random.default_rng(0))
    assert len(batches) == int(np.ceil(1000 / 32))
    assert sorted(len(b) for b in batches)[1:] == [32] * (len(batches) - 1)
    np.testing.assert_array_equal(np.sort(np.concatenate(batches)), np.arange(1000))


def test_batches_keep_the_source_mix_and_save_padding():
    lengths, sources = toy_lengths(n=3200)
    rng = np.random.default_rng(0)
    grouped = length_grouped_batches(lengths, sources, 32, 8, rng)
    random = np.split(rng.permutation(3200), 100)
    overall = (sources == 2).mean()
    share = np.array([(sources[b] == 2).mean() for b in grouped])
    # every batch holds the long source at close to its overall share
    assert np.abs(share - overall).max() < 0.15, (share.min(), share.max(), overall)
    padded = lambda bs: sum(lengths[b].max() * len(b) for b in bs)
    assert padded(grouped) < 0.7 * padded(random)


def test_fresh_batches_each_epoch():
    lengths, sources = toy_lengths()
    rng = np.random.default_rng(0)
    a = length_grouped_batches(lengths, sources, 32, 8, rng)
    b = length_grouped_batches(lengths, sources, 32, 8, rng)
    assert any(not np.array_equal(x, y) for x, y in zip(a, b))


N, BATCH = 60, 8


def write(tmp_path):
    rng = np.random.default_rng(0)
    words = [" ".join(["word"] * int(k)) for k in rng.integers(1, 12, N)]
    frame = pd.DataFrame({"text": [f"row {i} {w}" for i, w in enumerate(words)],
                          "dataset": np.repeat(["A", "B", "C"], N // 3),
                          "t1": rng.integers(0, 2, N).astype(float)})
    for split in ("train", "val", "test"):
        frame.to_csv(tmp_path / f"no_prop_overlap_{split}.csv", index=False)
    return str(tmp_path), frame


def real_tokens(x):
    """Each row's token ids without padding."""
    ids, mask = x["input_ids"].numpy(), x["attention_mask"].numpy()
    return [tuple(r[m.astype(bool)]) for r, m in zip(ids, mask)]


def test_load_splits_trims_batches_and_keeps_rows_and_labels(tmp_path):
    d, frame = write(tmp_path)
    kw = dict(encoder_config=ENCODER_CONFIGS["small-bert"], batch_size=BATCH, data_dir=d,
              max_length=32, seed=0)
    full_train, full_val, _ = load_splits(["t1"], **kw)
    train, val, test = load_splits(["t1"], length_chunk_batches=2, **kw)
    assert len(train) == len(full_train) == int(np.ceil(N / BATCH))
    assert len(val) == len(full_val)

    def rows(ds):
        return {t: float(label) for x, y in ds
                for t, label in zip(real_tokens(x), y["t1"].numpy())}

    reference = rows(full_val)  # padded to max_length, CSV order
    assert len(reference) == N
    assert rows(train) == reference and rows(val) == reference
    for ds in (train, val, test):
        for x, _ in ds:
            longest = int(x["attention_mask"].numpy().sum(axis=1).max())
            assert x["input_ids"].shape[1] == x["attention_mask"].shape[1] == longest < 32
    # val keeps the CSV order; train reshuffles
    np.testing.assert_array_equal(np.concatenate([y["t1"].numpy() for _, y in val]),
                                  frame["t1"].to_numpy())
    first, second = [[real_tokens(x) for x, _ in train] for _ in range(2)]
    assert first != second


def padded_batch(n=16, width=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.zeros((n, width), np.int32)
    mask = np.zeros((n, width), np.int32)
    for i, length in enumerate(rng.integers(4, 20, n)):
        ids[i, :length] = rng.integers(999, 30522, length)
        ids[i, 0], ids[i, length - 1] = 101, 102
        mask[i, :length] = 1
    return {"input_ids": tf.constant(ids), "attention_mask": tf.constant(mask)}


@pytest.mark.parametrize("encoder", ["small-bert", "electra-small"])
def test_trimming_the_padding_leaves_the_outputs_unchanged(encoder):
    model = make_trainer(encoder=encoder).model
    x = padded_batch()
    trimmed, _ = trim_padding(x, None)
    assert trimmed["input_ids"].shape[1] < x["input_ids"].shape[1]
    full, cut = model(x, training=False), model(trimmed, training=False)
    assert set(full) == set(cut)
    for k in full:
        np.testing.assert_allclose(full[k].numpy(), cut[k].numpy(), atol=1e-5)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

MULT = {"dense/kernel:0": 0.5, "dense/bias:0": 1.0, "emb/weight:0": 0.1}


def run_steps(make_optimizer, steps, linear=False, strategy=None):
    """Variable changes after ``steps`` steps. ``linear``: a loss with
    constant gradients, so Adam's path doesn't depend on the weights."""
    scope = strategy.scope() if strategy else tf.name_scope("plain")
    with scope:
        w = tf.Variable([[1.0, -2.0], [3.0, 4.0]], name="dense/kernel")
        b = tf.Variable([0.5, -0.5], name="dense/bias")
        e = tf.Variable(tf.ones([5, 2]), name="emb/weight")
        opt = make_optimizer()
    names = {"dense/kernel:0": w, "dense/bias:0": b, "emb/weight:0": e}
    variables = [w, b, e]
    start = [v.numpy().copy() for v in variables]

    def step():
        with tf.GradientTape() as tape:
            rows = tf.gather(e, [0, 2])  # a sparse gradient, as word embeddings give
            if linear:
                loss = tf.reduce_sum(w * [[1.0, 2.0], [-3.0, 0.5]]) + 3.0 * tf.reduce_sum(b) \
                    + tf.reduce_sum(rows)
            else:
                loss = tf.reduce_sum(w * w) + 3.0 * tf.reduce_sum(b) + tf.reduce_sum(rows ** 2)
        opt.apply_gradients(zip(tape.gradient(loss, variables), variables))

    run = tf.function(lambda: strategy.run(step)) if strategy else tf.function(step)
    for _ in range(steps):
        run()
    return {name: v.numpy() - s for (name, v), s in zip(names.items(), start)}, opt


def schedule():
    return tf.keras.optimizers.schedules.PolynomialDecay(0.1, 10, end_learning_rate=0.0)


def by_name(variable):
    return MULT[variable.name.split("plain/")[-1]]


def test_multiplier_one_is_plain_adamw():
    plain, _ = run_steps(lambda: tf.keras.optimizers.AdamW(schedule(), weight_decay=0.01), 3)
    ours, _ = run_steps(lambda: LayerwiseAdamW(schedule(), weight_decay=0.01), 3)
    for k in plain:
        np.testing.assert_allclose(ours[k], plain[k], rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize("use_strategy", [False, True])
def test_each_variable_moves_by_its_multiplier(use_strategy):
    strategy = lambda: tf.distribute.OneDeviceStrategy("/cpu:0") if use_strategy else None
    # one step with weight decay: the decay step scales with the rate too
    plain, _ = run_steps(lambda: tf.keras.optimizers.AdamW(schedule(), weight_decay=0.01), 1,
                         strategy=strategy())
    ours, _ = run_steps(
        lambda: LayerwiseAdamW(schedule(), weight_decay=0.01, lr_multiplier=by_name), 1,
        strategy=strategy())
    # Changes are read as differences of float32 weights near 4, so they are
    # only exact to about 5e-7; the multipliers differ by far more than that.
    for k, m in MULT.items():
        assert np.abs(plain[k]).max() > 1e-3
        np.testing.assert_allclose(ours[k], m * plain[k], rtol=1e-4, atol=1e-6)
    # several steps of a decaying schedule, constant gradients, no weight decay
    plain, _ = run_steps(lambda: tf.keras.optimizers.AdamW(schedule(), weight_decay=0.0), 4,
                         linear=True, strategy=strategy())
    ours, opt = run_steps(
        lambda: LayerwiseAdamW(schedule(), weight_decay=0.0, lr_multiplier=by_name), 4,
        linear=True, strategy=strategy())
    for k, m in MULT.items():
        np.testing.assert_allclose(ours[k], m * plain[k], rtol=1e-4, atol=1e-6)
        assert not np.allclose(ours[k], plain[k], rtol=1e-2, atol=0) or m == 1.0
    assert int(opt.iterations) == 4 and opt._lr_scale == 1.0


def test_clipping_and_weight_decay_exclusion():
    model, _ = make_model()
    opt = build_finetune_optimizer(model, 0.1, 0.5, clipnorm=1.0, decay_bias_norm=False)
    clipped = opt._clip_gradients([tf.constant([3.0, 4.0]), tf.constant([12.0])])
    np.testing.assert_allclose(float(tf.linalg.global_norm(clipped)), 1.0, rtol=1e-6)

    w = tf.Variable([[1.0, -2.0]], name="dense/kernel")
    b = tf.Variable([0.5], name="dense/bias")
    g = tf.Variable([2.0], name="embeddings/LayerNorm/gamma")
    variables = [w, b, g]
    # zero gradients: Adam's own step is 0, so only weight decay can move anything
    opt.apply_gradients([(tf.zeros_like(v), v) for v in variables])
    np.testing.assert_allclose(w.numpy(), [[0.95, -1.9]], rtol=1e-6)  # 1 - lr * decay
    np.testing.assert_array_equal(b.numpy(), [0.5])
    np.testing.assert_array_equal(g.numpy(), [2.0])


@pytest.mark.parametrize("encoder,layers", [("electra-small", 12), ("small-bert", 2)])
def test_layerwise_groups_follow_electras_rule(encoder, layers):
    model = make_trainer(encoder=encoder).model
    model(padded_batch(), training=False)
    decay = 0.8
    groups = layerwise_lr_groups(model, decay)
    assert set(groups) == {v._unique_id for v in model.trainable_variables}
    by_group = {}
    for v in model.trainable_variables:
        group, m = groups[v._unique_id]
        by_group.setdefault(group, set()).add(round(m, 10))
        in_encoder = any(v is e for e in model.encoder.trainable_variables)
        assert (group != "task layers") == in_encoder, v.name
    assert all(len(ms) == 1 for ms in by_group.values())
    value = {g: ms.pop() for g, ms in by_group.items()}
    assert value["task layers"] == 1.0
    np.testing.assert_allclose(value["embeddings"], decay ** (layers + 2))
    for i in range(layers):
        np.testing.assert_allclose(value[f"encoder layer {i}"], decay ** (layers + 1 - i))
    np.testing.assert_allclose(value[f"encoder layer {layers - 1}"], decay ** 2)
    if encoder == "small-bert":  # BERT's pooler sits above the top layer
        np.testing.assert_allclose(value["encoder, above the layers"], decay)
    else:
        assert "encoder, above the layers" not in value
    # the word embeddings and each layer's attention weights are where expected
    name = {v._unique_id: v.name for v in model.trainable_variables}
    assert any("word_embeddings" in name[i] for i, (g, _) in groups.items() if g == "embeddings")
    assert all("/layer_._0/" in name[i] for i, (g, _) in groups.items()
               if g == "encoder layer 0")


# ---------------------------------------------------------------------------
# Several scorings per epoch
# ---------------------------------------------------------------------------

OUTPUTS = ["propaganda_max_pool", "wa", "d1", "d2"]


def in_order(seed):
    """``data`` without the shuffle: the scorer reads labels and predictions
    in separate passes, so val and test must keep their order."""
    xs, ys = zip(*[(x, y) for x, y in data(seed=seed)])
    x = {k: np.concatenate([b[k].numpy() for b in xs]) for k in xs[0]}
    y = {k: np.concatenate([b[k].numpy() for b in ys]) for k in ys[0]}
    return tf.data.Dataset.from_tensor_slices((x, y)).batch(16)


class LossCheck(tf.keras.callbacks.Callback):
    """At each epoch's end: the loss of every val batch, worked out directly,
    next to the val loss Keras logs."""

    def __init__(self, val):
        super().__init__()
        self.val, self.batch_losses, self.keras = val, [], []

    def on_epoch_end(self, epoch, logs=None):
        names = list(self.model.task_names)
        losses = []
        for x, y in self.val:
            pred = self.model(x, training=False)
            losses.append(float(self.model._mtl_loss_fn(y, {k: pred[k] for k in names})))
        self.batch_losses.append(losses)
        self.keras.append(float(logs["val_loss"]))


def fit_scorer(evals_per_epoch, patience, epochs=3):
    tf.random.set_seed(0)
    model, loss_fn = make_model()
    train, val, test = data(seed=0), in_order(1), in_order(2)
    model(next(iter(train))[0], training=False)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=loss_fn)
    scorer = EpochScorer(val, test, monitor="f1", patience=patience, outputs=OUTPUTS,
                         evals_per_epoch=evals_per_epoch)
    scorer.trainer = PooledOutputLossTrainer(model, TrainingConfig(), pooled_tasks=["p1", "p2"],
                                             pooled_class_weights={0: 1.0, 1: 1.0})
    check = LossCheck(val)
    hist = model.fit(train, epochs=epochs, validation_data=val, verbose=0,
                     callbacks=[scorer, check]).history
    return model, scorer, check, hist


def test_scores_several_times_per_epoch():
    _, scorer, check, hist = fit_scorer(evals_per_epoch=2, patience=100)
    recs = scorer.records  # 4 steps per epoch: scored after steps 2 and 4
    assert [r["epoch"] for r in recs] == [0.5, 1, 1.5, 2, 2.5, 3]
    for r in recs:
        assert set(r["val"]) == set(r["test"]) == set(OUTPUTS)
        assert np.isfinite(r["val_loss"]) and 0.0 <= r["val_f1"] <= 1.0
    # val_loss is the mean over the val batches at every scoring
    np.testing.assert_allclose([r["val_loss"] for r in recs[1::2]],
                               [np.mean(b) for b in check.batch_losses], rtol=1e-5)
    best = int(np.argmax([r["val_f1"] for r in recs]))
    assert scorer.best_index == best + 1 and scorer.best_epoch == recs[best]["epoch"]


def test_one_scoring_per_epoch_is_unchanged():
    _, scorer, check, hist = fit_scorer(evals_per_epoch=1, patience=100)
    assert [r["epoch"] for r in scorer.records] == [1, 2, 3]
    np.testing.assert_allclose([r["val_loss"] for r in scorer.records], hist["val_loss"])
    assert scorer.best_index == scorer.best_epoch
    # What Keras logs as val_loss for this model is the loss of the LAST val
    # batch, not the mean over the val batches: the model's own test_step
    # returns each batch's loss and keeps no running mean of it.
    last = [b[-1] for b in check.batch_losses]
    mean = [np.mean(b) for b in check.batch_losses]
    np.testing.assert_allclose(check.keras, last, rtol=1e-5)
    assert not np.allclose(check.keras, mean, rtol=1e-3)


def test_patience_counts_scorings_and_can_stop_inside_an_epoch():
    _, scorer, _, hist = fit_scorer(evals_per_epoch=4, patience=1, epochs=10)
    recs = scorer.records
    assert len(recs) < 40
    n = len(recs)
    for m in ("loss", "f1", "auc"):  # every metric had stalled for one scoring
        assert n - scorer._best_by(m) >= 1
    assert [r["epoch"] for r in recs] == [(i + 1) / 4 for i in range(n)]
    assert len(hist["loss"]) == int(np.ceil(n / 4))  # Keras stopped in that epoch


# ---------------------------------------------------------------------------
# The whole run, as the launcher starts it
# ---------------------------------------------------------------------------

def paper_csvs(tmp_path, n=96):
    rng = np.random.default_rng(0)
    frames = {}
    for split in ("train", "val", "test"):
        source = np.repeat(["Propaganda", "IBM", "IAC"], n // 3)
        text = [" ".join(rng.choice(["good", "bad", "people", "say", "this", "never", "why"],
                                    size=int(k))) for k in rng.integers(2, 20, n)]
        frame = pd.DataFrame({"text": text, "dataset": source})
        for col in PROPAGANDA_TECHNIQUE_COLUMNS:
            frame[col] = np.where(source == "Propaganda", rng.integers(0, 2, n), np.nan)
        frame["WA_bi"] = np.where(source == "IBM", rng.integers(0, 2, n), -1)
        for col in DISCOURSE_BINARY_COLUMNS:
            frame[col] = np.where(source == "IAC", rng.integers(0, 2, n), np.nan)
        frame.to_csv(tmp_path / f"no_prop_overlap_{split}.csv", index=False)
        frames[split] = frame
    return str(tmp_path)


def test_the_new_recipe_runs_end_to_end(tmp_path):
    """famo_consf1_m5 on ELECTRA-small with every new option on, through
    run_paper_replication, on a toy copy of the data."""
    d = paper_csvs(tmp_path)
    out = str(tmp_path / "run.json")
    args = build_parser().parse_args([
        "paper_replication", "--data-dir", d, "--encoder", "electra-small",
        "--max-length", "32", "--seed", "0", "--dropout", "0", "--epochs", "2",
        "--batch-size", "16", "--learning-rate", "3e-4", "--verbose", "0", "--arch", "wide",
        "--weight-decay", "0.01", "--warmup-percent", "5", "--pooled-propaganda-loss",
        "--loss-balancer", "famo", "--mlm-lambda", "0", "--mlm-mask-prob", "0.05",
        "--consistency-weight", "1", "--consistency-keep", "0.1",
        "--early-stop-metric", "f1", "--early-stop-patience", "100",
        "--evals-per-epoch", "2", "--length-chunk-batches", "2", "--layer-lr-decay", "0.8",
        "--clipnorm", "1.0", "--no-decay-bias-norm", "--results-json", out,
        "--save-predictions"])
    args.learning_rate_explicit = True
    tf.keras.utils.set_random_seed(0)
    run_paper_replication(args)
    res = json.load(open(out))
    recs = res["epoch_scores"]
    assert [r["epoch"] for r in recs] == [0.5, 1, 1.5, 2]
    assert res["best_epoch"] in [r["epoch"] for r in recs]
    assert len(recs[0]["test"]) == 10
    assert all(np.isfinite(r["val_f1"]) and np.isfinite(r["test_f1"]) for r in recs)
    assert res["args"]["layer_lr_decay"] == 0.8 and res["args"]["length_chunk_batches"] == 2


def test_frozen_stage2_runs_on_top_of_the_new_recipe(tmp_path):
    """--stage2 after a run with every new option on: the stage-1 model is the
    kept scoring, the frozen part stays put, and stage 2 uses its own batch
    size on batches that were padded to their own longest text."""
    d = paper_csvs(tmp_path)
    out = str(tmp_path / "run.json")
    args = build_parser().parse_args([
        "paper_replication", "--data-dir", d, "--encoder", "electra-small",
        "--max-length", "32", "--seed", "0", "--dropout", "0", "--epochs", "2",
        "--batch-size", "16", "--learning-rate", "3e-4", "--verbose", "0", "--arch", "wide",
        "--weight-decay", "0.01", "--warmup-percent", "5", "--pooled-propaganda-loss",
        "--loss-balancer", "famo", "--mlm-lambda", "0", "--mlm-mask-prob", "0.05",
        "--consistency-weight", "1", "--consistency-keep", "0.1",
        "--early-stop-metric", "f1", "--early-stop-patience", "100",
        "--evals-per-epoch", "2", "--length-chunk-batches", "2", "--layer-lr-decay", "0.8",
        "--clipnorm", "1.0", "--no-decay-bias-norm", "--results-json", out,
        "--save-predictions", "--stage2", "--stage2-lrs", "1e-3,0", "--stage2-epochs", "2",
        "--stage2-batch-size", "32", "--stage2-save-weights"])
    args.learning_rate_explicit = True
    tf.keras.utils.set_random_seed(0)
    run_paper_replication(args)
    res = json.load(open(out))
    recs, s2 = res["epoch_scores"], res["stage2"]
    assert s2["batch_size"] == 32 and s2["frozen_unchanged"]
    assert s2["cache_rows"]["train"] == 96 and s2["cache_max_abs_gap"] < 1e-4
    # stage 1 is the scoring the run kept
    kept = next(r for r in recs if r["epoch"] == res["best_epoch"])
    assert kept["val_f1"] == max(r["val_f1"] for r in recs)
    np.testing.assert_allclose(s2["stage1"]["val_f1"], kept["val_f1"], atol=1e-6)
    np.testing.assert_allclose(s2["stage1"]["test_f1"], kept["test_f1"], atol=1e-6)
    # learning rate 0 changes nothing; every output keeps an epoch no worse than stage 1
    zero = s2["runs"]["0"]
    assert set(zero["chosen_epoch"].values()) == {0}
    np.testing.assert_allclose(zero["assembled"]["test_f1"], s2["stage1"]["test_f1"], atol=1e-6)
    run = s2["runs"]["1e-3"]
    assert len(run["epochs"]) == 3 and run["assembly_max_f1_gap"] < 1e-6
    for k in s2["outputs"]:
        assert run["assembled"]["val"][k]["f1"] >= s2["stage1"]["val"][k]["f1"] - 1e-9
    assert os.path.exists(str(tmp_path / "run_stage1_weights.npz"))


def test_frozen_stage2_with_a_hidden_label(tmp_path):
    """--hide-task with --stage2: the hidden output is left out of stage 2, so
    its block (and everything else outside the trained blocks) is unchanged,
    and its probe still runs."""
    d = paper_csvs(tmp_path)
    out = str(tmp_path / "run.json")
    hidden = "nasty_nice_bi"
    args = build_parser().parse_args([
        "paper_replication", "--data-dir", d, "--encoder", "electra-small",
        "--max-length", "32", "--seed", "0", "--dropout", "0", "--epochs", "1",
        "--batch-size", "16", "--learning-rate", "3e-4", "--verbose", "0", "--arch", "wide",
        "--weight-decay", "0.01", "--warmup-percent", "5", "--pooled-propaganda-loss",
        "--loss-balancer", "famo", "--mlm-lambda", "0", "--mlm-mask-prob", "0.05",
        "--consistency-weight", "1", "--consistency-keep", "0.1",
        "--early-stop-metric", "f1", "--early-stop-patience", "100",
        "--evals-per-epoch", "2", "--length-chunk-batches", "2", "--layer-lr-decay", "0.8",
        "--clipnorm", "1.0", "--no-decay-bias-norm", "--results-json", out,
        "--save-predictions", "--stage2", "--stage2-lrs", "1e-2", "--stage2-epochs", "2",
        "--hide-task", hidden, "--probe-tasks", "hidden"])
    args.learning_rate_explicit = True
    tf.keras.utils.set_random_seed(0)
    run_paper_replication(args)
    res = json.load(open(out))
    s2 = res["stage2"]
    assert hidden not in s2["outputs"] and len(s2["outputs"]) == 9
    assert s2["frozen_unchanged"] and s2["untrained_unchanged"]
    run = s2["runs"]["1e-2"]
    assert hidden not in run["chosen_epoch"] and hidden not in run["assembled"]["test"]
    assert any(e > 0 for e in run["chosen_epoch"].values())  # stage 2 did train something
    assert hidden not in res["epoch_scores"][0]["test"]
    assert hidden in res["probes"] and "test" in res["probes"][hidden]["shared"]
