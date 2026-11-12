"""Steadier cut-offs, offline on saved predictions (no training).

Re-scores famo_consf1_m5 and the single-task models on seeds 42, 1, 2, 3
with five ways of choosing each task's cut-off from validation rows:

  saved         the cut-off stored with the run (Youden's J on val, once)
  f1 once       the cut-off with the best weighted F1 on val, once
  resampled f1  median of the best-weighted-F1 cut-off over resampled val rows
  resampled J   median of the Youden's J cut-off over resampled val rows
  rate match    cut-off at which the share of "yes" predictions on val equals
                the share of "yes" in the training labels (no tuning on F1)

A rule is picked on validation only, by held-out validation F1: the cut-off
is fitted on a random half of the validation rows and scored on the other
half. Test F1 is reported afterwards.

    python replication/steadier_cutoffs.py
"""
import json
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PRED = ROOT / "predictions"
DATA = ROOT / "multitask" / "multitask_data"
SEEDS = [42, 1, 2, 3]
TABLE3 = {  # key: (label, paper multi-task F1, paper single-task F1)
    "propaganda_max_pool": ("Propaganda", 61.74, 63.07),
    "disagree_agree_bi": ("Disagree/Agree", 71.38, 71.15),
    "emotion_fact_bi": ("Emotion/Fact", 63.93, 68.11),
    "attacking_respectful_bi": ("Attacking/Respectful", 68.07, None),
    "nasty_nice_bi": ("Nasty/Nice", 73.69, None),
    "personal_audience_bi": ("Personal/Audience", 65.69, None),
    "defeater_undercutter_bi": ("Defeater/Undercutter", 55.65, None),
    "negotiate_attack_bi": ("Negotiate/Attack", 64.81, None),
    "questioning_asserting_bi": ("Questioning/Asserting", 63.23, None),
    "WA_bi": ("Argument Quality", 79.17, None),
}
PAPER_MTL_MEAN = 66.74
RULES = ["saved", "f1 once", "resampled f1", "resampled J", "rate match"]
N_RESAMPLES = 300
N_SPLITS = 20  # random half-splits of val, each used both ways
N_RESAMPLES_SPLIT = 200


def wf1(y, p, t):
    """Weighted F1 (x100) of 'yes if p > t'."""
    yes = p > t
    pos = y == 1
    tp, fp = np.sum(yes & pos), np.sum(yes & ~pos)
    fn, tn = np.sum(~yes & pos), np.sum(~yes & ~pos)
    f_pos = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f_neg = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return 100 * (pos.sum() * f_pos + (~pos).sum() * f_neg) / len(y)


def best_cutoffs(y, p, weights, metric):
    """Best cut-off per row of `weights` (one row = one resample of the rows).

    Rows are sorted by score; 'yes for the top k' is tried for every k that
    does not split tied scores. Where several k tie for the best value (rows
    a resample left out), the middle of that stretch is used.
    """
    order = np.argsort(-p, kind="stable")
    s, pos = p[order].astype(np.float64), (y[order] == 1)
    w = np.atleast_2d(weights)[:, order].astype(np.float64)
    zero = np.zeros((w.shape[0], 1))
    tp = np.hstack([zero, np.cumsum(w * pos, axis=1)])
    fp = np.hstack([zero, np.cumsum(w * ~pos, axis=1)])
    n_pos, n_neg = tp[:, -1:], fp[:, -1:]
    fn, tn = n_pos - tp, n_neg - fp
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "f1":
            f_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
            f_neg = np.where(2 * tn + fp + fn > 0, 2 * tn / (2 * tn + fp + fn), 0.0)
            score = (n_pos * f_pos + n_neg * f_neg) / (n_pos + n_neg)
        else:  # Youden's J
            score = np.nan_to_num(tp / n_pos) - np.nan_to_num(fp / n_neg)
    n = len(s)
    valid = np.ones(n + 1, bool)
    valid[1:n] = s[:-1] != s[1:]
    score = np.where(valid, score, -np.inf)
    # cut-off for 'top k': the score of row k (rows above it are > it)
    cut = np.append(s, s[-1] - 1e-6)
    best = score.max(axis=1, keepdims=True)
    first = np.argmax(score == best, axis=1)
    idx = np.arange(n + 1)
    # end of the tied stretch: skip invalid k (inside ties) when looking for it
    breaks = (score != best) & valid & (idx > first[:, None])
    last = np.where(breaks, idx, n + 1).min(axis=1) - 1
    last = np.array([np.flatnonzero(valid[: l + 1])[-1] for l in last])
    return (cut[first] + cut[last]) / 2


def fit(rule, y, p, rate, rng, n_resamples, saved=None):
    if rule == "saved":
        if saved is not None:
            return saved
        return best_cutoffs(y, p, np.ones(len(y)), "j")[0]
    if rule == "f1 once":
        return best_cutoffs(y, p, np.ones(len(y)), "f1")[0]
    if rule == "rate match":
        return float(np.quantile(p, 1 - rate))
    counts = rng.multinomial(len(y), np.full(len(y), 1 / len(y)), size=n_resamples)
    return float(np.median(best_cutoffs(y, p, counts, "f1" if rule == "resampled f1" else "j")))


def load_runs():
    """{config: {seed: {task: dict(vy, vp, ty, tp, saved, saved_test_f1)}}}"""
    runs = {"famo_consf1_m5": {}, "single-task": {}}
    for seed in SEEDS:
        d = PRED / "paper_replication_retune_famo"
        z = np.load(d / f"famo_consf1_m5_s{seed}_predictions.npz")
        j = json.loads((d / f"famo_consf1_m5_s{seed}.json").read_text())
        runs["famo_consf1_m5"][seed] = {t: _cell(z, j, t) for t in TABLE3}
        d = PRED / "paper_replication_single_retune"
        runs["single-task"][seed] = {}
        for t in TABLE3:
            z = np.load(d / f"only_{t}_s{seed}_predictions.npz")
            j = json.loads((d / f"only_{t}_s{seed}.json").read_text())
            runs["single-task"][seed][t] = _cell(z, j, t)
    return runs


def _cell(z, j, task):
    best = next(e for e in j["epoch_scores"] if e["epoch"] == j["best_epoch"])
    out = {"saved": j["thresholds"][task], "saved_test_f1": 100 * best["test"][task]["f1"]}
    for split, a in (("val", "v"), ("test", "t")):
        y = z[f"{split}/label/{task}"].reshape(-1)
        p = z[f"{split}/prob/{task}"].reshape(-1)
        keep = y != -1
        out[a + "y"], out[a + "p"] = y[keep], p[keep]
    return out


def train_rates():
    tr = pd.read_csv(DATA / "no_prop_overlap_train.csv", low_memory=False,
                     usecols=["propaganda"] + [t for t in TABLE3 if t != "propaganda_max_pool"])
    tr = tr.rename(columns={"propaganda": "propaganda_max_pool"})
    return {t: float((tr[t] == 1).sum() / ((tr[t] == 1) | (tr[t] == 0)).sum()) for t in TABLE3}


def ms(xs):
    return f"{statistics.mean(xs):.1f} ± {statistics.stdev(xs):.1f}"


def paired(a, b):
    """Mean difference, 95% t interval and two-sided paired t-test p (4 seeds, 3 df)."""
    d = [x - y for x, y in zip(a, b)]
    assert len(d) == 4
    m, se = statistics.mean(d), statistics.stdev(d) / len(d) ** 0.5
    h = 3.182446 * se  # t(0.975, 3 df)
    x = abs(m / se) / math.sqrt(3) if se else math.inf
    p = 1 - (2 / math.pi) * (x / (1 + x * x) + math.atan(x)) if se else 0.0  # t cdf, 3 df
    return f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}] p={p:.3f}"


def main():
    rng = np.random.default_rng(0)
    runs, rates = load_runs(), train_rates()
    configs = list(runs)

    worst = max(abs(wf1(c["ty"], c["tp"], c["saved"]) - c["saved_test_f1"])
                for r in runs.values() for s in r.values() for c in s.values())
    print(f"Check: test F1 at the saved cut-off vs the F1 stored with the run, "
          f"largest difference over 80 task-runs: {worst:.4f}")

    # res[config][rule][seed][task] = dict(cut, val, test, held)
    res = {c: {r: {s: {} for s in SEEDS} for r in RULES} for c in configs}
    for c in configs:
        for s in SEEDS:
            for t, cell in runs[c][s].items():
                vy, vp = cell["vy"], cell["vp"]
                halves = []
                for _ in range(N_SPLITS):
                    perm = rng.permutation(len(vy))
                    a, b = perm[: len(vy) // 2], perm[len(vy) // 2:]
                    halves += [(a, b), (b, a)]
                for rule in RULES:
                    cut = fit(rule, vy, vp, rates[t], rng, N_RESAMPLES, saved=cell["saved"])
                    held = [wf1(vy[b], vp[b], fit(rule, vy[a], vp[a], rates[t], rng,
                                                  N_RESAMPLES_SPLIT))
                            for a, b in halves]
                    half_cuts = [fit(rule, vy[a], vp[a], rates[t], rng, N_RESAMPLES_SPLIT)
                                 for a, _ in halves[:10]]
                    res[c][rule][s][t] = {
                        "cut": cut, "val": wf1(vy, vp, cut),
                        "test": wf1(cell["ty"], cell["tp"], cut),
                        "held": statistics.mean(held),
                        "wobble": statistics.stdev(half_cuts),
                        "test_yes": 100 * float(np.mean(cell["tp"] > cut)),
                    }
        print(f"  scored {c}")

    def per_seed(c, rule, field):
        return [statistics.mean(res[c][rule][s][t][field] for t in TABLE3) for s in SEEDS]

    print("\nTraining share of 'yes' per task (rate-matching target), %")
    for t, (label, *_) in TABLE3.items():
        test_share = statistics.mean(100 * float(np.mean(runs["single-task"][s][t]["ty"] == 1))
                                     for s in SEEDS)
        val_share = 100 * float(np.mean(runs["single-task"][42][t]["vy"] == 1))
        print(f"  {label:<24} train {100 * rates[t]:5.1f}   val {val_share:5.1f}   "
              f"test {test_share:5.1f}")

    print("\nMean over the 10 Table-3 outputs, weighted F1 x100, mean ± sd over 4 seeds")
    print("held-out val F1: cut-off fitted on half the val rows, scored on the other half")
    print("wobble: sd of the cut-off across random val halves, mean over tasks (lower = steadier)")
    for c in configs:
        print(f"\n{c}")
        print(f"{'rule':<14}{'val F1':>12}{'held-out val F1':>18}{'test F1':>12}"
              f"{'val-test':>10}{'wobble':>9}{'Paper (Table 3 F1)':>20}")
        for rule in RULES:
            v, h, t = (per_seed(c, rule, f) for f in ("val", "held", "test"))
            print(f"{rule:<14}{ms(v):>12}{ms(h):>18}{ms(t):>12}"
                  f"{statistics.mean(v) - statistics.mean(t):>10.1f}"
                  f"{statistics.mean(per_seed(c, rule, 'wobble')):>9.3f}"
                  f"{PAPER_MTL_MEAN:>20.2f}")

    print("\nGap, famo_consf1_m5 minus single-task, paired by seed, 95% t interval "
          "(paper Table 3: multi-task 66.74 vs single-task +3.2% Δm in favour of multi-task)")
    print(f"{'rule':<14}{'held-out val F1 gap':>34}{'test F1 gap':>34}")
    for rule in RULES:
        print(f"{rule:<14}"
              f"{paired(per_seed(configs[0], rule, 'held'), per_seed(configs[1], rule, 'held')):>34}"
              f"{paired(per_seed(configs[0], rule, 'test'), per_seed(configs[1], rule, 'test')):>34}")

    print("\nEach rule against the saved cut-off, same runs, paired by seed (test F1)")
    for c in configs:
        for rule in RULES[1:]:
            print(f"  {c:<16}{rule:<14}"
                  f"{paired(per_seed(c, rule, 'test'), per_seed(c, 'saved', 'test'))}"
                  f"   held-out val: "
                  f"{paired(per_seed(c, rule, 'held'), per_seed(c, 'saved', 'held'))}")

    for field, title in (("test", "Test"), ("held", "Held-out val")):
        print(f"\n{title} weighted F1 per task x100, mean over 4 seeds "
              "(F = famo_consf1_m5, S = single-task)")
        print(f"{'Task':<24}{'Paper (Table 3 F1)':>19}"
              + "".join(f"{'F ' + r:>15}{'S ' + r:>15}" for r in RULES))
        for t, (label, paper, _) in TABLE3.items():
            print(f"{label:<24}{paper:>19.2f}" + "".join(
                f"{statistics.mean(res[c][r][s][t][field] for s in SEEDS):>15.1f}"
                for r in RULES for c in configs))

    print("\nShare of test rows predicted 'yes' per task, %, mean over 4 seeds "
          "(F = famo_consf1_m5, S = single-task)")
    print(f"{'Task':<24}{'true yes':>9}" + "".join(f"{'F ' + r:>15}{'S ' + r:>15}" for r in RULES))
    for t, (label, *_) in TABLE3.items():
        true = statistics.mean(100 * float(np.mean(runs["single-task"][s][t]["ty"] == 1))
                               for s in SEEDS)
        print(f"{label:<24}{true:>9.1f}" + "".join(
            f"{statistics.mean(res[c][r][s][t]['test_yes'] for s in SEEDS):>15.1f}"
            for r in RULES for c in configs))

    out = ROOT / "replication" / "steadier_cutoffs_results.json"
    out.write_text(json.dumps({"train_rates": rates, "results": {
        c: {r: {str(s): res[c][r][s] for s in SEEDS} for r in RULES} for c in configs}}, indent=1))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
