"""Is the extra correlation between the IAC heads' predictions harmful?

    python replication/coupling_tests.py --runs <dir> --test-csv <csv> --val-csv <csv> \
        --configs lr0.0001,famo_consf1_m5 [--names Plain,Best] [--boot 1000] [--out report.md]

Predictions more correlated than labels are not wrong by themselves: labels
are noisy, and removing noise can raise a correlation. Two checks separate
harmless from harmful coupling. Both use IAC rows only (the one source where
annotators labelled all eight discourse outputs).

1. Leak test (test rows). For each ordered pair (i, j), a logistic regression
   of label i on a cubic in head i's standardised logit plus head j's
   standardised logit. If head i's score already held everything the text says
   about label i, head j's score could add nothing, so its coefficient would
   be 0, however correlated the two heads are. A negative coefficient means
   head i carries part of head j's signal that does not belong to label i
   (over-coupling); positive means head i misses information head j has. The
   cubic absorbs non-linear miscalibration of head i, so head j cannot score
   by standing in for it. Seeds: mean coefficient, with the mean of the
   per-seed standard errors (conservative, since the seeds share test rows).
   Benjamini-Hochberg over the 56 ordered pairs.

2. Cost test (fit on val, score on test). For each head, a logistic
   regression of its label on all eight heads' logits, fitted on val IAC
   rows, then applied to test IAC rows. The decision threshold is picked by
   Youden's J on val for both the original and the recombined score. The
   gain in AUC and weighted F1 is the accuracy the coupling costs that a
   post-hoc fix recovers; it also shows whether the recombined predictions'
   correlations move back toward the labels'. Intervals: bootstrap over test
   rows and seeds.
"""
import argparse
import glob
import itertools
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

IAC = {
    "disagree_agree_bi": ("Disagree/Agree", 71.38),
    "emotion_fact_bi": ("Emotion/Fact", 63.93),
    "attacking_respectful_bi": ("Attacking/Respectful", 68.07),
    "nasty_nice_bi": ("Nasty/Nice", 73.69),
    "personal_audience_bi": ("Personal/Audience", 65.69),
    "defeater_undercutter_bi": ("Defeater/Undercutter", 55.65),
    "negotiate_attack_bi": ("Negotiate/Attack", 64.81),
    "questioning_asserting_bi": ("Questioning/Asserting", 63.23),
}
HEADS = list(IAC)
EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load(runs_dir, config, val_src, test_src):
    out = []
    for path in sorted(glob.glob(os.path.join(runs_dir, f"{config}_s*.json"))):
        m = re.fullmatch(rf"{re.escape(config)}_s(\d+)", os.path.basename(path)[:-5])
        npz = path[:-5] + "_predictions.npz"
        if not m or not os.path.exists(npz):
            continue
        z = np.load(npz)
        run = {"seed": int(m.group(1))}
        for split, src in (("val", val_src), ("test", test_src)):
            iac = src == "IAC"
            run[split] = {
                "y": np.stack([z[f"{split}/label/{k}"].reshape(-1)[iac] for k in HEADS], 1),
                "z": np.stack([logit(z[f"{split}/prob/{k}"].reshape(-1)[iac]) for k in HEADS], 1),
            }
        out.append(run)
    if not out:
        raise SystemExit(f"no runs for {config}")
    return out


def youden(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    return thr[np.argmax(tpr - fpr)]


def phi(a, b):
    return np.corrcoef(a.astype(float), b.astype(float))[0, 1]


def bh(ps):
    ps = np.asarray(ps, float)
    n, order, adj, running = len(ps), np.argsort(ps), np.empty(len(ps)), 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        running = min(running, ps[i] * n / (rank + 1))
        adj[i] = running
    return adj


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------- 1. leak test
def leak_test(runs):
    rows = []
    for a, b in itertools.permutations(range(len(HEADS)), 2):
        coefs, ses = [], []
        for r in runs:
            y, z = r["test"]["y"][:, a], r["test"]["z"]
            m = y >= 0
            zi = (z[m, a] - z[m, a].mean()) / z[m, a].std()
            zj = (z[m, b] - z[m, b].mean()) / z[m, b].std()
            X = sm.add_constant(np.column_stack([zi, zi ** 2, zi ** 3, zj]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sm.Logit(y[m], X).fit(disp=0)
            coefs.append(fit.params[-1])
            ses.append(fit.bse[-1])
        c, se = float(np.mean(coefs)), float(np.mean(ses))
        rows.append({"label": HEADS[a], "added_head": HEADS[b], "coef": c, "se": se,
                     "p": 2 * stats.norm.sf(abs(c) / se),
                     "seeds_same_sign_sig": sum(abs(k / s) > 1.96 and np.sign(k) == np.sign(c)
                                                for k, s in zip(coefs, ses))})
    df = pd.DataFrame(rows)
    df["q"] = bh(df.p)
    return df


# ------------------------------------------------------------- 2. cost test
def cost_test(runs, boot, rng):
    per = []  # per seed: dict of arrays on test rows
    for r in runs:
        V, T = r["val"], r["test"]
        res = {"orig_s": [], "stack_s": [], "orig_h": [], "stack_h": [], "y": [], "m": [],
               "weights": []}
        for a in range(len(HEADS)):
            mv, mt = V["y"][:, a] >= 0, T["y"][:, a] >= 0
            mu, sd = V["z"].mean(0), V["z"].std(0)
            Xv, Xt = (V["z"] - mu) / sd, (T["z"] - mu) / sd
            lr = LogisticRegression(C=1e6, max_iter=2000).fit(Xv[mv], V["y"][mv, a])
            sv, st = lr.decision_function(Xv), lr.decision_function(Xt)
            t_orig, t_stack = youden(V["y"][mv, a], V["z"][mv, a]), youden(V["y"][mv, a], sv[mv])
            res["orig_s"].append(T["z"][:, a]); res["stack_s"].append(st)
            res["orig_h"].append(T["z"][:, a] >= t_orig); res["stack_h"].append(st >= t_stack)
            res["y"].append(T["y"][:, a]); res["m"].append(mt)
            res["weights"].append(lr.coef_[0])
        per.append({k: np.array(v) for k, v in res.items()})

    n = per[0]["y"].shape[1]

    def scores(idx):
        """(n_seeds, heads, 4): AUC orig, AUC stacked, F1 orig, F1 stacked on rows idx."""
        out = np.empty((len(per), len(HEADS), 4))
        for s, P in enumerate(per):
            for a in range(len(HEADS)):
                ii = idx[P["m"][a][idx]]
                y = P["y"][a][ii]
                out[s, a] = (roc_auc_score(y, P["orig_s"][a][ii]),
                             roc_auc_score(y, P["stack_s"][a][ii]),
                             f1_score(y, P["orig_h"][a][ii], average="weighted"),
                             f1_score(y, P["stack_h"][a][ii], average="weighted"))
        return out

    point = scores(np.arange(n))
    draws = np.empty((boot, len(HEADS), 4))
    for b in range(boot):
        rows = rng.integers(0, n, n)
        seeds = rng.integers(0, len(per), len(per))
        draws[b] = scores(rows)[seeds].mean(0)

    # correlations of hard predictions, before and after, on fully labelled rows
    pairs = []
    for a, b in itertools.combinations(range(len(HEADS)), 2):
        m = per[0]["m"][a] & per[0]["m"][b]
        y = per[0]["y"]
        pairs.append((phi(y[a][m], y[b][m]),
                      np.mean([phi(P["orig_h"][a][m], P["orig_h"][b][m]) for P in per]),
                      np.mean([phi(P["stack_h"][a][m], P["stack_h"][b][m]) for P in per])))
    weights = np.mean([P["weights"] for P in per], 0)
    return point.mean(0), draws, np.array(pairs), weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--names", default="")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    configs = args.configs.split(",")
    names = dict(zip(configs, args.names.split(","))) if args.names else {c: c for c in configs}
    val_src = pd.read_csv(args.val_csv, usecols=["dataset"])["dataset"].to_numpy()
    test_src = pd.read_csv(args.test_csv, usecols=["dataset"])["dataset"].to_numpy()
    rng = np.random.default_rng(args.seed)
    short = {k: v[0] for k, v in IAC.items()}

    out = ["# Is the extra correlation harmful? (IAC rows)\n"]
    w = out.append
    leak_all = []
    for c in configs:
        runs = load(args.runs, c, val_src, test_src)
        w(f"## {names[c]} ({c}, seeds {','.join(str(r['seed']) for r in runs)})\n")

        leak = leak_test(runs)
        leak_all.append(leak.assign(setup=c))
        sig = leak[leak.q < 0.05]
        w(f"### 1. Leak test\n\n{len(sig)} of {len(leak)} ordered pairs have a "
          f"coefficient ≠ 0 at q < 0.05: {int((sig.coef < 0).sum())} negative "
          f"(head carries another head's signal), {int((sig.coef > 0).sum())} positive "
          "(head misses information the other head has). Coefficient = change in the "
          "log-odds of the label per 1 SD of the added head's logit, holding the "
          "label's own head fixed.\n")
        w("| Label | Added head | coefficient [95% CI] | q | seeds sig. same sign |")
        w("|---|---|---|---|---|")
        for _, r in leak.sort_values("p").iterrows():
            if r.q >= 0.05:
                continue
            w(f"| {short[r.label]} | {short[r.added_head]} | {r.coef:+.2f} "
              f"[{r.coef - 1.96 * r.se:+.2f}, {r.coef + 1.96 * r.se:+.2f}] | "
              f"{fmt_p(r.q)} | {r.seeds_same_sign_sig}/{len(runs)} |")
        w("")

        point, draws, pairs, weights = cost_test(runs, args.boot, rng)
        w("### 2. Cost test: recombine the eight scores (fit on val, scored on test)\n")
        w("Weighted F1 and AUC on test IAC rows only (the paper's Table 3 F1 is over "
          "all labelled test rows, so it is shown for reference, not as a like-for-like "
          "comparison). Δ = recombined − original, 95% interval.\n")
        w("| Output | Paper (Table 3 F1) | F1 original | F1 recombined | ΔF1 [95% CI] | "
          "AUC original | AUC recombined | ΔAUC [95% CI] |")
        w("|---|---|---|---|---|---|---|---|")
        df1 = draws[:, :, 3] - draws[:, :, 2]
        dauc = draws[:, :, 1] - draws[:, :, 0]
        for a, k in enumerate(HEADS):
            lo, hi = np.percentile(df1[:, a], [2.5, 97.5])
            alo, ahi = np.percentile(dauc[:, a], [2.5, 97.5])
            w(f"| {short[k]} | {IAC[k][1]:.2f} | {100 * point[a, 2]:.1f} | {100 * point[a, 3]:.1f} | "
              f"{100 * (point[a, 3] - point[a, 2]):+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
              f"{point[a, 0]:.3f} | {point[a, 1]:.3f} | {point[a, 1] - point[a, 0]:+.3f} "
              f"[{alo:+.3f}, {ahi:+.3f}] |")
        mf = df1.mean(1)
        ma = dauc.mean(1)
        w(f"| **Mean of 8** | {np.mean([v[1] for v in IAC.values()]):.2f} | "
          f"{100 * point[:, 2].mean():.1f} | {100 * point[:, 3].mean():.1f} | "
          f"{100 * (point[:, 3] - point[:, 2]).mean():+.1f} "
          f"[{100 * np.percentile(mf, 2.5):+.1f}, {100 * np.percentile(mf, 97.5):+.1f}] | "
          f"{point[:, 0].mean():.3f} | {point[:, 1].mean():.3f} | "
          f"{(point[:, 1] - point[:, 0]).mean():+.3f} "
          f"[{np.percentile(ma, 2.5):+.3f}, {np.percentile(ma, 97.5):+.3f}] |\n")
        w(f"Mean |phi| over the 28 pairs: labels {np.abs(pairs[:, 0]).mean():.3f}, original "
          f"predictions {np.abs(pairs[:, 1]).mean():.3f}, recombined predictions "
          f"{np.abs(pairs[:, 2]).mean():.3f}.\n")
        w("Recombination weights (rows: label predicted; columns: heads' standardised "
          "logits; mean over seeds):\n")
        w("| Label | " + " | ".join(short[k] for k in HEADS) + " |")
        w("|---|" + "---|" * len(HEADS))
        for a, k in enumerate(HEADS):
            w(f"| {short[k]} | " + " | ".join(f"{x:+.2f}" for x in weights[a]) + " |")
        w("")

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        pd.concat(leak_all).to_csv(os.path.splitext(args.out)[0] + "_leak.csv", index=False)


if __name__ == "__main__":
    main()
