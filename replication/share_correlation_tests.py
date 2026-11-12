"""Do the model's predictions reproduce the label shares and label correlations?

    python replication/share_correlation_tests.py --runs <dir> --test-csv <csv> \
        --configs lr0.0001,famo_fixed,famo_consf1_m5 [--boot 2000] [--out report.md]

<dir> holds <config>_s<seed>.json and <config>_s<seed>_predictions.npz, as
written by the replication script with --save-predictions. Only the test split
is used. Hard predictions use the run's own saved thresholds (Youden's J on
val), so they are exactly the predictions behind the reported F1.

1. Shares. For each Table-3 output and each source (IAC / IBM / Propaganda)
   that has labels for it:
     true share      = fraction of labelled rows whose label is 1
     predicted share = fraction predicted 1 at the run's threshold
     mean score      = average predicted probability (no threshold involved)
   Predicted - true share equals (false positives - false negatives) / n, so
   per seed it is checked with McNemar's exact test (are FP and FN equally
   common?).

2. Correlations. For each pair of outputs and each source where both outputs
   are labelled on >= 50 rows and both labels vary, the phi coefficient (the
   Pearson correlation of two 0/1 variables) of the labels is compared with
   the phi of the hard predictions (like with like), and with the phi that
   the same predictions would have if each head's mistakes were independent
   of the other's ("independent-errors reference", computed exactly from each
   head's true- and false-positive rates). Independent mistakes pull a
   correlation toward 0, so predictions more correlated than the labels
   point to a shared cue that drives several heads together. The Pearson
   correlation of the raw scores is shown beside it.

Uncertainty: a two-level bootstrap. Each draw resamples test rows (the same
rows for every run, so comparisons between setups are paired on rows) and
then resamples seeds within each setup, and averages over the drawn seeds.
This counts both test-set noise and seed-to-seed noise. 95% intervals are
percentile intervals; p-values are two-sided z-tests using the bootstrap
standard error (a percentile p cannot go below 1/draws, which a many-test
correction would then swamp).
Corrections: Holm within each setup's family of share cells; Benjamini-
Hochberg (false discovery rate) within each setup's family of pairs; Holm
over the setup-vs-setup comparisons of the summary scores.
"""
import argparse
import glob
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

# Paper Table 3 multi-task weighted F1 (small BERT).
TABLE3 = {
    "propaganda_max_pool": ("Propaganda", 61.74),
    "disagree_agree_bi": ("Disagree/Agree", 71.38),
    "emotion_fact_bi": ("Emotion/Fact", 63.93),
    "attacking_respectful_bi": ("Attacking/Respectful", 68.07),
    "nasty_nice_bi": ("Nasty/Nice", 73.69),
    "personal_audience_bi": ("Personal/Audience", 65.69),
    "defeater_undercutter_bi": ("Defeater/Undercutter", 55.65),
    "negotiate_attack_bi": ("Negotiate/Attack", 64.81),
    "questioning_asserting_bi": ("Questioning/Asserting", 63.23),
    "WA_bi": ("Argument Quality", 79.17),
}
PAPER_MEAN = 66.74
SOURCES = ("IAC", "IBM", "Propaganda")
MIN_SHARE_ROWS, MIN_PAIR_ROWS, MIN_VARY = 30, 50, 0.02
# Labels made by rules from the technique labels, not by annotators: the IAC
# discourse heads on Propaganda rows.
HUMAN = {"propaganda_max_pool": {"Propaganda"}, "WA_bi": {"IBM"}}


def rule_made(k, s):
    return s not in HUMAN.get(k, {"IAC"})


# ----------------------------------------------------------------- loading
def load_runs(runs_dir, configs):
    runs, labels = {}, None
    for c in configs:
        for path in sorted(glob.glob(os.path.join(runs_dir, f"{c}_s*.json"))):
            m = re.fullmatch(rf"{re.escape(c)}_s(\d+)", os.path.basename(path)[:-5])
            npz = path[:-5] + "_predictions.npz"
            if not m or not os.path.exists(npz):
                continue
            with open(path) as f:
                d = json.load(f)
            z = np.load(npz)
            y = {k: z[f"test/label/{k}"].reshape(-1).astype(np.float32) for k in TABLE3}
            if labels is None:
                labels = y
            elif any(not np.array_equal(labels[k], y[k]) for k in TABLE3):
                raise SystemExit(f"{path}: test labels differ from the other runs")
            p = {k: z[f"test/prob/{k}"].reshape(-1).astype(np.float32) for k in TABLE3}
            yhat = {k: (p[k] >= d["thresholds"][k]).astype(np.float32) for k in TABLE3}
            tm = {**d["test"]["task_metrics"], **d["test"]["pool_metrics"]}
            runs.setdefault(c, []).append({
                "seed": int(m.group(1)), "p": p, "yhat": yhat,
                "f1": {k: 100 * tm[k]["f1"] for k in TABLE3}})
    missing = [c for c in configs if c not in runs]
    if missing:
        raise SystemExit(f"No runs with predictions for: {', '.join(missing)}")
    return runs, labels


# --------------------------------------------------------------- statistics
class Sums:
    """Weighted column sums for every bootstrap draw in one matrix product.
    Row 0 of the weights is all ones (the full test set), rows 1..B are draws."""

    def __init__(self):
        self.cols, self.index = [], {}

    def add(self, key, vec):
        if key not in self.index:
            self.index[key] = len(self.cols)
            self.cols.append(np.asarray(vec, np.float32))
        return self.index[key]

    def compute(self, weights):
        out = np.empty((weights.shape[0], len(self.cols)), np.float64)
        step = 512
        for a in range(0, len(self.cols), step):
            out[:, a:a + step] = weights @ np.stack(self.cols[a:a + step], 1)
        self.values = out

    def __getitem__(self, key):
        return self.values[:, self.index[key]]


def row_weights(n, boot, rng):
    w = np.empty((boot + 1, n), np.float32)
    w[0] = 1
    for b in range(1, boot + 1):
        w[b] = np.bincount(rng.integers(0, n, n), minlength=n)
    return w


def phi(n, sa, sb, sab):
    n, sa, sb, sab = (np.asarray(x, np.float64) for x in (n, sa, sb, sab))
    den = np.sqrt(np.clip(sa * (n - sa) * sb * (n - sb), 1e-12, None))
    return (n * sab - sa * sb) / den


def pearson(n, sa, sb, sab, saa, sbb):
    den = np.sqrt(np.clip((n * saa - sa ** 2) * (n * sbb - sb ** 2), 1e-12, None))
    return (n * sab - sa * sb) / den


def independent_errors_phi(n, yi, yj, yy, hi, hj, hiyi, hjyj):
    """Expected phi of two heads' predictions if, given the true labels, each
    head errs independently of the other at its observed true-positive and
    false-positive rates. Inputs are sums over the pair's rows: labels yi, yj,
    yi*yj; predictions hi, hj; hits hi*yi, hj*yj."""
    tpr_i, tpr_j = hiyi / np.maximum(yi, 1), hjyj / np.maximum(yj, 1)
    fpr_i, fpr_j = (hi - hiyi) / np.maximum(n - yi, 1), (hj - hjyj) / np.maximum(n - yj, 1)
    n11, n10, n01 = yy, yi - yy, yj - yy
    n00 = n - n11 - n10 - n01
    e11 = (n11 * tpr_i * tpr_j + n10 * tpr_i * fpr_j
           + n01 * fpr_i * tpr_j + n00 * fpr_i * fpr_j)
    return phi(n, hi, hj, e11)


def combine(per_seed, idx):
    """per_seed: (n_seeds, B+1, ...). Returns point estimate (mean over seeds on
    the full test set) and B draws, each the mean over resampled seeds."""
    point = per_seed[:, 0].mean(0)
    b = np.arange(1, per_seed.shape[1])[:, None]
    draws = per_seed[idx, b].mean(1)
    return point, draws


def boot_p(point, draws):
    """Two-sided p of a z-test with the bootstrap standard error."""
    se = np.maximum(draws.std(0, ddof=1), 1e-12)
    return 2 * stats.norm.sf(np.abs(point) / se)


def ci(draws):
    return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def holm(ps):
    ps = np.asarray(ps, float)
    order, adj, running = np.argsort(ps), np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def bh(ps):
    ps = np.asarray(ps, float)
    n, order = len(ps), np.argsort(ps)
    adj = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        running = min(running, ps[i] * n / (rank + 1))
        adj[i] = running
    return adj


def mcnemar_p(fp, fn):
    fp, fn = int(round(fp)), int(round(fn))
    return 1.0 if fp + fn == 0 else stats.binomtest(fp, fp + fn, 0.5).pvalue


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--configs", required=True, help="comma-separated run-name prefixes")
    ap.add_argument("--names", default="", help="comma-separated display names")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pair-sources", default="IAC",
                    help="sources whose rows enter the correlation tests (default IAC: "
                         "the only source with several human-labelled outputs)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    configs = args.configs.split(",")
    names = dict(zip(configs, args.names.split(","))) if args.names else {c: c for c in configs}
    runs, Y = load_runs(args.runs, configs)
    src = pd.read_csv(args.test_csv, usecols=["dataset"])["dataset"].to_numpy()
    n_rows = len(src)
    if len(Y["WA_bi"]) != n_rows:
        raise SystemExit("test CSV and predictions have different row counts")
    rng = np.random.default_rng(args.seed)
    W = row_weights(n_rows, args.boot, rng)
    seed_idx = {c: rng.integers(0, len(runs[c]), (args.boot, len(runs[c]))) for c in configs}
    S = Sums()
    lab = {k: (Y[k] >= 0) for k in TABLE3}
    y01 = {k: np.where(lab[k], Y[k], 0).astype(np.float32) for k in TABLE3}

    # ---- share cells
    cells = []
    for k in TABLE3:
        for s in SOURCES:
            m = (lab[k] & (src == s)).astype(np.float32)
            if m.sum() < MIN_SHARE_ROWS:
                continue
            cells.append((k, s))
            S.add(("m", k, s), m)
            S.add(("y", k, s), m * y01[k])
            for c in configs:
                for r in runs[c]:
                    S.add(("yhat", c, r["seed"], k, s), m * r["yhat"][k])
                    S.add(("p", c, r["seed"], k, s), m * r["p"][k])
                    S.add(("tp", c, r["seed"], k, s), m * r["yhat"][k] * y01[k])

    # ---- correlation pairs
    pairs = []
    for i, j in itertools.combinations(TABLE3, 2):
        for s in args.pair_sources.split(","):
            m = lab[i] & lab[j] & (src == s)
            if m.sum() < MIN_PAIR_ROWS:
                continue
            si, sj = y01[i][m].mean(), y01[j][m].mean()
            if not (MIN_VARY < si < 1 - MIN_VARY and MIN_VARY < sj < 1 - MIN_VARY):
                continue
            pairs.append((i, j, s))
            m = m.astype(np.float32)
            S.add(("pm", i, j, s), m)
            S.add(("pyi", i, j, s), m * y01[i])
            S.add(("pyj", i, j, s), m * y01[j])
            S.add(("pyy", i, j, s), m * y01[i] * y01[j])
            for c in configs:
                for r in runs[c]:
                    key = (c, r["seed"], i, j, s)
                    hi, hj, pi, pj = r["yhat"][i], r["yhat"][j], r["p"][i], r["p"][j]
                    for tag, vec in (("hi", hi), ("hj", hj), ("hh", hi * hj),
                                     ("hiyi", hi * y01[i]), ("hjyj", hj * y01[j]),
                                     ("si", pi), ("sj", pj), ("ss", pi * pj),
                                     ("sii", pi * pi), ("sjj", pj * pj)):
                        S.add((tag,) + key, m * vec)

    print(f"{len(cells)} share cells, {len(pairs)} pairs, {len(S.cols)} columns, "
          f"{args.boot} draws ...", flush=True)
    S.compute(W)

    out = []
    w = out.append
    w("# Shares and correlations: predictions vs labels (test set)\n")
    w(f"Bootstrap: {args.boot} draws over test rows ({n_rows}) and seeds. "
      "Hard predictions use each run's saved threshold (Youden's J on val).\n")

    # ---- summary table
    w("## Setups\n")
    w("| Setup | run name | seeds | test F1 (mean of 10, x100) | Paper (Table 3 F1) |")
    w("|---|---|---|---|---|")
    for c in configs:
        f1s = [np.mean(list(r["f1"].values())) for r in runs[c]]
        sd = f" ± {np.std(f1s, ddof=1):.1f}" if len(f1s) > 1 else ""
        w(f"| {names[c]} | {c} | {','.join(str(r['seed']) for r in runs[c])} | "
          f"{np.mean(f1s):.1f}{sd} | {PAPER_MEAN:.2f} |")
    w("")

    summary = {}  # config -> {stat: (point, draws)}

    # ---- shares
    w("## 1. Shares of 'yes'\n")
    w("Columns: true share; predicted share at the threshold and its difference "
      "from the true share with a 95% interval and Holm-adjusted p (family = all "
      "cells of that setup); mean score and its difference; McNemar = how many "
      "seeds have significantly more false positives than false negatives (+) or "
      "the reverse (−) at p < 0.05.\n")
    share_csv = []
    for c in configs:
        rs = runs[c]
        rows, ps, abs_hard, abs_soft = [], [], [], []
        for k, s in cells:
            n = S[("m", k, s)]
            true = S[("y", k, s)] / n
            hard = np.stack([S[("yhat", c, r["seed"], k, s)] / n for r in rs])
            soft = np.stack([S[("p", c, r["seed"], k, s)] / n for r in rs])
            dh, dh_draws = combine(hard - true, seed_idx[c])
            ds, ds_draws = combine(soft - true, seed_idx[c])
            abs_hard.append(np.abs(hard - true))
            abs_soft.append(np.abs(soft - true))
            mc = []
            for r in rs:
                yh, tp = S[("yhat", c, r["seed"], k, s)][0], S[("tp", c, r["seed"], k, s)][0]
                fp, fn = yh - tp, S[("y", k, s)][0] - tp
                if mcnemar_p(fp, fn) < 0.05:
                    mc.append("+" if fp > fn else "−")
            rows.append((k, s, int(n[0]), true[0], hard[:, 0].mean(), dh, ci(dh_draws),
                         boot_p(dh, dh_draws), soft[:, 0].mean(), ds, ci(ds_draws),
                         boot_p(ds, ds_draws), "".join(sorted(mc)) or "0"))
            ps.append(boot_p(dh, dh_draws))
        adj = holm(ps)
        summary[c] = {
            "mean |pred − true share|, hard": combine(np.mean(abs_hard, 0), seed_idx[c]),
            "mean |mean score − true share|": combine(np.mean(abs_soft, 0), seed_idx[c]),
        }
        w(f"### {names[c]} ({len(rs)} seeds)\n")
        w("| Output | Source | n | Paper (Table 3 F1) | our F1 | true share | pred share | "
          "pred − true [95% CI] | Holm p | mean score | score − true [95% CI] | McNemar |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (k, s, n, true, hard, dh, (lo, hi), p, soft, ds, (slo, shi), sp, mc), pa in zip(rows, adj):
            our_f1 = np.mean([r["f1"][k] for r in rs])
            s_txt = f"{s} (rule labels)" if rule_made(k, s) else s
            w(f"| {TABLE3[k][0]} | {s_txt} | {n} | {TABLE3[k][1]:.2f} | {our_f1:.1f} | "
              f"{100 * true:.1f}% | {100 * hard:.1f}% | {100 * dh:+.1f} [{100 * lo:+.1f}, "
              f"{100 * hi:+.1f}] | {fmt_p(pa)} | {100 * soft:.1f}% | {100 * ds:+.1f} "
              f"[{100 * slo:+.1f}, {100 * shi:+.1f}] | {mc} |")
            share_csv.append({"setup": c, "output": k, "source": s, "rule_labels": rule_made(k, s), "n": n,
                              "true_share": true, "pred_share": hard, "diff": dh,
                              "diff_lo": lo, "diff_hi": hi, "holm_p": pa,
                              "mean_score": soft, "score_diff": ds})
        w("")

    # ---- correlations
    w("## 2. Correlations between outputs\n")
    w("phi = correlation of two yes/no variables (−1 to 1). 'true' uses the labels, "
      "'pred' the hard predictions, 'indep.' what pred would be if the two heads' "
      "mistakes were independent, 'score r' the Pearson correlation of the raw "
      "scores. Δ = pred − true with a 95% interval; q = Benjamini–Hochberg "
      "adjusted p over this setup's pairs. Pairs need ≥ 50 rows with both labels "
      f"and both labels between {100 * MIN_VARY:.0f}% and {100 - 100 * MIN_VARY:.0f}% yes.\n")
    pair_csv = []
    for c in configs:
        rs = runs[c]
        per = {"true": [], "pred": [], "null": [], "score": []}
        for i, j, s in pairs:
            n = S[("pm", i, j, s)]
            yi, yj, yy = S[("pyi", i, j, s)], S[("pyj", i, j, s)], S[("pyy", i, j, s)]
            true = phi(n, yi, yj, yy)
            pr, nu, sc = [], [], []
            for r in rs:
                g = lambda t: S[(t, c, r["seed"], i, j, s)]
                hi, hj, hh = g("hi"), g("hj"), g("hh")
                pr.append(phi(n, hi, hj, hh))
                nu.append(independent_errors_phi(n, yi, yj, yy, hi, hj, g("hiyi"), g("hjyj")))
                sc.append(pearson(n, g("si"), g("sj"), g("ss"), g("sii"), g("sjj")))
            per["true"].append(np.broadcast_to(true, (len(rs), len(true))))
            per["pred"].append(np.stack(pr))
            per["null"].append(np.stack(nu))
            per["score"].append(np.stack(sc))
        T, P, N, R = (np.stack(per[x], -1) for x in ("true", "pred", "null", "score"))
        # T, P, N, R: (n_seeds, B+1, n_pairs)
        d_point, d_draws = combine(P - T, seed_idx[c])
        x_point, x_draws = combine(P - N, seed_idx[c])
        q = bh(boot_p(d_point, d_draws))
        qx = bh(boot_p(x_point, x_draws))
        dlo, dhi = ci(d_draws)
        summary[c].update({
            "mean |phi|, labels": combine(np.abs(T).mean(-1), seed_idx[c]),
            "mean |phi|, predictions": combine(np.abs(P).mean(-1), seed_idx[c]),
            "mean |score r|": combine(np.abs(R).mean(-1), seed_idx[c]),
            "mean (|pred phi| − |true phi|)": combine((np.abs(P) - np.abs(T)).mean(-1), seed_idx[c]),
            "RMS (pred phi − true phi)": combine(np.sqrt(((P - T) ** 2).mean(-1)), seed_idx[c]),
            "mean (pred phi − indep. phi)": combine((P - N).mean(-1), seed_idx[c]),
        })
        w(f"### {names[c]} ({len(rs)} seeds)\n")
        w("| Pair | Source | n | true phi | pred phi | Δ [95% CI] | q | indep. phi | "
          "pred − indep. q | score r |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        order = np.argsort(-np.abs(d_point))
        for t in order:
            i, j, s = pairs[t]
            w(f"| {TABLE3[i][0]} × {TABLE3[j][0]} | {s} | {int(S[('pm', i, j, s)][0])} | "
              f"{T[0, 0, t]:+.2f} | {P[:, 0, t].mean():+.2f} | {d_point[t]:+.2f} "
              f"[{dlo[t]:+.2f}, {dhi[t]:+.2f}] | {fmt_p(q[t])} | {N[:, 0, t].mean():+.2f} | "
              f"{fmt_p(qx[t])} | {R[:, 0, t].mean():+.2f} |")
            pair_csv.append({"setup": c, "a": i, "b": j, "source": s,
                             "n": int(S[("pm", i, j, s)][0]), "true_phi": T[0, 0, t],
                             "pred_phi": P[:, 0, t].mean(), "diff": d_point[t],
                             "diff_lo": dlo[t], "diff_hi": dhi[t], "bh_q": q[t],
                             "indep_phi": N[:, 0, t].mean(), "pred_minus_indep_q": qx[t],
                             "score_r": R[:, 0, t].mean()})
        w("")

    # ---- summary scores and setup-vs-setup comparisons
    w("## 3. Summary scores (one number per setup)\n")
    stats_names = list(summary[configs[0]])
    w("| Score | " + " | ".join(names[c] for c in configs) + " |")
    w("|---|" + "---|" * len(configs))
    for st in stats_names:
        cells_txt = []
        for c in configs:
            pt, dr = summary[c][st]
            lo, hi = ci(dr)
            scale = 100 if "share" in st else 1
            f = "{:.1f}" if scale == 100 else "{:.3f}"
            cells_txt.append(f"{f.format(scale * pt)} [{f.format(scale * lo)}, {f.format(scale * hi)}]")
        w(f"| {st} | " + " | ".join(cells_txt) + " |")
    w("\nShare scores are in percentage points; others in correlation units. "
      "Paper (Table 3 F1) has no counterpart for these scores; see the Setups table.\n")

    if len(configs) > 1:
        w("### Setup vs setup (row-paired bootstrap; Holm over all rows of this table)\n")
        comps = []
        for st in ("mean |pred − true share|, hard", "mean |mean score − true share|",
                   "RMS (pred phi − true phi)", "mean (|pred phi| − |true phi|)"):
            for a, b in itertools.combinations(configs, 2):
                pa, da = summary[a][st]
                pb, db = summary[b][st]
                dd = db - da
                comps.append((st, a, b, pb - pa, ci(dd), boot_p(pb - pa, dd)))
        adj = holm([x[-1] for x in comps])
        w("| Score | A | B | B − A [95% CI] | Holm p |")
        w("|---|---|---|---|---|")
        for (st, a, b, d, (lo, hi), _), pa in zip(comps, adj):
            scale = 100 if "share" in st else 1
            f = "{:+.1f}" if scale == 100 else "{:+.3f}"
            w(f"| {st} | {names[a]} | {names[b]} | {f.format(scale * d)} "
              f"[{f.format(scale * lo)}, {f.format(scale * hi)}] | {fmt_p(pa)} |")
        w("")

    text = "\n".join(out)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        base = os.path.splitext(args.out)[0]
        pd.DataFrame(share_csv).to_csv(base + "_shares.csv", index=False)
        pd.DataFrame(pair_csv).to_csv(base + "_pairs.csv", index=False)
        print(f"\nWrote {args.out}, {base}_shares.csv, {base}_pairs.csv")


if __name__ == "__main__":
    main()
