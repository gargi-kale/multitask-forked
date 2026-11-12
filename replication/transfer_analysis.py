"""Negative-transfer analysis for the runs of modal_replicate.py::transfer.

    modal run modal_replicate.py::transfer_download
    python replication/transfer_analysis.py [--runs replication/transfer_runs]

Reads <runs>/only_<task>_s<seed>.json (single-task), full_s<seed>.json (all
ten outputs, with gradient probes) and mtl_lam0_s<seed>.json (the masked-word
study's lambda-0 runs: same settings, same training). Writes a markdown
report and a per-seed CSV next to the runs.

Multi-task score for each seed: mtl_lam0_s<seed> if present, else
full_s<seed>. Delta_i = multi-task - single-task test weighted F1 for the same
seed, so the CIs are paired over seeds (t, df = n - 1), with Holm's correction
over the ten tasks. Delta_m (Maninis et al., 2019) is the mean over tasks of
(multi - single) / single, in percent.
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np
from scipy import stats

# Paper Table 3 (weighted F1, small BERT): single-task, multi-task.
PAPER = {
    "propaganda_max_pool": ("Propaganda", 63.07, 61.74),
    "disagree_agree_bi": ("Disagree/Agree", 71.15, 71.38),
    "emotion_fact_bi": ("Emotion/Fact", 68.11, 63.93),
    "attacking_respectful_bi": ("Attacking/Respectful", 67.46, 68.07),
    "nasty_nice_bi": ("Nasty/Nice", 66.90, 73.69),
    "personal_audience_bi": ("Personal/Audience", 63.25, 65.69),
    "defeater_undercutter_bi": ("Defeater/Undercutter", 45.97, 55.65),
    "negotiate_attack_bi": ("Negotiate/Attack", 64.76, 64.81),
    "questioning_asserting_bi": ("Questioning/Asserting", 59.61, 63.23),
    "WA_bi": ("Argument Quality", 80.93, 79.17),
}
SHORT = {
    "propaganda_max_pool": "Prop", "WA_bi": "AQ", "disagree_agree_bi": "D/A",
    "attacking_respectful_bi": "A/R", "emotion_fact_bi": "E/F", "nasty_nice_bi": "N/N",
    "personal_audience_bi": "P/A", "defeater_undercutter_bi": "D/U",
    "negotiate_attack_bi": "N/At", "questioning_asserting_bi": "Q/A",
}


def test_metrics(d):
    return {**d["test"]["task_metrics"], **d["test"]["pool_metrics"]}


def load(runs_dir):
    runs = {}
    for path in glob.glob(f"{runs_dir}/*.json"):
        name = os.path.basename(path)[:-5]
        m = re.fullmatch(r"(.+)_s(\d+)", name)
        if m:
            with open(path) as f:
                runs[(m.group(1), int(m.group(2)))] = json.load(f)
    return runs


def paired(deltas):
    """mean, 95% CI half-width, two-sided p of a paired t-test."""
    d = np.asarray(deltas, float)
    if len(d) < 2:
        return float(d.mean()) if len(d) else np.nan, np.nan, np.nan
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(half), float(stats.ttest_1samp(d, 0.0).pvalue)


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def ms(xs):
    xs = np.asarray(xs, float)
    return "–" if not len(xs) else (f"{xs.mean():.1f} ± {xs.std(ddof=1):.1f}" if len(xs) > 1
                                    else f"{xs[0]:.1f}")


def matrix_md(mat, names, fmt="{:+.2f}", diag=True):
    head = "| row → col | " + " | ".join(SHORT[n] for n in names) + " |"
    lines = [head, "|" + "---|" * (len(names) + 1)]
    for i, n in enumerate(names):
        cells = [("—" if (i == j and not diag) else fmt.format(mat[i][j])) for j in range(len(names))]
        lines.append(f"| **{SHORT[n]}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="replication/transfer_runs")
    ap.add_argument("--out", default=None, help="markdown report path")
    args = ap.parse_args()
    runs = load(args.runs)
    tasks = list(PAPER)

    seeds = sorted({s for (v, s) in runs if v.startswith("only_")})
    mtl_source = {}
    for s in seeds:
        if ("mtl_lam0", s) in runs:
            mtl_source[s] = ("mtl_lam0", s)
        elif ("full", s) in runs:
            mtl_source[s] = ("full", s)

    rows, csv_rows = [], []
    rel = {}  # seed -> list of (M - S) / S over tasks
    for t in tasks:
        st, mt, dl, st_auc, mt_auc, dl_auc = [], [], [], [], [], []
        for s in seeds:
            if ("only_" + t, s) not in runs or s not in mtl_source:
                continue
            a, b = test_metrics(runs[("only_" + t, s)])[t], test_metrics(runs[mtl_source[s]])[t]
            st.append(100 * a["f1"]); mt.append(100 * b["f1"]); dl.append(mt[-1] - st[-1])
            st_auc.append(100 * a["auc"]); mt_auc.append(100 * b["auc"]); dl_auc.append(mt_auc[-1] - st_auc[-1])
            rel.setdefault(s, []).append((b["f1"] - a["f1"]) / a["f1"])
            csv_rows.append({"task": t, "seed": s, "mtl_run": "_s".join(map(str, mtl_source[s])),
                             "single_f1": st[-1], "multi_f1": mt[-1], "delta_f1": dl[-1],
                             "single_auc": st_auc[-1], "multi_auc": mt_auc[-1], "delta_auc": dl_auc[-1]})
        rows.append((t, st, mt, dl, st_auc, mt_auc, dl_auc))

    ps = np.array([paired(r[3])[2] for r in rows])
    ok = ~np.isnan(ps)
    adj = np.full(len(ps), np.nan)
    if ok.any():
        adj[ok] = holm(ps[ok])

    out = []
    n_seeds = sorted({len(r[3]) for r in rows})
    out.append("# Negative-transfer study: multi-task vs single-task\n")
    out.append(f"Fine-tuned small BERT, sweep-winner settings, seeds {seeds} "
               f"(n per task: {n_seeds}). Multi-task runs per seed: "
               + ", ".join(f"s{s}: `{'_s'.join(map(str, v))}`" for s, v in mtl_source.items()) + ".\n")
    out.append("Δ = multi-task − single-task test weighted F1 ×100, paired by seed; "
               "95% CI from a t distribution; p from a paired t-test, Holm-adjusted over 10 tasks.\n")
    out.append("| Task | Paper single | Paper multi | Paper Δ | Our single | Our multi | Δ [95% CI] | p (Holm) | Verdict |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for (t, st, mt, dl, *_), p_adj in zip(rows, adj):
        label, ps_, pm_ = PAPER[t]
        mean, half, _ = paired(dl)
        if np.isnan(p_adj):
            verdict = "–"
        elif p_adj < 0.05:
            verdict = "**positive transfer**" if mean > 0 else "**negative transfer**"
        elif not np.isnan(half) and abs(mean) > half:
            verdict = ("suggestive " + ("gain" if mean > 0 else "loss")
                       + " (CI excludes 0, not after Holm)")
        else:
            verdict = "no clear difference"
        ci = f"{mean:+.1f} [{mean - half:+.1f}, {mean + half:+.1f}]" if not np.isnan(half) else f"{mean:+.1f}"
        out.append(f"| {label} | {ps_:.2f} | {pm_:.2f} | {pm_ - ps_:+.2f} | {ms(st)} | {ms(mt)} | {ci} | "
                   f"{'–' if np.isnan(p_adj) else f'{p_adj:.3f}'} | {verdict} |")
    pmean_s = np.mean([v[1] for v in PAPER.values()]); pmean_m = np.mean([v[2] for v in PAPER.values()])
    all_s = [np.mean([r[1][i] for r in rows]) for i in range(min(len(r[1]) for r in rows))] if rows else []
    all_m = [np.mean([r[2][i] for r in rows]) for i in range(min(len(r[2]) for r in rows))] if rows else []
    dm, dh, dp = paired(np.array(all_m) - np.array(all_s)) if all_s else (np.nan,) * 3
    out.append(f"| **Mean over 10** | {pmean_s:.2f} | {pmean_m:.2f} | {pmean_m - pmean_s:+.2f} | {ms(all_s)} | "
               f"{ms(all_m)} | {dm:+.1f} [{dm - dh:+.1f}, {dm + dh:+.1f}] | p={dp:.3f} (unadjusted) | |\n")

    dms = [100 * np.mean(v) for v in rel.values() if len(v) == len(tasks)]
    m_, h_, p_ = paired(dms)
    out.append(f"**Δm** (mean relative change over tasks): {m_:+.2f}% [{m_ - h_:+.2f}, {m_ + h_:+.2f}], "
               f"p = {p_:.3f}, n = {len(dms)} seeds. Paper: "
               f"{100 * np.mean([(v[2] - v[1]) / v[1] for v in PAPER.values()]):+.2f}%.\n")

    out.append("## AUC (paper reports none)\n")
    out.append("| Task | Our single AUC | Our multi AUC | Δ AUC [95% CI] |")
    out.append("|---|---|---|---|")
    for t, st, mt, dl, sa, ma, da in rows:
        mean, half, _ = paired(da)
        out.append(f"| {PAPER[t][0]} | {ms(sa)} | {ms(ma)} | {mean:+.1f} [{mean - half:+.1f}, {mean + half:+.1f}] |")
    out.append("")

    # Gradient metrics from the probed full runs.
    probed = {s: runs[("full", s)]["grad_probe"] for (v, s) in runs
              if v == "full" and runs[("full", s)].get("grad_probe")}
    if probed:
        names = next(iter(probed.values()))["tasks"]
        out.append("## Gradient metrics (shared parameters: encoder + shared layers)\n")
        out.append(f"From the probed full runs (seeds {sorted(probed)}), averaged over every probe after "
                   "step 0 (about twice per epoch) and over runs. Per-task losses are class-weighted, "
                   "without the task-type weights. Row/column labels: "
                   + ", ".join(f"{SHORT[n]} = {PAPER[n][0]}" for n in names) + ".\n")

        def avg(key, skip0=True):
            mats = [np.array(r[key]) for g in probed.values() for r in g["probes"]
                    if key in r and (r["step"] > 0 or not skip0)]
            return np.mean(mats, axis=0) if mats else None

        cos, conf = avg("cosine"), avg("conflict_rate")
        out.append("### Gradient cosine similarity\n")
        out.append(matrix_md(cos, names, diag=False) + "\n")
        out.append("### Conflict rate (share of probe batches where the two gradients have cosine < 0)\n")
        out.append(matrix_md(conf, names, fmt="{:.2f}", diag=False) + "\n")
        for key, title in (("affinity", "Adam-sized lookahead step"),
                           ("affinity_small", "a tenth of that step (near first order)")):
            aff = avg(key)
            if aff is None:
                continue
            out.append(f"### Task affinity, {title}\n")
            out.append("Row i, column j: 1 − L_j(after a step on task i) / L_j(before), ×100. "
                       "Positive = a step for task i also lowers task j's loss.\n")
            bad = [names[i] for i in range(len(names)) if aff[i][i] <= 0]
            out.append(matrix_md(100 * aff, names, fmt="{:+.2f}") + "\n")
            out.append(("Sanity check: every diagonal entry is positive.\n" if not bad else
                        f"**Warning:** non-positive diagonal for {bad}: the step overshoots, so "
                        "this matrix measures curvature as much as transfer.\n"))
        # Per-epoch trend of the mean off-diagonal cosine.
        out.append("### Mean off-diagonal cosine by epoch\n")
        out.append("| seed | " + " | ".join(f"epoch {e + 1}" for e in range(10)) + " |")
        out.append("|---|" + "---|" * 10)
        off = ~np.eye(len(names), dtype=bool)
        for s, g in sorted(probed.items()):
            by_ep = {}
            for r in g["probes"]:
                by_ep.setdefault(r["epoch"], []).append(np.array(r["cosine"])[off].mean())
            out.append(f"| {s} | " + " | ".join(f"{np.mean(by_ep[e]):+.3f}" if e in by_ep else ""
                                               for e in range(10)) + " |")
        out.append("")

        # Pairs to confirm with leave-one-task-out runs.
        aff = avg("affinity_small")
        pairs = []
        for i in range(len(names)):
            for j in range(len(names)):
                if i != j:
                    pairs.append((aff[i][j] if aff is not None else cos[i][j], names[i], names[j]))
        pairs.sort()
        out.append("### Most negative pairs (candidates for leave-one-task-out confirmation)\n")
        for v, a, b in pairs[:5]:
            out.append(f"- step on **{PAPER[a][0]}** → {PAPER[b][0]}: "
                       f"{'affinity' if aff is not None else 'cosine'} {100 * v if aff is not None else v:+.3f}, "
                       f"cosine {cos[names.index(a)][names.index(b)]:+.3f}")
        out.append("")

    # Seed-42 code-path check: the lambda-0 run against the plain probed run.
    if ("mtl_lam0", 42) in runs and ("full", 42) in runs:
        a, b = test_metrics(runs[("mtl_lam0", 42)]), test_metrics(runs[("full", 42)])
        fa = 100 * np.mean([a[t]["f1"] for t in tasks]); fb = 100 * np.mean([b[t]["f1"] for t in tasks])
        out.append(f"Seed-42 check: mean test F1 {fa:.2f} (λ=0 run) vs {fb:.2f} (plain probed run); "
                   "GPU runs aren't bit-reproducible, and earlier same-seed reruns drifted about 1 point.\n")

    out.append("Note: in the training CSVs the Propaganda rows also carry rule-derived IAC labels "
               "(e.g. Name_Calling → nasty = 1), and they're in the test set too, so the IAC tasks' "
               "single-task baselines also learn that rule.\n")

    report = args.out or os.path.join(args.runs, "transfer_report.md")
    with open(report, "w") as f:
        f.write("\n".join(out))
    with open(os.path.join(args.runs, "transfer_deltas.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]) if csv_rows else ["task"])
        w.writeheader(); w.writerows(csv_rows)
    print("\n".join(out))
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
