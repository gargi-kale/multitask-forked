"""Summaries of finished runs (``multitask report RUNS_DIR``).

Runs that differ only in seed are one row (mean, and the standard deviation
over seeds when there are several). Single-task runs (one per output) are
combined into one "single_task" row. Scores are weighted F1 and AUC x100 at
the epoch the run kept; "stage 2" is frozen stage 2 at the learning rate with
the best validation F1.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

from multitask.train.runner import RESULT_FILE


def load_results(root: str) -> List[Dict[str, Any]]:
    out = []
    for path in sorted(glob.glob(os.path.join(root, "**", RESULT_FILE), recursive=True)):
        with open(path) as f:
            r = json.load(f)
        r["_path"] = path
        out.append(r)
    return out


def _setup(run: str) -> Tuple[str, Optional[int]]:
    """'famo_consf1_m5_stage2_s42' -> ('famo_consf1_m5_stage2', 42);
    'single_task_<output>_s42' -> ('single_task', 42)."""
    m = re.match(r"^(.*)_s(\d+)$", run)
    name, seed = (m.group(1), int(m.group(2))) if m else (run, None)
    if name.startswith("single_task_"):
        name = "single_task"
    return name, seed


def stage2_scores(r: Dict[str, Any]) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
    s2 = r.get("stage2")
    if not s2 or not s2.get("runs"):
        return None
    best = max(s2["runs"].values(), key=lambda run: run["assembled"]["val_f1"])
    a = best["assembled"]
    return {split: {o: {k: a[split][o][k] for k in ("f1", "auc")} for o in a[split]
                    if isinstance(a[split].get(o), dict)} for split in ("val", "test")}


def _per_output(r: Dict[str, Any], split: str, stage2: bool = False) -> Dict[str, Dict[str, float]]:
    if stage2:
        s = stage2_scores(r)
        return s[split] if s else {}
    return r.get(split, {})


def summarize(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per setup: per-output and mean scores over seeds."""
    setups: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in results:
        if r.get("untrained"):
            continue
        name, seed = _setup(r["run"])
        key = (r["dataset"], name)
        row = setups.setdefault(key, {"dataset": r["dataset"], "setup": name, "seeds": set(),
                                      "outputs": [], "labels": {}, "reference": r.get("reference"),
                                      "hidden": r.get("hidden_output"), "cells": {},
                                      "stage2": False})
        row["seeds"].add(seed)
        row["labels"].update(r.get("labels", {}))
        has_s2 = stage2_scores(r) is not None
        row["stage2"] |= has_s2
        for o in r.get("scored_outputs", r.get("outputs", [])):
            if o not in row["outputs"]:
                row["outputs"].append(o)
            cell = row["cells"].setdefault(o, {})
            for split in ("val", "test"):
                for metric in ("f1", "auc"):
                    v = r.get(split, {}).get(o, {}).get(metric)
                    if v is not None:
                        cell.setdefault(f"{split}_{metric}", []).append(100 * v)
                    if has_s2:
                        v2 = _per_output(r, split, True).get(o, {}).get(metric)
                        if v2 is not None:
                            cell.setdefault(f"s2_{split}_{metric}", []).append(100 * v2)
        hidden = r.get("hidden_output")
        if hidden and r.get("probes", {}).get(hidden):
            best = r["probes"][hidden]
            cell = row["cells"].setdefault(f"{hidden} (hidden, probe)", {})
            for rep in ("encoder", "shared", "group"):
                t = best.get(rep, {}).get("test")
                if t:
                    cell.setdefault(f"probe_{rep}_auc", []).append(100 * t["auc"])
        if r.get("hidden_cells"):
            cell = row["cells"].setdefault("hidden train cells", {})
            sc = [v for v in r["hidden_cells"]["scores"].values() if v.get("f1") is not None]
            if sc:
                cell.setdefault("hid_f1", []).append(100 * statistics.mean(v["f1"] for v in sc))
                cell.setdefault("hid_auc", []).append(100 * statistics.mean(v["auc"] for v in sc))
    rows = list(setups.values())
    for row in rows:
        row["seeds"] = sorted(s for s in row["seeds"] if s is not None)
        means = {}
        for key in ("val_f1", "test_f1", "test_auc", "s2_val_f1", "s2_test_f1", "s2_test_auc"):
            per_output = [statistics.mean(row["cells"][o][key]) for o in row["outputs"]
                          if row["cells"].get(o, {}).get(key)]
            if per_output and len(per_output) == len(row["outputs"]):
                means[key] = statistics.mean(per_output)
        row["means"] = means
    return rows


def _fmt(vals: Optional[List[float]]) -> str:
    if not vals:
        return "    -     "
    if len(vals) == 1:
        return f"{vals[0]:6.2f}    "
    return f"{statistics.mean(vals):6.2f}±{statistics.stdev(vals):<3.1f}"


def render(rows: List[Dict[str, Any]], detail: bool = True) -> str:
    lines = []
    lines.append(f"{'Dataset':<16}{'Setup':<44}{'Seeds':<10}{'Val F1':>8}{'Test F1':>9}"
                 f"{'Test AUC':>10}{'Stage-2 test F1':>17}")
    for row in sorted(rows, key=lambda r: (r["dataset"], r["setup"])):
        m = row["means"]
        seeds = ",".join(map(str, row["seeds"])) or "-"
        s2 = f"{m['s2_test_f1']:.2f}" if "s2_test_f1" in m else ""
        lines.append(f"{row['dataset'][:15]:<16}{row['setup'][:43]:<44}{seeds[:9]:<10}"
                     f"{m.get('val_f1', float('nan')):8.2f}{m.get('test_f1', float('nan')):9.2f}"
                     f"{m.get('test_auc', float('nan')):10.2f}{s2:>17}")
    if not detail:
        return "\n".join(lines)
    for row in sorted(rows, key=lambda r: (r["dataset"], r["setup"])):
        ref = (row.get("reference") or {}).get("f1", {})
        title = (row.get("reference") or {}).get("title", "Reference")
        lines.append("")
        lines.append(f"== {row['dataset']} / {row['setup']} (seeds {row['seeds'] or '-'}): "
                     "weighted F1 and AUC x100, mean±sd over seeds ==")
        head = f"{'Output':<30}" + (f"{'Ref F1':>8}" if ref else "") + \
            f"{'Val F1':>11}{'Test F1':>11}{'Test AUC':>11}"
        if row["stage2"]:
            head += f"{'S2 test F1':>12}{'S2 test AUC':>12}"
        lines.append(head + (f"   ({title})" if ref else ""))
        for o in row["outputs"]:
            c = row["cells"].get(o, {})
            label = row["labels"].get(o, o)
            line = f"{label[:29]:<30}" + (f"{ref[o]:8.2f}" if o in ref else ("       -" if ref else ""))
            line += f" {_fmt(c.get('val_f1'))} {_fmt(c.get('test_f1'))} {_fmt(c.get('test_auc'))}"
            if row["stage2"]:
                line += f" {_fmt(c.get('s2_test_f1'))} {_fmt(c.get('s2_test_auc'))}"
            lines.append(line)
        m = row["means"]
        if m:
            ref_mean = (sum(ref[o] for o in row["outputs"] if o in ref) / len(ref)
                        if ref and all(o in ref for o in row["outputs"]) else None)
            line = f"{'Mean':<30}" + (f"{ref_mean:8.2f}" if ref_mean is not None else
                                      ("       -" if ref else ""))
            line += "".join(f" {m[k]:6.2f}    " if k in m else "     -     "
                            for k in ("val_f1", "test_f1", "test_auc"))
            if row["stage2"]:
                line += "".join(f" {m[k]:6.2f}    " if k in m else "     -     "
                                for k in ("s2_test_f1", "s2_test_auc"))
            lines.append(line)
        for name, c in row["cells"].items():
            if name.endswith("(hidden, probe)"):
                lines.append(f"{name}: probe test AUC encoder {_fmt(c.get('probe_encoder_auc'))} "
                             f"shared {_fmt(c.get('probe_shared_auc'))} "
                             f"group {_fmt(c.get('probe_group_auc'))}")
            if name == "hidden train cells":
                lines.append(f"Hidden train cells (mean over labels): F1 {_fmt(c.get('hid_f1'))} "
                             f"AUC {_fmt(c.get('hid_auc'))}")
    return "\n".join(lines)


def hide_label_table(results: List[Dict[str, Any]]) -> str:
    """Leave-one-label-out summary: for each `<setup>_hide-<output>` run with a
    `<setup>_full` run of the same seed, the mean test F1 / AUC of the other
    outputs in both runs and the change, and the hidden output's probe AUC
    (shared layers) against the probe on the full run."""
    by_key = {}
    for r in results:
        name, seed = _setup(r["run"])
        by_key[(r["dataset"], name, seed)] = r
    lines = []
    for (dataset, name, seed), r in sorted(by_key.items(), key=lambda kv: str(kv[0])):
        m = re.match(r"^(.*)_hide-(.+)$", name)
        if not m:
            continue
        full = by_key.get((dataset, f"{m.group(1)}_full", seed))
        hidden = r.get("hidden_output")
        if not full or not hidden:
            continue
        others = [o for o in r["scored_outputs"] if o in full.get("test", {})]

        def mean(run, metric):
            vals = [run["test"][o][metric] for o in others if metric in run["test"].get(o, {})]
            return 100 * statistics.mean(vals) if vals else float("nan")

        def probe(run):
            t = (run.get("probes") or {}).get(hidden, {}).get("shared", {}).get("test")
            return 100 * t["auc"] if t else float("nan")

        lines.append(f"{dataset[:15]:<16}{m.group(1)[:24]:<25}{seed!s:<6}{hidden[:26]:<27}"
                     f"{mean(r, 'f1'):8.2f}{mean(full, 'f1'):8.2f}"
                     f"{mean(r, 'f1') - mean(full, 'f1'):+8.2f}"
                     f"{mean(r, 'auc') - mean(full, 'auc'):+9.2f}"
                     f"{probe(r):9.2f}{probe(full):9.2f}")
    if not lines:
        return ""
    head = (f"{'Dataset':<16}{'Setup':<25}{'Seed':<6}{'Hidden output':<27}"
            f"{'Others':>8}{'Full':>8}{'ΔF1':>8}{'ΔAUC':>9}{'Probe':>9}{'Probe':>9}")
    sub = (f"{'':<74}{'test F1':>8}{'test F1':>8}{'':>8}{'':>9}{'hidden':>9}{'full':>9}")
    return "\n".join(["", "== Hidden labels: other outputs vs the full run (x100); probe = "
                      "test AUC of a probe on the shared layers ==", head, sub, *lines])


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "setup", "seeds", "output", "reference_f1", "val_f1", "test_f1",
                    "test_auc", "stage2_test_f1", "stage2_test_auc"])
        for row in rows:
            ref = (row.get("reference") or {}).get("f1", {})
            for o in row["outputs"]:
                c = row["cells"].get(o, {})
                mean = lambda k: f"{statistics.mean(c[k]):.4f}" if c.get(k) else ""
                w.writerow([row["dataset"], row["setup"], " ".join(map(str, row["seeds"])), o,
                            ref.get(o, ""), mean("val_f1"), mean("test_f1"), mean("test_auc"),
                            mean("s2_test_f1"), mean("s2_test_auc")])
