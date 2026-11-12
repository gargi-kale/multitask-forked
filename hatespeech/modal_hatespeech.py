"""Modal launcher and report for the hate speech masking experiment.

Separate from modal_replicate.py; it shares only the container image and
volumes (modal_common.py). Run from `multitask-forked/`:

    modal run hatespeech/modal_hatespeech.py::upload_data
    modal run hatespeech/modal_hatespeech.py::runs --seeds 42,1,2 --shares 0,0.25,0.5,0.75
    modal run hatespeech/modal_hatespeech.py::report

Each run writes /outputs/hatespeech_mask/mask<share>_s<seed>.json (+ _predictions.npz).
Runs whose JSON already exists are skipped.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modal  # noqa: E402

from modal_common import REPO, VOLUMES, data_vol, image, out_vol, run  # noqa: E402

app = modal.App("multitask-hatespeech")
# A comma-separated list is a fallback order: each run takes the first type with
# capacity. Only the speed differs; the code and settings are the same.
GPU = [g.strip() for g in os.environ.get("MULTITASK_GPU", "L4,A10G,L40S").split(",")]
GPU = GPU[0] if len(GPU) == 1 else GPU
OUT_DIR = "hatespeech_mask"
JOINT_DIR = "hatespeech_joint"
DATA_DIR = "/data/hatespeech"
IAC_DIR = "/data/paper"  # no_prop_overlap_*.csv, the main runs' data
IAC_TASKS = ["disagree_agree_bi", "attacking_respectful_bi", "emotion_fact_bi", "nasty_nice_bi",
             "personal_audience_bi", "defeater_undercutter_bi", "negotiate_attack_bi",
             "questioning_asserting_bi"]
# Paper Table 3 multi-task F1 for the IAC outputs (scored there on the full
# test split, which includes the rule-labelled news rows; ours: IAC rows only).
PAPER_IAC_F1 = {"disagree_agree_bi": 71.38, "emotion_fact_bi": 63.93, "attacking_respectful_bi": 68.07,
                "nasty_nice_bi": 73.69, "personal_audience_bi": 65.69, "defeater_undercutter_bi": 55.65,
                "negotiate_attack_bi": 64.81, "questioning_asserting_bi": 63.23}
ITEMS = ["sentiment", "respect", "insult", "humiliate", "status",
         "dehumanize", "violence", "genocide", "attack_defend", "hatespeech"]
TASKS = [f"{c}_bi" for c in ITEMS]


def run_name(share: float, seed: int, method: str = "plain") -> str:
    return f"{'' if method == 'plain' else method + '_'}mask{share:g}_s{seed}"


@app.function(image=image, volumes=VOLUMES, gpu=GPU, timeout=60 * 60 * 3, memory=32768)
def train(share: float, seed: int, cut04: str = ">2", cut02: str = ">=1",
          sources: str = "hatespeech", out: str = "", method: str = "plain",
          items: str = "", branches: bool = False, extreme_items: str = "violence,genocide",
          hide_task: str = "", probe_tasks: str = "", label_share: str = "") -> int:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}", flush=True)
    missing = [s for s in ("train", "val", "test")
               if not os.path.exists(f"{DATA_DIR}/hatespeech_{s}.csv")]
    if missing:
        raise FileNotFoundError(f"{DATA_DIR} lacks {missing}; run ::upload_data first")
    rc = run(["python", f"{REPO}/hatespeech/run_hatespeech.py",
              "--sources", sources, "--method", method,
              *(["--items", items] if items else []),
              "--data-dir", DATA_DIR, "--iac-dir", IAC_DIR,
              "--mask-share", str(share), "--seed", str(seed),
              "--cut04", cut04, "--cut02", cut02,
              *(["--branches", "--extreme-items", extreme_items] if branches else []),
              *(["--hide-task", hide_task] if hide_task else []),
              *(["--probe-tasks", probe_tasks] if probe_tasks else []),
              *(["--label-share", label_share] if label_share else []),
              "--out", f"/outputs/{out or f'{OUT_DIR}/{run_name(share, seed, method)}'}"])
    out_vol.commit()
    return rc


@app.local_entrypoint()
def upload_data():
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    with data_vol.batch_upload(force=True) as batch:
        for s in ("train", "val", "test"):
            batch.put_file(os.path.join(here, f"hatespeech_{s}.csv"), f"/hatespeech/hatespeech_{s}.csv")
    print("uploaded hatespeech_{train,val,test}.csv to multitask-data:/hatespeech/")


def _done(directory: str = OUT_DIR) -> set:
    try:
        return {e.path.rsplit("/", 1)[-1][:-5] for e in out_vol.listdir(directory)
                if e.path.endswith(".json")}
    except Exception:
        return set()


def joint_name(sources: str, seed: int, method: str = "plain", share: float = 0.0) -> str:
    return (f"{'' if method == 'plain' else method + '_'}{sources.replace('+', '_')}"
            f"{f'_mask{share:g}' if share else ''}_s{seed}")


@app.local_entrypoint()
def joint_runs(seeds: str = "42,1,2", sources: str = "iac,iac+hatespeech", method: str = "plain",
               share: float = 0.0):
    """IAC alone and IAC + hate speech in one model. ``share`` > 0 hides that
    share of the hate speech labels (IAC labels are never hidden), with the
    same cells as the hate speech-only run of that seed. The hate speech-only
    comparison is the runs of the same method and share in hatespeech_mask/."""
    jobs = [(src, int(s)) for src in sources.split(",") for s in seeds.split(",")]
    if share and any("hatespeech" not in src for src, _ in jobs):
        raise SystemExit("share > 0 hides hate speech labels: use --sources iac+hatespeech")
    done = _done(JOINT_DIR)
    todo = [(src, s) for src, s in jobs if joint_name(src, s, method, share) not in done]
    print(f"{method}: {len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = [(joint_name(src, s, method, share),
              train.spawn(share, s, sources=src, method=method,
                          out=f"{JOINT_DIR}/{joint_name(src, s, method, share)}"))
             for src, s in todo]
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


@app.local_entrypoint()
def joint_report(method: str = "plain"):
    """Each head trained alone-with-its-own-source vs in the IAC + hate speech
    model, all with the same method."""
    import json

    import numpy as np
    from scipy import stats

    def load(directory, keep):
        out = {}
        for e in out_vol.listdir(directory):
            if e.path.endswith(".json"):
                d = json.loads(b"".join(out_vol.read_file(e.path)))
                if keep(d):
                    out[d["seed"]] = d
        return out

    same = lambda d: d.get("method", "plain") == method and d["mask_share"] == 0
    alone_hs = load(OUT_DIR, lambda d: d["mask_share"] == 0 and same(d))
    alone_iac = load(JOINT_DIR, lambda d: d.get("sources") == "iac" and same(d))
    joint = load(JOINT_DIR, lambda d: d.get("sources") == "iac+hatespeech" and same(d))
    print(f"Method: {method}")
    seeds = sorted(joint)
    hs_tasks = [f"{c}_bi" for c in ITEMS]

    def ms(xs):
        xs = [100 * x for x in xs]
        if not xs:
            return "—"
        return f"{np.mean(xs):.1f} ± {np.std(xs, ddof=1):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"

    def paired(a, b):
        d = [100 * (a[s] - b[s]) for s in a if s in b]
        if not d:
            return "—"
        if len(d) < 2:
            return f"{d[0]:+.1f} (n=1)"
        m, se = np.mean(d), np.std(d, ddof=1) / np.sqrt(len(d))
        h = stats.t.ppf(0.975, len(d) - 1) * se
        return f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}]"

    def col(runs, t, key):
        return {s: runs[s]["test"][t][key] for s in runs if t in runs[s]["test"]}

    def group_mean(runs, names, key):
        return {s: float(np.mean([runs[s]["test"][t][key] for t in names])) for s in runs
                if all(t in runs[s]["test"] for t in names)}

    print(f"IAC + hate speech in one model vs each source on its own. Test, x100, mean ± sd over "
          f"seeds; Δ = joint − alone, paired by seed, 95% t interval.")
    print(f"Runs: hate speech alone {sorted(alone_hs)}, IAC alone {sorted(alone_iac)}, "
          f"joint {seeds}. IAC heads scored on IAC test rows only.")
    print("Paper (Table 3 F1) is the paper's multi-task model on its full test split "
          "(incl. rule-labelled news rows); the paper has no hate speech results (n/a).\n")
    epochs = lambda runs: f"{np.mean([r['best_epoch'] for r in runs.values()]):.1f}" if runs else "—"
    print(f"Best epoch (mean): hate speech alone {epochs(alone_hs)}, IAC alone {epochs(alone_iac)}, "
          f"joint {epochs(joint)}\n")
    for key in ("f1", "auc"):
        print(f"{'Test ' + key.upper():28s}{'Paper (Table 3 F1)':>20s}{'alone':>14s}{'joint':>14s}"
              f"{'Δ joint − alone':>24s}")
        for name, names, alone in (("hate speech", hs_tasks, alone_hs), ("IAC", IAC_TASKS, alone_iac)):
            for t in names:
                paper = f"{PAPER_IAC_F1[t]:.2f}" if (t in PAPER_IAC_F1 and key == "f1") else "n/a"
                a, j = col(alone, t, key), col(joint, t, key)
                print(f"  {t[:-3]:26s}{paper:>20s}{ms(a.values()):>14s}{ms(j.values()):>14s}"
                      f"{paired(j, a):>24s}")
            a, j = group_mean(alone, names, key), group_mean(joint, names, key)
            paper = (f"{np.mean(list(PAPER_IAC_F1.values())):.2f}" if name == "IAC" and key == "f1"
                     else "n/a")
            print(f"{'mean, ' + name + ' heads':28s}{paper:>20s}{ms(a.values()):>14s}"
                  f"{ms(j.values()):>14s}{paired(j, a):>24s}\n")


@app.local_entrypoint()
def runs(seeds: str = "42,1,2", shares: str = "0,0.25,0.5,0.75", method: str = "plain"):
    jobs = [(float(p), int(s)) for p in shares.split(",") for s in seeds.split(",")]
    done = _done()
    todo = [(p, s) for p, s in jobs if run_name(p, s, method) not in done]
    print(f"{method}: {len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = [(run_name(p, s, method), train.spawn(p, s, method=method)) for p, s in todo]
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


@app.local_entrypoint()
def report():
    import json

    import numpy as np
    from scipy import stats

    runs_, others = {}, {}
    for e in out_vol.listdir(OUT_DIR):
        if e.path.endswith(".json"):
            d = json.loads(b"".join(out_vol.read_file(e.path)))
            if d.get("method", "plain") == "plain":
                runs_[(d["mask_share"], d["seed"])] = d
            else:
                others.setdefault(d["method"], {})[(d["mask_share"], d["seed"])] = d
    if not runs_:
        print("no runs yet")
        return
    shares = sorted({p for p, _ in runs_})
    seeds = sorted({s for _, s in runs_})
    first = next(iter(runs_.values()))
    label = lambda p: "unmasked" if p == 0 else f"{100 * p:g}% hidden"

    def ms(xs):
        xs = [100 * x for x in xs if x is not None]
        if not xs:
            return "—"
        return f"{np.mean(xs):.1f} ± {np.std(xs, ddof=1):.1f}" if len(xs) > 1 else f"{xs[0]:.1f}"

    def paired(a, b):
        """mean(a - b) over seeds both have, with a 95% t interval."""
        d = [100 * (a[s] - b[s]) for s in a if s in b and a[s] is not None and b[s] is not None]
        if not d:
            return "—"
        if len(d) < 2:
            return f"{d[0]:+.1f} (n=1)"
        m, se = np.mean(d), np.std(d, ddof=1) / np.sqrt(len(d))
        h = stats.t.ppf(0.975, len(d) - 1) * se
        return f"{m:+.1f} [{m - h:+.1f}, {m + h:+.1f}]"

    def mean_over_items(scores, key):
        vals = [v[key] for v in scores.values() if v.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def by_seed(p, fn):
        return {s: fn(runs_[(p, s)]) for s in seeds if (p, s) in runs_}

    print(f"Hate speech masking experiment. Cut-offs: 0-4 items '{first['cut04']}', "
          f"hatespeech '{first['cut02']}'. Seeds {seeds}. Weighted F1 / AUC x100, mean ± sd over seeds;")
    print("Δ is paired by seed with a 95% t interval. The paper has no hate speech results, "
          "so there is no Paper (Table 3 F1) column here.\n")

    # 1. Overall
    print(f"{'setup':14s}{'n':>3s}{'epoch':>7s}{'val F1':>14s}{'test F1':>14s}{'test AUC':>14s}"
          f"{'ΔF1 vs unmasked':>22s}{'ΔAUC vs unmasked':>22s}")
    base_f1 = by_seed(0.0, lambda d: mean_over_items(d["test"], "f1"))
    base_auc = by_seed(0.0, lambda d: mean_over_items(d["test"], "auc"))
    for p in shares:
        f1 = by_seed(p, lambda d: mean_over_items(d["test"], "f1"))
        auc = by_seed(p, lambda d: mean_over_items(d["test"], "auc"))
        val = by_seed(p, lambda d: mean_over_items(d["val"], "f1"))
        ep = np.mean([runs_[(p, s)]["best_epoch"] for s in seeds if (p, s) in runs_])
        print(f"{label(p):14s}{len(f1):3d}{ep:7.1f}{ms(val.values()):>14s}{ms(f1.values()):>14s}"
              f"{ms(auc.values()):>14s}"
              f"{'' if p == 0 else paired(f1, base_f1):>22s}{'' if p == 0 else paired(auc, base_auc):>22s}")

    # 2. Per item, test
    print("\nTest F1 per item")
    print(f"{'item':16s}" + "".join(f"{label(p):>16s}" for p in shares))
    for t, c in zip(TASKS, ITEMS):
        print(f"{c:16s}" + "".join(
            f"{ms([runs_[(p, s)]['test'][t]['f1'] for s in seeds if (p, s) in runs_]):>16s}" for p in shares))

    # 3. The hidden cells: masked model (never saw them) vs unmasked model (trained on them)
    masked = [p for p in shares if p > 0]
    if masked:
        print("\nTrain cells the masked run hid: F1 / AUC of the masked model (never saw these labels) "
              "vs the unmasked model of the same seed (trained on them)")
        print(f"{'setup':14s}{'metric':>7s}{'masked model':>16s}{'unmasked model':>16s}"
              f"{'Δ masked − unmasked':>24s}{'visible cells (masked model)':>30s}")
        for p in masked:
            for key in ("f1", "auc"):
                hid = by_seed(p, lambda d: mean_over_items(d["train_cells"]["hidden"], key))
                # Unmasked runs made before 90/95% were added have no reference for them.
                ref = {s: mean_over_items(runs_[(0.0, s)]["train_cells"]["reference"][f"{p:g}"], key)
                       for s in seeds if (0.0, s) in runs_
                       and f"{p:g}" in runs_[(0.0, s)]["train_cells"]["reference"]}
                vis = by_seed(p, lambda d: mean_over_items(d["train_cells"]["visible"], key))
                print(f"{label(p):14s}{key.upper():>7s}{ms(hid.values()):>16s}{ms(ref.values()):>16s}"
                      f"{paired(hid, ref):>24s}{ms(vis.values()):>30s}")

        print("\nHidden-cell F1 per item (masked model)")
        print(f"{'item':16s}" + "".join(f"{label(p):>16s}" for p in masked))
        for t, c in zip(TASKS, ITEMS):
            print(f"{c:16s}" + "".join(
                f"{ms([runs_[(p, s)]['train_cells']['hidden'][t]['f1'] for s in seeds if (p, s) in runs_]):>16s}"
                for p in masked))

    _other_methods_section(runs_, others, ms, paired, mean_over_items)


def _other_methods_section(runs_, others, ms, paired, mean_over_items):
    """Each non-plain method vs plain at the same hidden share, paired by seed."""
    for method, mruns in sorted(others.items()):
        print(f"\n{method} vs plain at the same share of hidden labels (test, and hidden train cells)")
        print(f"{'labels hidden':14s}{'n':>3s}{'epoch':>7s}{'test F1':>14s}{'ΔF1 vs plain':>22s}"
              f"{'test AUC':>14s}{'ΔAUC vs plain':>22s}{'hidden-cell F1':>16s}{'Δ vs plain':>22s}")
        for p in sorted({p for p, _ in mruns}):
            seeds = sorted(s for q, s in mruns if q == p)
            get = lambda rr, split, key: {s: mean_over_items(rr[(p, s)][split], key)
                                          for s in seeds if (p, s) in rr}
            f1, auc = get(mruns, "test", "f1"), get(mruns, "test", "auc")
            pf1, pauc = get(runs_, "test", "f1"), get(runs_, "test", "auc")
            if p:
                hid = {s: mean_over_items(mruns[(p, s)]["train_cells"]["hidden"], "f1") for s in seeds}
                phid = {s: mean_over_items(runs_[(p, s)]["train_cells"]["hidden"], "f1")
                        for s in seeds if (p, s) in runs_}
                hid_txt, dhid = ms(hid.values()), paired(hid, phid)
            else:
                hid_txt = dhid = "—"
            ep = sum(mruns[(p, s)]["best_epoch"] for s in seeds) / len(seeds)
            label = "none" if p == 0 else f"{100 * p:g}%"
            print(f"{label:14s}{len(seeds):3d}{ep:7.1f}{ms(f1.values()):>14s}{paired(f1, pf1):>22s}"
                  f"{ms(auc.values()):>14s}{paired(auc, pauc):>22s}{hid_txt:>16s}{dhid:>22s}")


# --------------------------------------------------------------------------
# Single-task baseline: one model per hate speech item, same settings as the
# multi-task runs (plain, all labels). Runs go to hatespeech_single/.
#
#     modal run --detach hatespeech/modal_hatespeech.py::single_runs --seeds 42
#     modal run hatespeech/modal_hatespeech.py::single_report
# --------------------------------------------------------------------------
SINGLE_DIR = "hatespeech_single"


@app.local_entrypoint()
def single_runs(seeds: str = "42", items: str = ""):
    names = items.split(",") if items else list(ITEMS)
    jobs = [(c, int(s)) for c in names for s in seeds.split(",")]
    done = _done(SINGLE_DIR)
    todo = [(c, s) for c, s in jobs if f"only_{c}_s{s}" not in done]
    print(f"single-task: {len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    calls = [(f"only_{c}_s{s}", train.spawn(0.0, s, items=c, out=f"{SINGLE_DIR}/only_{c}_s{s}"))
             for c, s in todo]
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM, or the local connection dropping
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)


@app.local_entrypoint()
def single_report():
    """Single-task model per item vs the multi-task hate speech models (all
    labels). Single-task seeds are matched to the multi-task seeds that exist."""
    import json

    import numpy as np

    single, multi = {}, {}
    for e in out_vol.listdir(SINGLE_DIR):
        if e.path.endswith(".json"):
            d = json.loads(b"".join(out_vol.read_file(e.path)))
            single[(d["items"][0], d["seed"])] = d
    for e in out_vol.listdir(OUT_DIR):
        if e.path.endswith(".json"):
            d = json.loads(b"".join(out_vol.read_file(e.path)))
            if d["mask_share"] == 0:
                multi.setdefault(d.get("method", "plain"), {})[d["seed"]] = d
    seeds = sorted({s for _, s in single})
    methods = [m for m in ("plain", "famo", "famo_consf1_m5") if m in multi]
    f = lambda x: f"{100 * x:.1f}"

    def multi_cells(m, t):
        same = [multi[m][s]["test"][t]["f1"] for s in seeds if s in multi[m]]
        allv = [multi[m][s]["test"][t]["f1"] for s in multi[m]]
        return (f(np.mean(same)) if same else "—",
                f"{100 * np.mean(allv):.1f} ± {100 * np.std(allv, ddof=1):.1f}" if len(allv) > 1 else "—")

    print(f"Hate speech: single-task (one model per item) vs multi-task (one model, 10 items), "
          f"all labels, test weighted F1 x100.")
    print(f"Single-task seeds: {seeds}. Multi-task shown on the same seed(s), and as mean ± sd "
          f"over its 3 seeds. The paper has no hate speech results, so no Paper column.\n")
    head = f"{'item':16s}{'single':>9s}{'epoch':>7s}"
    for m in methods:
        head += f"{m + ' (same seed)':>26s}{'Δ multi − single':>18s}{m + ' (3 seeds)':>24s}"
    print(head)
    rows = {m: [] for m in methods}
    singles = []
    for c in ITEMS:
        t = f"{c}_bi"
        runs = [single[(c, s)] for s in seeds if (c, s) in single]
        if not runs:
            continue
        sv = np.mean([r["test"][t]["f1"] for r in runs])
        singles.append(sv)
        line = f"{c:16s}{f(sv):>9s}{np.mean([r['best_epoch'] for r in runs]):7.1f}"
        for m in methods:
            same, allv = multi_cells(m, t)
            mv = np.mean([multi[m][s]["test"][t]["f1"] for s in seeds if s in multi[m]])
            rows[m].append(mv)
            line += f"{same:>26s}{100 * (mv - sv):>+18.1f}{allv:>24s}"
        print(line)
    line = f"{'mean':16s}{f(np.mean(singles)):>9s}{'':7s}"
    for m in methods:
        mv = np.mean(rows[m])
        allv = [np.mean([multi[m][s]["test"][f'{c}_bi']['f1'] for c in ITEMS]) for s in multi[m]]
        line += (f"{f(mv):>26s}{100 * (mv - np.mean(singles)):>+18.1f}"
                 f"{100 * np.mean(allv):>18.1f} ± {100 * np.std(allv, ddof=1):.1f}")
    print(line)
    auc = np.mean([single[(c, s)]["test"][f"{c}_bi"]["auc"] for c in ITEMS for s in seeds if (c, s) in single])
    print(f"\nMean test AUC: single-task {100 * auc:.1f}; multi-task plain "
          f"{100 * np.mean([np.mean([multi['plain'][s]['test'][f'{c}_bi']['auc'] for c in ITEMS]) for s in seeds if s in multi['plain']]):.1f} (same seed)")


# --------------------------------------------------------------------------
# Joint IAC + hate speech with hate speech branches, and hide-one-label runs
# --------------------------------------------------------------------------
# Three branches: hate speech "extreme" items (default violence, genocide),
# the other hate speech items, and IAC. All hate speech heads keep one task
# type, so loss weights match the unbranched joint runs.
#
# `branched_runs`: full joint runs with probes on all 18 labels (the reference).
# `hide_one_runs`: one run per label with it hidden on every train row; a
#   probe reads it from the frozen model, and the other 17 labels give
#   leave-one-task-out transfer against the full run of the same method/seed.
#
#     MULTITASK_GPU=L4 modal run --detach hatespeech/modal_hatespeech.py::hide_one_runs --tasks violence_bi --seeds 42
#
# Writes /outputs/hatespeech_hide_one/<method>_br_{full,hide_<label>}_s<seed>.json.

HIDE_ONE_DIR = "hatespeech_hide_one"
ALL_JOINT = [*TASKS, *IAC_TASKS]


def _br_name(method: str, what: str, seed: int) -> str:
    return f"{method}_br_{what}_s{seed}"


@app.local_entrypoint()
def branched_runs(seeds: str = "42,1,2", methods: str = "plain,famo_consf1_m5",
                  extreme_items: str = "violence,genocide"):
    jobs = [(m, int(s)) for m in methods.split(",") for s in seeds.split(",")]
    done = _done(HIDE_ONE_DIR)
    todo = [(m, s) for m, s in jobs if _br_name(m, "full", s) not in done]
    print(f"{len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    _wait([(_br_name(m, "full", s), train.spawn(
        0.0, s, sources="iac+hatespeech", method=m, branches=True, extreme_items=extreme_items,
        probe_tasks="all", out=f"{HIDE_ONE_DIR}/{_br_name(m, 'full', s)}")) for m, s in todo])


@app.local_entrypoint()
def hide_one_runs(tasks: str = "", seeds: str = "42,1,2", method: str = "plain",
                  extreme_items: str = "violence,genocide"):
    labels = tasks.split(",") if tasks else ALL_JOINT
    unknown = [t for t in labels if t not in ALL_JOINT]
    if unknown:
        raise SystemExit(f"unknown labels {unknown}; choose from {ALL_JOINT}")
    jobs = [(t, int(s)) for t in labels for s in seeds.split(",")]
    done = _done(HIDE_ONE_DIR)
    todo = [(t, s) for t, s in jobs if _br_name(method, f"hide_{t}", s) not in done]
    print(f"{method}: {len(jobs)} runs requested, {len(jobs) - len(todo)} already done, "
          f"launching {len(todo)}.", flush=True)
    _wait([(_br_name(method, f"hide_{t}", s), train.spawn(
        0.0, s, sources="iac+hatespeech", method=method, branches=True,
        extreme_items=extreme_items, hide_task=t, probe_tasks="hidden",
        out=f"{HIDE_ONE_DIR}/{_br_name(method, f'hide_{t}', s)}")) for t, s in todo])


# Label-budget runs on the joint branched model: one label keeps 5 / 10 / 25%
# of its train labels (famo_consf1_m5). The 100% run is
# hatespeech_hide_one/famo_consf1_m5_br_full_s<seed>. For an IAC label the
# single-task side is the main-data single-task run of the same share and seed
# (paper_replication_label_budget/single_<label>_p<pct>_s<seed>): same train
# rows, same IAC test rows.
LABEL_BUDGET_DIR = "hatespeech_label_budget"


@app.local_entrypoint()
def label_budget_runs(task: str = "attacking_respectful_bi", shares: str = "0.05,0.1,0.25",
                      seeds: str = "42", method: str = "famo_consf1_m5",
                      extreme_items: str = "violence,genocide"):
    if task not in ALL_JOINT:
        raise SystemExit(f"unknown label {task}; choose from {ALL_JOINT}")
    done = _done(LABEL_BUDGET_DIR)
    jobs = [(_br_name(method, f"{task}_p{round(100 * float(p))}", int(s)), float(p), int(s))
            for s in seeds.split(",") for p in shares.split(",")]
    todo = [j for j in jobs if j[0] not in done]
    print(f"launching {len(todo)}: {[n for n, _, _ in todo]}", flush=True)
    _wait([(n, train.spawn(0.0, s, sources="iac+hatespeech", method=method, branches=True,
                           extreme_items=extreme_items, label_share=f"{task}={p}",
                           out=f"{LABEL_BUDGET_DIR}/{n}")) for n, p, s in todo])


def _wait(calls: list) -> None:
    for name, call in calls:
        try:
            rc = call.get()
        except Exception as exc:  # timeout, preemption, OOM
            rc = repr(exc)
        print(f"{'ok  ' if rc == 0 else 'FAIL'} {name}  ({rc})", flush=True)
