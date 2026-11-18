"""The ``multitask`` command: detect, prepare, train, report, embed.

    multitask detect  --data data/my_data
    multitask prepare --input raw.csv --out data/my_data --style cmp
    multitask train   --data data/my_data --method famo_consf1_m5 --seeds 42,1,2
    multitask report  runs/my_data

``--data`` takes a folder (columns are detected), a dataset.yaml, or the name
of a bundled dataset (iac, hatespeech), optionally with the folder holding
its files: ``--data iac=/path/to/files``. Give it more than once to train
the datasets together in one joint model.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from typing import List, Optional, Sequence

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLED_DIR = os.path.join(REPO_ROOT, "datasets")


# ---------------------------------------------------------------------------
# --data
# ---------------------------------------------------------------------------

def bundled_datasets() -> List[str]:
    if not os.path.isdir(BUNDLED_DIR):
        return []
    return sorted(d for d in os.listdir(BUNDLED_DIR)
                  if os.path.exists(os.path.join(BUNDLED_DIR, d, "dataset.yaml")))


def _comma(values: Optional[Sequence[str]]) -> List[str]:
    return [x.strip() for v in (values or []) for x in v.split(",") if x.strip()]


def load_data(args: argparse.Namespace, quiet: bool = False):
    """[DatasetSpec] for every --data, with --tasks / --ignore / --pool applied."""
    from multitask.train.dataset import describe, detect, load_spec, split_prefix

    specs, reports = [], []
    for item in args.data:
        target, _, folder = item.partition("=")
        if target.endswith((".yaml", ".yml")):
            spec, report = load_spec(target), None
        elif target in bundled_datasets() and (folder or not os.path.isdir(target)):
            # NAME=FOLDER always means the bundled dataset; a bare NAME does too
            # unless a folder of that name is in the working directory.
            spec, report = load_spec(os.path.join(BUNDLED_DIR, target, "dataset.yaml")), None
        elif os.path.isdir(target) or os.path.isfile(target):
            where, prefix = (target, "") if os.path.isdir(target) else split_prefix(target)
            spec, report = detect(where, prefix=prefix, text_column=args.text_column,
                                  source_column=args.source_column,
                                  binarize=args.binarize or (), group_by=args.groups)
        else:
            raise SystemExit(f"--data {target!r}: not a folder, a data file, a .yaml file or a "
                             "bundled dataset "
                             f"({', '.join(bundled_datasets()) or 'none found'})")
        if folder:
            spec.data_dir = os.path.abspath(folder)
        elif report is None and not all(os.path.exists(spec.path(s)) for s in spec.files):
            raise SystemExit(f"--data {target}: data files not found in {spec.data_dir}; "
                             f"give the folder that holds {sorted(spec.files.values())} as "
                             f"--data {target}=FOLDER")
        if report is None and (args.binarize or args.text_column or args.source_column):
            raise SystemExit("--binarize / --text-column / --source-column apply to detected "
                             "folders; edit the dataset.yaml instead")
        specs.append(spec)
        reports.append(report)

    tasks, ignore = _comma(args.tasks), _comma(args.ignore)
    if tasks or ignore:
        for refs, what in ((tasks, "--tasks"), (ignore, "--ignore")):
            for ref in refs:
                if not any(_matches(s, ref) for s in specs):
                    raise SystemExit(f"{what} {ref!r} matches no task")
        out, out_reports = [], []
        for spec, report in zip(specs, reports):
            t = [r for r in tasks if _matches(spec, r)]
            if tasks and not t:
                continue  # none of this dataset's tasks were asked for
            out.append(spec.select(t, [r for r in ignore if _matches(spec, r)]))
            if report is not None:
                kept = {k.column for k in out[-1].tasks}
                for row in report:
                    if row.role == "task" and row.column not in kept:
                        row.role, row.detail = "left out", "by --tasks / --ignore"
            out_reports.append(report)
        specs, reports = out, out_reports
    for pool in args.pool or []:
        name, _, rest = pool.partition("=")
        refs, _, strategy = rest.partition(":")
        if not name or not refs:
            raise SystemExit(f"--pool {pool!r}: expected NAME=TASKS[:strategy]")
        hits = [i for i, s in enumerate(specs) if all(_matches(s, r) for r in refs.split(","))]
        if len(hits) != 1:
            raise SystemExit(f"--pool {pool!r}: its tasks must all belong to one dataset")
        specs[hits[0]] = specs[hits[0]].with_pool(name, refs.split(","), strategy or "max")
    if not quiet:
        for spec, report in zip(specs, reports):
            print(describe(spec, report))
            if report is not None:
                print("  (detected; to change groups or merged outputs, edit the dataset.yaml "
                      "a run saves and pass it with --data)")
    return specs


def _matches(spec, ref: str) -> bool:
    try:
        spec.expand([ref])
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_detect(args: argparse.Namespace) -> None:
    from multitask.train.dataset import save_spec

    specs = load_data(args)
    if args.save:
        if len(specs) != 1:
            raise SystemExit("--save writes one dataset; give one --data")
        save_spec(specs[0], args.save, header="Written by `multitask detect`. Edit freely: "
                  "task groups, merged outputs (pools),\nscored outputs, display labels.")
        print(f"Wrote {args.save}")


def cmd_prepare(args: argparse.Namespace) -> None:
    from multitask.prepare import prepare

    split = tuple(float(x) for x in args.split.split(","))
    print(prepare(args.input, args.out, name=args.name, text_column=args.text_column,
                  style=args.style, steps=_comma(args.clean), remove_tokens=_comma(args.remove_token),
                  min_words=args.min_words, keep_duplicates=args.keep_duplicates,
                  dedupe_keep_max=args.dedupe_keep_max, split=split, seed=args.seed,
                  stratify=_comma(args.stratify)))


def _settings(args: argparse.Namespace):
    from multitask.train.methods import parse_set, settings_for

    over = dict(
        encoder=args.encoder, max_length=args.max_length, batch_size=args.batch_size,
        epochs=args.epochs, learning_rate=args.lr, patience=args.patience,
        stop_metric=args.stop_metric, arch=args.arch, stage2=args.stage2,
        length_chunk_batches=args.fast_batches, embedding_cache=args.embedding_cache or "",
        hide_source=args.hide_source or "", keep_sources=_comma(args.keep_sources),
        unweighted_loss=args.unweighted_loss, no_grouping=args.no_grouping,
        probes=args.probes or "", probe_untrained=args.probe_untrained,
        repeat_train=args.repeat_train, verbose=args.verbose,
        hide_label="" if args.hide_label in (None, "all") else args.hide_label,
    )
    over = {k: v for k, v in over.items() if v is not None}
    try:
        over.update(parse_set(args.set or []))
        return settings_for(args.method, **over)
    except ValueError as exc:
        raise SystemExit(str(exc))


def cmd_train(args: argparse.Namespace) -> None:
    from multitask.train.runner import run_all

    specs = load_data(args)
    base = _settings(args)
    seeds = [int(x) for x in _comma([args.seeds])]
    variants = [base]
    if args.hide_share:
        variants = [replace(v, hide_share=float(x)) for v in variants
                    for x in _comma([args.hide_share])]
    if args.label_budget:
        task, _, shares = args.label_budget.partition("=")
        if not shares:
            raise SystemExit("--label-budget expects TASK=SHARE[,SHARE...]")
        variants = [replace(v, label_budget={task: float(x)}) for v in variants
                    for x in shares.split(",")]
    hide_each = args.hide_label == "all"
    name = args.out or os.path.join("runs", "+".join(s.name for s in specs))
    try:
        for v in variants:
            run_all(specs, v, name, seeds=seeds, hide_each=hide_each, overwrite=args.overwrite,
                    dry_run=args.dry_run)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}")
    if not args.dry_run:
        cmd_report(argparse.Namespace(runs=name, csv=None, brief=False))


def cmd_report(args: argparse.Namespace) -> None:
    from multitask.train.report import (
        hide_label_table, load_results, render, summarize, write_csv,
    )

    results = load_results(args.runs)
    if not results:
        raise SystemExit(f"no results under {args.runs}")
    rows = summarize(results)
    print(render(rows, detail=not args.brief))
    hidden = hide_label_table(results)
    if hidden:
        print(hidden)
    if args.csv:
        write_csv(rows, args.csv)
        print(f"Wrote {args.csv}")


def cmd_embed(args: argparse.Namespace) -> None:
    from multitask.train import openai_embeddings as oe
    from multitask.train.encoders import OPENAI_EMBEDDING_MODELS
    from multitask.train.runner import prepare_data
    from multitask.train.methods import RunSettings

    if args.encoder not in OPENAI_EMBEDDING_MODELS:
        raise SystemExit(f"--encoder must be one of {', '.join(OPENAI_EMBEDDING_MODELS)}")
    specs = load_data(args, quiet=True)
    out = args.out or os.path.join("runs", "+".join(s.name for s in specs))
    prepared = prepare_data(specs, RunSettings(), out)
    model = OPENAI_EMBEDDING_MODELS[args.encoder]
    texts = [t for split in oe.SPLITS for t in oe.read_texts(prepared.data_dir, split, prefix="")]
    info = oe.plan(texts, model, args.embedding_cache)
    info.pop("missing")
    print(f"{model}: {len(texts):,} rows, {info['unique_texts']:,} unique texts, "
          f"{info['already_stored']:,} already stored in {args.embedding_cache}.")
    print(f"To embed: {info['to_embed']:,} texts, {info['tokens']:,} tokens, about "
          f"${info['est_cost_usd']:.2f} at list price.")
    if not info["to_embed"]:
        return
    if not args.yes:
        print("Nothing sent. Run again with --yes to call the OpenAI API (needs OPENAI_API_KEY).")
        return
    done = oe.embed_missing(texts, model, args.embedding_cache, workers=args.workers)
    print(f"Embedded {done.get('embedding_now', 0):,} texts.")


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _data_options(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data")
    g.add_argument("--data", action="append", required=True, metavar="FOLDER|FILE|YAML|NAME[=FOLDER]",
                   help="dataset: a folder or one of its train/val/test files (detected), a "
                        "dataset.yaml, or a bundled name "
                        f"({', '.join(bundled_datasets()) or 'none'}); repeat for a joint model")
    g.add_argument("--tasks", action="append", metavar="LIST",
                   help="train only these tasks/outputs (comma list; globs, group:NAME, pool:NAME)")
    g.add_argument("--ignore", action="append", metavar="LIST", help="leave these tasks out")
    g.add_argument("--binarize", action="append", metavar="COLS OP VALUE",
                   help="turn a score into a yes/no label, e.g. 'violence>=2' or "
                        "'sentiment,respect>2' or 'label==toxic' (detected folders)")
    g.add_argument("--text-column", help="the text column (detected folders; default: guessed)")
    g.add_argument("--source-column", help="column naming each row's source (default: a column "
                   "named source/dataset/corpus/domain, if any)")
    g.add_argument("--groups", default="auto", choices=("auto", "source", "single"),
                   help="task groups for detected folders: by the source that labels each task, "
                        "or one group (default: by source when there is one)")
    g.add_argument("--pool", action="append", metavar="NAME=TASKS[:max|noisy_or|lse]",
                   help="add a merged output scoring 'any of these tasks'")


def build_parser() -> argparse.ArgumentParser:
    from multitask.train.methods import METHOD_HELP, PAPER_ARCHITECTURES, RunSettings

    d = RunSettings()
    parser = argparse.ArgumentParser(prog="multitask", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect", help="show what a dataset's columns will be used for")
    _data_options(p)
    p.add_argument("--save", metavar="YAML", help="write the description to edit and reuse")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("prepare", help="clean text and make train/val/test files",
                       description=__import__("multitask.prepare", fromlist=["x"]).__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="a file to split, or a folder of split files")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--name", help="file name prefix (default: the input's name)")
    p.add_argument("--text-column")
    p.add_argument("--style", default="light", choices=("none", "light", "cmp"),
                   help="cmp = ascii, urls, handles and punct (the cleaned CMP / hate speech style)")
    p.add_argument("--clean", action="append", metavar="STEPS",
                   help="extra steps: ascii, urls, handles, punct, lower (comma list)")
    p.add_argument("--remove-token", action="append", metavar="WORDS",
                   help="placeholder words to delete, e.g. URL")
    p.add_argument("--min-words", type=int, default=1)
    p.add_argument("--keep-duplicates", action="store_true")
    p.add_argument("--dedupe-keep-max", metavar="COLUMN",
                   help="of duplicate texts keep the row with the highest COLUMN (e.g. n_annotators)")
    p.add_argument("--split", default="0.8,0.1,0.1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify", action="append", metavar="COLS",
                   help="keep these columns' shares equal across splits (e.g. platform,label>=1)")
    p.set_defaults(func=cmd_prepare)

    methods = "\n".join(f"  {k:<16} {v}" for k, v in METHOD_HELP.items())
    p = sub.add_parser(
        "train", help="train and score models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"Methods:\n{methods}\n\nResults go to --out/<run>/result.json; "
                    "`multitask report` summarises them.")
    _data_options(p)
    p.add_argument("--method", default=d.method, choices=list(METHOD_HELP))
    p.add_argument("--seeds", default=str(d.seed), help="comma list, e.g. 42,1,2")
    p.add_argument("--out", help="results folder (default: runs/<dataset>)")
    p.add_argument("--dry-run", action="store_true", help="show the plan, train nothing")
    p.add_argument("--overwrite", action="store_true", help="re-train runs that have results")
    g = p.add_argument_group("model and training (defaults: the re-tuned setup)")
    g.add_argument("--encoder", help=f"default {d.encoder}; also electra-small, electra-small-L4, "
                   "distilbert, openai-small (needs `multitask embed`)")
    g.add_argument("--max-length", type=int, help=f"word pieces per text (default {d.max_length})")
    g.add_argument("--batch-size", type=int, help=f"default {d.batch_size}")
    g.add_argument("--epochs", type=int, help=f"most epochs (default {d.epochs})")
    g.add_argument("--lr", type=float, help=f"learning rate (default {d.learning_rate:g})")
    g.add_argument("--patience", type=int,
                   help=f"stop after this many epochs without improvement (default {d.patience})")
    g.add_argument("--stop-metric", choices=("f1", "auc", "loss"),
                   help=f"validation score that picks the epoch (default {d.stop_metric})")
    g.add_argument("--arch", help=f"layer sizes: {', '.join(PAPER_ARCHITECTURES)} or "
                   f"SHARED/GROUP/TASK widths like 256,128/128,64/64,32 (default {d.arch})")
    g.add_argument("--stage2", action="store_true",
                   help="frozen stage 2: re-train each output's own layers on the frozen model")
    g.add_argument("--fast-batches", type=int, metavar="N",
                   help="group texts of similar length (chunks of N batches; ~1.7x faster)")
    g.add_argument("--embedding-cache", help="folder of stored OpenAI embeddings")
    g.add_argument("--verbose", type=int, choices=(0, 1, 2))
    g.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="any other setting, e.g. --set consistency_keep=0.2 (see methods.py)")
    g = p.add_argument_group("ablations")
    g.add_argument("--hide-label", metavar="OUTPUT|all",
                   help="hide one output's labels from every train row (all = a reference run "
                        "plus one run per output); probes then read it from the trained model")
    g.add_argument("--hide-share", metavar="SHARES",
                   help="hide this share of every label's train cells at random, e.g. 0.5 or "
                        "0.25,0.5,0.9 (scores the hidden cells too)")
    g.add_argument("--label-budget", metavar="TASK=SHARES",
                   help="keep only this share of one task's train labels, e.g. A=0.05,0.1,0.25")
    g.add_argument("--hide-source", metavar="SOURCE[:TASKS]",
                   help="remove these tasks' labels (default all) on SOURCE's rows in every split")
    g.add_argument("--keep-sources", action="append", metavar="LIST",
                   help="use only rows of these sources")
    g.add_argument("--unweighted-loss", action="store_true",
                   help="no task or class weights: plain mean loss over labelled cells")
    g.add_argument("--no-grouping", action="store_true",
                   help="weight each output on its own instead of per task group")
    g.add_argument("--probes", metavar="hidden|all|LIST",
                   help="after training, fit linear probes for these outputs")
    g.add_argument("--probe-untrained", action="store_true",
                   help="control: the probes on the model before any training")
    g.add_argument("--repeat-train", type=int, metavar="N",
                   help="repeat train rows N times (same steps per epoch with fewer labels)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("report", help="summarise finished runs")
    p.add_argument("runs", help="results folder")
    p.add_argument("--csv", help="also write per-output scores to this CSV")
    p.add_argument("--brief", action="store_true", help="one line per setup")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("embed", help="store OpenAI embeddings for --encoder openai-*")
    _data_options(p)
    p.add_argument("--encoder", default="openai-small")
    p.add_argument("--embedding-cache", required=True)
    p.add_argument("--out", help="results folder (where the prepared data is kept)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--yes", action="store_true", help="really call the API (costs money)")
    p.set_defaults(func=cmd_embed)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main(sys.argv[1:])
