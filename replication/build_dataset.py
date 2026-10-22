"""
Rebuild the `no_prop_overlap_*.csv` dataset from the three public source corpora.

Target layout is testing.md section 4.3 on the `testing` branch, which is what
`replication/test_no_prop_overlap.py` reads.

    python replication/build_dataset.py \
        --iac       path/to/iac_v2 \
        --ibm-rank  path/to/IBM-ArgQ-Rank-30kArgs \
        --ptc       path/to/ptc-corpus \
        --out       data \
        --augment \
        --seed 42

Then validate against the paper's Table 1 before training:

    python replication/build_dataset.py --validate-only --out data

WHAT IS GROUNDED IN THE PAPER (arXiv:2307.01401, ACL 2024.argmining-1.5)
------------------------------------------------------------------------
- Three sources: Internet Argument Corpus (IAC), IBM-Rank-30k, and the
  Da San Martino et al. (2019) propaganda corpus.
- Dichotomization: "we dichotomize the data by simply cutting on the scale
  midpoint" -- IAC scales are [-5, 5] so the cut is at 0; IBM-Rank-30k quality
  is [0, 1] so the cut is at 0.5. The paper explicitly does NOT drop
  observations near the midpoint: doing so "would remove too much information".
- Split: "We use 80% for training and set aside 10% each for validation and
  test sets." No stratification is mentioned.
- Augmentation, applied to TRAINING data only, four kinds:
    1. back-translation through German
    2. contextual word embedding - 30% of tokens replaced by BERT predictions
    3. synonym substitution - 30% of tokens replaced via WordNet
    4. random cropping - 30% of tokens deleted
  Table 1's counts require this: Argument Quality shows 96,036 training rows
  from a 30,497-argument corpus, i.e. roughly 24.4k x 4.
- The model has TEN outputs. The 18 propaganda techniques are "collapsed to a
  single binary propaganda classification" by max pooling, and that pooled
  output is what the results tables report.

WHAT IS NOT IN THE PAPER (assumptions, flagged at the point of use)
-------------------------------------------------------------------
- The meaning of "no_prop_overlap". The paper says only that they "extract all
  sentences"; it never describes span-overlap filtering. The default here is
  --overlap-policy=first (a sentence carrying several technique spans keeps
  them all; nothing is dropped). --overlap-policy=drop-multi reproduces a
  stricter reading. Neither is verified.
- Whether augmented rows carry labels for every task or only the task they
  were drawn from. This script keeps the source row's label pattern.
- The random seed and therefore the exact split membership.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column layout (testing.md section 4.3)
# --------------------------------------------------------------------------

# Exact spellings matter: two names contain commas, and the trainer maps these
# to sanitized Keras layer names via PROPAGANDA_BINARY_LABEL_MAP.
PROPAGANDA_TECHNIQUES = [
    "Loaded_Language", "Slogans", "Black-and-White_Fallacy", "Flag-Waving",
    "Name_Calling", "Whataboutism", "Causal_Oversimplification",
    "Exaggeration,Minimisation", "Doubt", "Appeal_to_Authority", "Repetition",
    "Appeal_to_fear-prejudice", "Red_Herring", "Thought-terminating_Cliches",
    "Reductio_ad_hitlerum", "Bandwagon",
    "Obfuscation,Intentional_Vagueness,Confusion", "Straw_Men",
]

DISCOURSE_DIMENSIONS = [
    "disagree_agree", "attacking_respectful", "emotion_fact", "nasty_nice",
    "personal_audience", "defeater_undercutter", "negotiate_attack",
    "questioning_asserting",
]

OUTPUT_COLUMNS = (
    ["text"]
    + PROPAGANDA_TECHNIQUES
    + DISCOURSE_DIMENSIONS
    + [f"{d}_bi" for d in DISCOURSE_DIMENSIONS]
    + ["propaganda", "WA", "WA_bi", "dataset"]
)

# Paper Table 1: training-row count and class balance (% class0/class1) per
# reported task. Used by --validate-only as a rebuild check.
PAPER_TABLE_1 = {
    "propaganda":              (61_909, "63/37"),
    "disagree_agree_bi":       (66_684, "21/79"),
    "emotion_fact_bi":         (76_403, "41/59"),
    "attacking_respectful_bi": (65_998, "66/34"),
    "nasty_nice_bi":           (65_829, "73/27"),
    "personal_audience_bi":    (24_749, "25/75"),
    "defeater_undercutter_bi": (24_357, "38/62"),
    "negotiate_attack_bi":     (26_604, "44/56"),
    "questioning_asserting_bi":(29_791, "66/34"),
    "WA_bi":                   (96_036, "6/94"),
}

NA = "NA"  # what the trainer converts to mask_value=-1.0


# --------------------------------------------------------------------------
# Source loaders
# --------------------------------------------------------------------------
# Each loader returns a DataFrame with `text`, `dataset`, and only the columns
# its corpus actually annotates. Everything else is filled with NA at merge.
#
# These raise with an explicit message rather than guessing, because the raw
# archives are license-gated and their on-disk schema could not be inspected
# while writing this. Point --iac/--ibm-rank/--ptc at the extracted archives
# and adjust the column names flagged ADAPT if your copy differs.


# IAC's native annotation column -> the paper's task name. Verified against
# v1/data/MechanicalTurk/mTurkQuestions.csv, which gives each question's scale.
# Task 1 questions (agreement, fact-feeling, attack, nicenasty) are annotated
# on ~9,980 pairs; task 2 (the other four) on ~8,977.
IAC_COLUMN_MAP = {
    "agreement": "disagree_agree",
    "fact-feeling": "emotion_fact",
    "attack": "attacking_respectful",
    "nicenasty": "nasty_nice",
    "personal-audience": "personal_audience",
    "defeater-undercutter": "defeater_undercutter",
    "negotiate-attack": "negotiate_attack",
    "questioning-asserting": "questioning_asserting",
}

# emotion_fact is the one dimension whose sign runs the other way. IAC names the
# column `fact-feeling` while mTurkQuestions.csv labels its scale "feelings-fact",
# and empirically the unflipped cut lands at 46/54 against the paper's 41/59
# while the flipped one lands at 54/46. Calibrated to Table 1, not documented.
IAC_NO_FLIP = {"emotion_fact"}


def load_iac_v1(root: Path, flip_polarity: bool = True) -> pd.DataFrame:
    """IAC v1 (4forums Mechanical Turk quote-response annotations).

    Layout, as published at github.com/sl-m-lab/Internet-Argument-Corpus:
        v1/data/fourforum/annotations/mechanical_turk/qr_averages.csv
            key + the 8 dimensions, already averaged across annotators, [-5, 5]
        v1/data/fourforum/annotations/mechanical_turk/qr_meta.csv
            same key + `quote` and `response` text

    10,001 annotated pairs. The annotations describe the RESPONDENT, so
    `response` is the text; `quote` is the prior post being replied to.

    flip_polarity: IAC's sign convention is the opposite of the paper's binary
    encoding. Dichotomizing at 0 with `score < 0 -> 1` reproduces the paper's
    Table 1 class balances on 7 of 8 tasks to within a few points (e.g.
    disagree_agree 20/80 here vs 21/79 reported). With `score > 0 -> 1` every
    one of those 7 comes out mirrored. emotion_fact does not match either way
    (54/46 here vs 41/59 reported) and is unexplained. This flag is an
    empirical calibration to Table 1, not a documented transformation.
    """
    averages = next(root.rglob("qr_averages.csv"), None)
    meta = next(root.rglob("qr_meta.csv"), None)
    if averages is None or meta is None:
        raise FileNotFoundError(
            f"Expected qr_averages.csv and qr_meta.csv under {root}. Clone "
            "https://github.com/sl-m-lab/Internet-Argument-Corpus (the v1/ tree; "
            "its v2/ directory is an empty placeholder)."
        )

    av = pd.read_csv(averages)
    me = pd.read_csv(meta)
    missing = [c for c in IAC_COLUMN_MAP if c not in av.columns]
    if missing:
        raise ValueError(f"{averages} is missing expected columns: {missing}")

    df = av.merge(me[["key", "response", "quote"]], on="key", how="inner")
    out = pd.DataFrame({"text": df["response"].astype(str), "dataset": "iac_v1_4forums"})

    for iac_col, task in IAC_COLUMN_MAP.items():
        score = pd.to_numeric(df[iac_col], errors="coerce")
        out[task] = score
        flip = flip_polarity and task not in IAC_NO_FLIP
        binary = (score < 0) if flip else (score > 0)
        # Keep NA where the dimension was never annotated for this pair, so
        # the trainer masks it instead of learning from a fabricated 0.
        out[f"{task}_bi"] = np.where(score.isna(), np.nan, binary.astype(float))

    before = len(out)
    out = out[out["text"].str.strip().astype(bool)]
    if len(out) < before:
        print(f"  dropped {before - len(out):,} rows with empty response text")
    return out


def load_iac(root: Path, flip_polarity: bool = True) -> pd.DataFrame:
    """Internet Argument Corpus -> the 8 discourse dimensions.

    Dispatches to load_iac_v1() when the v1 Mechanical Turk tables are present.
    Otherwise falls back to a generic scan for a text column plus dimension
    columns, for a v2 copy whose schema could not be inspected here.
    """
    if next(root.rglob("qr_averages.csv"), None) is not None:
        print("  detected IAC v1 Mechanical Turk layout")
        return load_iac_v1(root, flip_polarity=flip_polarity)

    candidates = list(root.rglob("*.csv")) + list(root.rglob("*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No .csv/.json found under {root}. Expected the extracted IAC v2 "
            "archive. IAC v2 is registration-gated: request it at "
            "https://nlds.soe.ucsc.edu/iac"
        )

    frames = []
    for path in candidates:
        try:
            df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_json(path)
        except Exception:
            continue
        # ADAPT: text column name varies across IAC releases.
        text_col = next(
            (c for c in ("text", "post_text", "body", "content") if c in df.columns),
            None,
        )
        present = [d for d in DISCOURSE_DIMENSIONS if d in df.columns]
        if text_col is None or not present:
            continue
        sub = df[[text_col] + present].rename(columns={text_col: "text"})
        frames.append(sub)

    if not frames:
        raise ValueError(
            f"Found files under {root} but none had a text column plus at least "
            f"one of {DISCOURSE_DIMENSIONS}. Inspect your IAC copy and adjust "
            "the ADAPT lines in load_iac()."
        )

    out = pd.concat(frames, ignore_index=True)
    # Average duplicate annotations of the same post, then dichotomize.
    out = out.groupby("text", as_index=False).mean(numeric_only=True)
    for dim in DISCOURSE_DIMENSIONS:
        if dim in out.columns:
            # Paper: cut on the scale midpoint, keeping midpoint observations.
            out[f"{dim}_bi"] = (out[dim] > 0).astype(int)
    out["dataset"] = "iac"
    return out


def load_ibm_rank(root: Path) -> pd.DataFrame:
    """IBM-Rank-30k -> argument quality (WA, and WA_bi cut at 0.5)."""
    files = list(root.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No .csv found under {root}. Expected IBM-ArgQ-Rank-30kArgs. "
            "Download (license acceptance required) from IBM Debater: "
            "https://research.ibm.com/haifa/dept/vst/debating_data.shtml"
        )

    frames = []
    for path in files:
        df = pd.read_csv(path)
        # ADAPT: IBM ships the weighted-average score as `WA`; some releases
        # use `MACE-P` as an alternative quality estimate.
        text_col = next((c for c in ("argument", "text") if c in df.columns), None)
        score_col = next((c for c in ("WA", "wa", "MACE-P") if c in df.columns), None)
        if text_col is None or score_col is None:
            continue
        sub = df[[text_col, score_col]].rename(
            columns={text_col: "text", score_col: "WA"}
        )
        frames.append(sub)

    if not frames:
        raise ValueError(
            f"Found CSVs under {root} but none had an argument/text column plus "
            "a WA or MACE-P score. Adjust the ADAPT lines in load_ibm_rank()."
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.groupby("text", as_index=False).mean(numeric_only=True)
    out["WA_bi"] = (out["WA"] > 0.5).astype(int)  # paper: midpoint of [0, 1]
    out["dataset"] = "ibm_rank_30k"
    return out


def load_ptc(root: Path, overlap_policy: str = "first") -> pd.DataFrame:
    """Da San Martino et al. (2019) propaganda corpus -> 18 technique columns.

    IMPORTANT: use the EMNLP 2019 PTC corpus, which annotates **18**
    techniques. The openly-licensed SemEval-2020 Task 11 release merged rare
    techniques down to **14** classes and will NOT fill all 18 columns here.
    Request the 18-technique corpus at https://propaganda.qcri.org/

    Expects article text files plus span-annotation files:
        <id>.txt                                      article text
        <id>.task[23]-TC.labels / *.labels            technique, start, end
    """
    label_files = [
        p for p in root.rglob("*")
        if p.suffix in (".labels", ".tsv", ".txt") and "label" in p.name.lower()
    ]
    article_files = {p.stem.replace("article", ""): p for p in root.rglob("article*.txt")}

    if not label_files or not article_files:
        raise FileNotFoundError(
            f"Expected article*.txt plus *.labels under {root}. Found "
            f"{len(article_files)} articles and {len(label_files)} label files. "
            "Request the 18-technique PTC corpus at https://propaganda.qcri.org/"
        )

    # sentence -> set of techniques
    sent_tech: dict[str, set[str]] = {}
    all_sentences: set[str] = set()

    for art_id, art_path in article_files.items():
        text = art_path.read_text(encoding="utf-8", errors="replace")
        # Character offsets in the labels are into the raw article, so track
        # each sentence's span to attribute techniques correctly.
        offset = 0
        spans = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped:
                spans.append((offset, offset + len(line), stripped))
                all_sentences.add(stripped)
            offset += len(line) + 1

        for lf in label_files:
            if art_id not in lf.name:
                continue
            for row in lf.read_text(encoding="utf-8", errors="replace").splitlines():
                parts = row.split("\t")
                if len(parts) < 4:
                    continue
                technique = parts[1].strip()
                try:
                    start, end = int(parts[2]), int(parts[3])
                except ValueError:
                    continue
                for s_start, s_end, sentence in spans:
                    if start < s_end and end > s_start:  # span overlaps sentence
                        sent_tech.setdefault(sentence, set()).add(technique)

    # ASSUMPTION (not in the paper): how to treat a sentence carrying several
    # technique spans. "first" keeps every technique; "drop-multi" discards
    # such sentences entirely, a stricter reading of "no_prop_overlap".
    if overlap_policy == "drop-multi":
        multi = {s for s, t in sent_tech.items() if len(t) > 1}
        for s in multi:
            sent_tech.pop(s)
            all_sentences.discard(s)

    rows = []
    for sentence in sorted(all_sentences):
        techniques = sent_tech.get(sentence, set())
        row = {"text": sentence, "dataset": "ptc"}
        for tech in PROPAGANDA_TECHNIQUES:
            row[tech] = int(tech in techniques)
        # Paper: the 18 techniques collapse to one binary propaganda label.
        # The corpus deliberately includes non-propaganda sentences.
        row["propaganda"] = int(bool(techniques))
        rows.append(row)

    if not rows:
        raise ValueError(f"Parsed no sentences from {root}.")

    unmatched = {t for ts in sent_tech.values() for t in ts} - set(PROPAGANDA_TECHNIQUES)
    if unmatched:
        print(
            f"  WARNING: {len(unmatched)} technique label(s) in the corpus do not "
            f"match the expected 18: {sorted(unmatched)}\n"
            "  This usually means you have the 14-class SemEval-2020 release "
            "rather than the 18-technique 2019 corpus.",
            file=sys.stderr,
        )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Augmentation (paper section: four kinds, training split only)
# --------------------------------------------------------------------------

def augment(df: pd.DataFrame, seed: int = 42, keep_original: bool = False) -> pd.DataFrame:
    """Apply the paper's four augmentations to a training frame.

    Needs optional extras:
        pip install nlpaug torch transformers sacremoses nltk
        python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

    Produces one augmented copy per method. keep_original=False (the default)
    drops the source rows, giving a 4x expansion, which is what the paper's
    counts imply: Argument Quality has 24,389 raw training rows in a rebuild
    here against 96,036 reported -- a factor of 3.94. Keeping the originals
    would give 5x and overshoot. Labels are copied unchanged from the source
    row (an assumption; the paper does not say).

    Returns the input untouched if nlpaug is unavailable.
    """
    try:
        import nlpaug.augmenter.word as naw
    except ImportError:
        print(
            "  nlpaug not installed - skipping augmentation. Table 1 counts "
            "will not be reproducible (they are roughly 4x the raw corpus).\n"
            "  pip install nlpaug torch transformers sacremoses nltk",
            file=sys.stderr,
        )
        return df

    random.seed(seed)
    np.random.seed(seed)

    augmenters = {
        "backtranslation": naw.BackTranslationAug(
            from_model_name="Helsinki-NLP/opus-mt-en-de",
            to_model_name="Helsinki-NLP/opus-mt-de-en",
        ),
        "contextual": naw.ContextualWordEmbsAug(
            model_path="bert-base-uncased", action="substitute", aug_p=0.30
        ),
        "synonym": naw.SynonymAug(aug_src="wordnet", aug_p=0.30),
        "crop": naw.RandomWordAug(action="delete", aug_p=0.30),
    }

    out = [df] if keep_original else []
    texts = df["text"].astype(str).tolist()
    for name, aug in augmenters.items():
        print(f"  augmenting: {name} ({len(texts):,} rows)...", flush=True)
        try:
            new_texts = aug.augment(texts)
        except Exception as exc:  # a single augmenter failing shouldn't kill the build
            print(f"    {name} failed ({exc}); skipping", file=sys.stderr)
            continue
        copy = df.copy()
        copy["text"] = [t if isinstance(t, str) else t[0] for t in new_texts]
        copy["dataset"] = copy["dataset"].astype(str) + f"_aug_{name}"
        out.append(copy)

    if not out:
        print("    every augmenter failed; returning the original rows", file=sys.stderr)
        return df
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------
# Merge, split, validate
# --------------------------------------------------------------------------

def merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Union the per-corpus frames, filling unannotated cells with NA."""
    combined = pd.concat(frames, ignore_index=True)
    for col in OUTPUT_COLUMNS:
        if col not in combined.columns:
            combined[col] = NA
    combined = combined[OUTPUT_COLUMNS]
    # A corpus that doesn't annotate a task leaves NA -> trainer masks it.
    return combined.fillna(NA)


def split(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, ...]:
    """80/10/10 random split, per the paper. Not stratified."""
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    return (
        shuffled.iloc[:n_train],
        shuffled.iloc[n_train:n_train + n_val],
        shuffled.iloc[n_train + n_val:],
    )


def validate(out_dir: Path) -> int:
    """Compare a rebuilt training split against the paper's Table 1."""
    path = out_dir / "no_prop_overlap_train.csv"
    if not path.exists():
        print(f"Not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    print(f"\n{path.name}: {len(df):,} rows x {len(df.columns)} columns\n")
    print(f"{'task':<26} {'rebuilt N':>10} {'paper N':>9} {'rebuilt bal':>12} {'paper bal':>10}")
    print("-" * 72)

    worst = 0.0
    for task, (paper_n, paper_bal) in PAPER_TABLE_1.items():
        if task not in df.columns:
            print(f"{task:<26} {'MISSING':>10} {paper_n:>9,}")
            worst = 1.0
            continue
        s = pd.to_numeric(df[task].replace([NA, "na", "NaN", ""], np.nan), errors="coerce")
        valid = s.dropna()
        n = len(valid)
        if n:
            pct1 = 100.0 * float((valid > 0).mean())
            bal = f"{100 - pct1:.0f}/{pct1:.0f}"
        else:
            bal = "-"
        drift = abs(n - paper_n) / paper_n if paper_n else 0.0
        worst = max(worst, drift)
        flag = "  <-- off by >25%" if drift > 0.25 else ""
        print(f"{task:<26} {n:>10,} {paper_n:>9,} {bal:>12} {paper_bal:>10}{flag}")

    print(f"\nLargest count deviation from Table 1: {worst * 100:.0f}%")
    if worst > 0.25:
        print(
            "A large gap usually means augmentation was skipped (counts ~4x too "
            "low), or the 14-class SemEval propaganda release was used."
        )
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iac", type=Path, help="extracted Internet Argument Corpus v2")
    p.add_argument("--ibm-rank", type=Path, help="extracted IBM-ArgQ-Rank-30kArgs")
    p.add_argument("--ptc", type=Path, help="extracted 18-technique PTC corpus")
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--augment", action="store_true", help="apply the paper's 4 augmentations to train")
    p.add_argument("--overlap-policy", choices=["first", "drop-multi"], default="first",
                   help="UNVERIFIED: how to treat sentences with several technique spans")
    p.add_argument("--augment-keep-original", action="store_true",
                   help="keep source rows alongside the 4 augmented copies (5x). Default "
                        "drops them (4x), which matches the paper's reported counts.")
    p.add_argument("--no-flip-polarity", action="store_true",
                   help="IAC: use score>0 as class 1. Default flips (score<0), which "
                        "matches the paper's Table 1 balances on 7 of 8 tasks.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validate-only", action="store_true", help="just check --out against Table 1")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        return validate(args.out)

    if not any([args.iac, args.ibm_rank, args.ptc]):
        p.error("give at least one of --iac / --ibm-rank / --ptc (all three for a full rebuild)")

    frames = []
    for flag, loader, kwargs in (
        (args.iac, load_iac, {"flip_polarity": not args.no_flip_polarity}),
        (args.ibm_rank, load_ibm_rank, {}),
        (args.ptc, load_ptc, {"overlap_policy": args.overlap_policy}),
    ):
        if flag is None:
            continue
        print(f"loading {loader.__name__} from {flag} ...")
        df = loader(flag, **kwargs)
        print(f"  -> {len(df):,} rows")
        frames.append(df)

    combined = merge(frames)
    print(f"\nmerged: {len(combined):,} rows x {len(combined.columns)} columns")

    train, val, test = split(combined, seed=args.seed)
    print(f"split: train={len(train):,}  val={len(val):,}  test={len(test):,}")

    if args.augment:
        print("\naugmenting training split (paper applies this to train only)...")
        train = augment(train, seed=args.seed, keep_original=args.augment_keep_original)
        print(f"  -> train={len(train):,} rows")

    for name, part in (("train", train), ("val", val), ("test", test)):
        dest = args.out / f"no_prop_overlap_{name}.csv"
        part.to_csv(dest, index=False)
        print(f"wrote {dest}  ({len(part):,} rows)")

    (args.out / "build_manifest.json").write_text(json.dumps({
        "seed": args.seed,
        "augmented": bool(args.augment),
        "overlap_policy": args.overlap_policy,
        "sources": {k: str(v) for k, v in
                    (("iac", args.iac), ("ibm_rank", args.ibm_rank), ("ptc", args.ptc)) if v},
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
    }, indent=2))

    print("\nnow check the rebuild against the paper:")
    print(f"  python {sys.argv[0]} --validate-only --out {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
