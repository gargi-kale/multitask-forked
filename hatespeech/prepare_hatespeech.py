"""Clean and split the Measuring Hate Speech data (hatespeech_dlab.csv).

Standalone: nothing here imports or changes the main model code.

What it does
1. Cleans the comment text to match the existing cleaned data (CMP style):
   plain ASCII, case kept, apostrophes deleted ("don't" -> "dont"),
   all other punctuation replaced by a space, single spaces.
   Also removes @handles at the start of a word, the "URL" placeholder and emoji.
   An @ or * inside a word (obfuscation such as "f@ggot", "d*ck") is deleted,
   not treated as a handle or punctuation.
2. Drops comments that are empty after cleaning, and keeps one copy of any
   text that appears more than once after cleaning (the copy with the most
   annotators), so the same text can't land in two splits.
3. Splits 80/10/10 into train/val/test with seed 42, stratified by platform
   and by whether the comment is rated hate speech (hatespeech score >= 1).
4. Keeps the 10 survey items as continuous scores (averages over annotators).
   The yes/no cut-off is NOT applied here; this script only reports the
   "yes" share at several cut-offs so one can be picked.

Usage
    python hatespeech/prepare_hatespeech.py \
        [--raw hatespeech/raw/hatespeech_dlab.csv] [--out hatespeech/data]
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
SEED = 42

# The 10 survey items. Higher = more hateful on every item.
# hatespeech is on 0-2 (no / unclear / yes); the others on 0-4.
ITEMS = ["sentiment", "respect", "insult", "humiliate", "status",
         "dehumanize", "violence", "genocide", "attack_defend", "hatespeech"]
ITEM_MAX = {c: (2 if c == "hatespeech" else 4) for c in ITEMS}

# Cut-offs reported for the yes/no conversion ("yes" if score >= cut-off,
# or > cut-off for the "gt" entries).
CUTOFFS_04 = [("> 1", "gt", 1.0), (">= 2", "ge", 2.0), ("> 2", "gt", 2.0), (">= 3", "ge", 3.0)]
CUTOFFS_02 = [(">= 0.5", "ge", 0.5), (">= 1", "ge", 1.0), ("> 1", "gt", 1.0), (">= 1.5", "ge", 1.5)]

_SMART = str.maketrans({"‘": "'", "’": "'", "‛": "'", "′": "'",
                        "“": '"', "”": '"', "–": "-", "—": "-"})


def clean_text(text: str) -> str:
    t = str(text).translate(_SMART)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")  # drops emoji, zero-width, folds accents
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = re.sub(r"\bURL\b", " ", t)                   # dataset's own link placeholder
    t = re.sub(r"(?<![A-Za-z0-9_])@\w+", " ", t)     # @handles
    t = re.sub(r"(?<=[A-Za-z])[@*]+(?=[A-Za-z])", "", t)  # in-word @ or * (obfuscation: f@ggot, d*ck)
    t = t.replace("'", "")                           # don't -> dont (as in CMP)
    t = re.sub(r"[^A-Za-z0-9 ]+", " ", t)            # all other punctuation
    return re.sub(r"\s+", " ", t).strip()


def yes_share(scores: pd.Series, op: str, cut: float) -> float:
    return float((scores > cut).mean() if op == "gt" else (scores >= cut).mean())


def cutoff_table(df: pd.DataFrame) -> str:
    lines = ["Share of comments marked 'yes' at each cut-off (all splits together, %)", ""]
    head04 = [c[0] for c in CUTOFFS_04]
    lines.append(f"{'item (0-4)':16s}" + "".join(f"{h:>9s}" for h in head04) + f"{'mean':>9s}")
    for c in ITEMS:
        if c == "hatespeech":
            continue
        row = "".join(f"{100 * yes_share(df[c], op, v):9.1f}" for _, op, v in CUTOFFS_04)
        lines.append(f"{c:16s}{row}{df[c].mean():9.2f}")
    lines.append("")
    head02 = [c[0] for c in CUTOFFS_02]
    lines.append(f"{'item (0-2)':16s}" + "".join(f"{h:>9s}" for h in head02) + f"{'mean':>9s}")
    row = "".join(f"{100 * yes_share(df['hatespeech'], op, v):9.1f}" for _, op, v in CUTOFFS_02)
    lines.append(f"{'hatespeech':16s}{row}{df['hatespeech'].mean():9.2f}")
    return "\n".join(lines)


def correlation_table(df: pd.DataFrame, label: str, cut04: tuple, cut02: tuple) -> str:
    b = pd.DataFrame({c: ((df[c] > (cut02 if c == "hatespeech" else cut04)[1])
                          if (cut02 if c == "hatespeech" else cut04)[0] == "gt"
                          else (df[c] >= (cut02 if c == "hatespeech" else cut04)[1])).astype(int)
                      for c in ITEMS})
    phi = b.corr()
    short = {c: c[:6] for c in ITEMS}
    lines = [f"Correlation (phi) between yes/no labels at {label}", "",
             f"{'':12s}" + "".join(f"{short[c]:>8s}" for c in ITEMS)]
    for r in ITEMS:
        lines.append(f"{r:12s}" + "".join(f"{phi.loc[r, c]:8.2f}" for c in ITEMS))
    off = phi.values[~np.eye(len(ITEMS), dtype=bool)]
    lines.append(f"\nmean off-diagonal phi {off.mean():.2f}  (min {off.min():.2f}, max {off.max():.2f})")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw", default=str(HERE / "raw" / "hatespeech_dlab.csv"))
    ap.add_argument("--out", default=str(HERE / "data"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.raw)
    log = [f"raw rows: {len(raw)}"]
    assert raw["comment_id"].is_unique, "expected one row per comment"
    assert raw[ITEMS].notna().all().all(), "expected every item on every row"
    for c in ITEMS:
        assert raw[c].between(0, ITEM_MAX[c]).all(), f"{c} outside 0-{ITEM_MAX[c]}"

    df = raw.copy()
    df["raw_text"] = df["text"]
    df["text"] = df["text"].map(clean_text)
    words = df["text"].str.split().str.len().fillna(0)

    empty = df["text"].eq("")
    log.append(f"empty after cleaning, dropped: {int(empty.sum())}")
    df = df[~empty]

    df = df.sort_values(["n_annotators", "comment_id"], ascending=[False, True])
    dup = df["text"].duplicated(keep="first")
    log.append(f"duplicate texts after cleaning, dropped (kept the copy with most annotators): {int(dup.sum())}")
    df = df[~dup].sort_values("comment_id").reset_index(drop=True)
    log.append(f"rows kept: {len(df)}")
    log.append(f"comments of 1-2 words kept: {int((df['text'].str.split().str.len() <= 2).sum())}")
    log.append(f"median words: {int(df['text'].str.split().str.len().median())} "
               f"(raw median {int(words.median())})")

    # Stratify by platform and hate speech yes/no. Platform 1 has only ~70 rows;
    # fold it into its own stratum only if it can be split, else merge with 0.
    hs = (df["hatespeech"] >= 1).astype(int)
    plat = df["platform"].where(df["platform"].map(df["platform"].value_counts()) >= 50, -1)
    strata = plat.astype(str) + "_" + hs.astype(str)
    train, rest = train_test_split(df, test_size=0.2, random_state=SEED, stratify=strata)
    val, test = train_test_split(rest, test_size=0.5, random_state=SEED, stratify=strata.loc[rest.index])

    target_cols = [c for c in df.columns if c.startswith("target_")]
    keep = ["comment_id", "text", "platform", "n_annotators", *ITEMS, "hate_speech_score", *target_cols]
    for name, part in (("train", train), ("val", val), ("test", test)):
        part.sort_values("comment_id")[keep].to_csv(out / f"hatespeech_{name}.csv", index=False)
        log.append(f"{name}: {len(part)} rows, hatespeech>=1 {100 * (part['hatespeech'] >= 1).mean():.1f}%, "
                   f"platforms {part['platform'].value_counts().sort_index().to_dict()}")

    # A few before/after examples for checking the cleaning by eye.
    ex = df[df["raw_text"].str.contains(r"@|[^\x00-\x7f]|URL|#", regex=True)].sample(8, random_state=SEED)
    examples = "\n".join(f"  RAW:   {r[:150]!r}\n  CLEAN: {c[:150]!r}" for r, c in zip(ex["raw_text"], ex["text"]))

    report = "\n\n".join([
        "\n".join(log),
        cutoff_table(df),
        correlation_table(df, "'> 2' (0-4 items) and '>= 1' (hatespeech)", ("gt", 2.0), ("ge", 1.0)),
        correlation_table(df, "'>= 2' (0-4 items) and '>= 1' (hatespeech)", ("ge", 2.0), ("ge", 1.0)),
        "Cleaning examples\n" + examples,
    ])
    (out / "prepare_report.txt").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
