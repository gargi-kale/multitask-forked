"""Clean text and make train / val / test files (``multitask prepare``).

Input: one file (it is split) or a folder that already has train / val /
test files (each is cleaned, the split is kept). CSV, TSV, Excel and JSON
lines are read; CSV is written.

Cleaning, in this order (each step optional; ``--style cmp`` turns on all of
them, matching the cleaned CMP and hate speech data):
  ascii      plain ASCII: accents folded, emoji and zero-width characters dropped
  urls       links removed (and any --remove-token placeholders such as URL)
  handles    @handles at the start of a word removed; @ or * inside a word
             deleted ("f@ggot" -> "fggot")
  punct      apostrophes deleted ("don't" -> "dont"), all other punctuation
             becomes a space
  lower      lower case
Always: spaces collapsed, ends trimmed. Then rows with fewer than
--min-words words are dropped and, unless --keep-duplicates, only one copy of
each text is kept (the first, or the one with the highest --dedupe-keep-max
column), so a text can't land in two splits.

Split: --split 0.8,0.1,0.1 with --seed, stratified by --stratify columns
(a column, or a cut-off like "hatespeech>=1"); strata too small to split are
merged into one.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from multitask.train.dataset import (
    SPLITS, column_labels, describe, detect, find_split_files, parse_binarize_rules, TaskSpec,
)

_SMART = str.maketrans({"‘": "'", "’": "'", "‛": "'", "′": "'",
                        "“": '"', "”": '"', "–": "-", "—": "-"})
STEPS = ("ascii", "urls", "handles", "punct", "lower")
STYLES = {"none": (), "light": (), "cmp": ("ascii", "urls", "handles", "punct")}


def clean_text(text: str, steps: Sequence[str] = (), remove_tokens: Sequence[str] = ()) -> str:
    t = "" if text is None or (isinstance(text, float) and np.isnan(text)) else str(text)
    if "ascii" in steps:
        t = t.translate(_SMART)
        t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    if "urls" in steps:
        t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    for tok in remove_tokens:
        t = re.sub(rf"\b{re.escape(tok)}\b", " ", t)
    if "handles" in steps:
        t = re.sub(r"(?<![A-Za-z0-9_])@\w+", " ", t)
        t = re.sub(r"(?<=[A-Za-z])[@*]+(?=[A-Za-z])", "", t)
    if "punct" in steps:
        t = t.replace("'", "")
        t = re.sub(r"[^A-Za-z0-9 ]+" if "ascii" in steps else r"[^\w ]+", " ", t)
    if "lower" in steps:
        t = t.lower()
    return re.sub(r"\s+", " ", t).strip()


def read_any(path: str) -> pd.DataFrame:
    low = path.lower()
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if low.endswith((".jsonl", ".ndjson")):
        return pd.read_json(path, lines=True)
    if low.endswith(".json"):
        return pd.read_json(path)
    if low.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def _guess_text_column(df: pd.DataFrame) -> str:
    best, best_len = None, 0.0
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        n = df[c].dropna().astype(str).str.len().mean()
        if n > best_len:
            best, best_len = c, n
    if best is None or best_len < 15:
        raise ValueError("no text column found; name it with --text-column")
    return best


def _strata(df: pd.DataFrame, stratify: Sequence[str], min_count: int) -> Optional[pd.Series]:
    if not stratify:
        return None
    parts = []
    for item in stratify:
        if any(op in item for op in ("<", ">", "=")):
            (cols, rule), = parse_binarize_rules([item])
            y = column_labels(df[cols[0]], TaskSpec(name=cols[0], column=cols[0], binarize=rule), [])
            parts.append(pd.Series(np.where(np.isnan(y), "na", y.astype(str)), index=df.index))
        else:
            if item not in df.columns:
                raise ValueError(f"--stratify {item!r} is not a column")
            parts.append(df[item].astype(str))
    s = parts[0]
    for p in parts[1:]:
        s = s + "|" + p
    counts = s.map(s.value_counts())
    return s.where(counts >= min_count, "rare")


def prepare(inputs: str, out_dir: str, *, name: Optional[str] = None,
            text_column: Optional[str] = None, style: str = "light",
            steps: Sequence[str] = (), remove_tokens: Sequence[str] = (),
            min_words: int = 1, keep_duplicates: bool = False,
            dedupe_keep_max: Optional[str] = None, split: Sequence[float] = (0.8, 0.1, 0.1),
            seed: int = 42, stratify: Sequence[str] = ()) -> str:
    """Clean and split; writes ``<out_dir>/<name>_{train,val,test}.csv`` and
    ``prepare_report.txt``. Returns the report text."""
    from sklearn.model_selection import train_test_split

    if style not in STYLES:
        raise ValueError(f"--style must be one of {', '.join(STYLES)}")
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise ValueError(f"unknown cleaning steps {bad}; choose from {STEPS}")
    steps = list(dict.fromkeys([*STYLES[style], *steps]))
    if len(split) != 3 or abs(sum(split) - 1) > 1e-6 or min(split) <= 0:
        raise ValueError("--split needs three positive shares that add up to 1, e.g. 0.8,0.1,0.1")

    if os.path.isdir(inputs):
        files = find_split_files(inputs)
        frames = {s: read_any(os.path.join(inputs, f)) for s, f in files.items()}
        presplit = True
        name = name or os.path.basename(os.path.normpath(inputs))
    else:
        frames = {"all": read_any(inputs)}
        presplit = False
        name = name or os.path.splitext(os.path.basename(inputs))[0]
    first = next(iter(frames.values()))
    text_column = text_column or _guess_text_column(first)
    log = [f"input: {inputs}", f"text column: {text_column}",
           f"cleaning: {', '.join(steps) or 'spaces only'}"
           + (f"; removed tokens {list(remove_tokens)}" if remove_tokens else "")]

    examples = []
    cleaned: Dict[str, pd.DataFrame] = {}
    for part, df in frames.items():
        if text_column not in df.columns:
            raise ValueError(f"{part}: no column {text_column!r}")
        df = df.copy()
        raw = df[text_column]
        df[text_column] = [clean_text(t, steps, remove_tokens) for t in raw]
        changed = raw.astype(str) != df[text_column]
        examples += [(r, c) for r, c in zip(raw[changed][:4], df[text_column][changed][:4])]
        short = df[text_column].str.split().str.len().fillna(0) < max(min_words, 1)
        log.append(f"{part}: {len(df):,} rows; {int(changed.sum()):,} texts changed by cleaning; "
                   f"{int(short.sum()):,} dropped as shorter than {max(min_words, 1)} word(s)")
        cleaned[part] = df[~short]

    if not keep_duplicates:
        if dedupe_keep_max:
            for part in cleaned:
                cleaned[part] = cleaned[part].sort_values(dedupe_keep_max, ascending=False,
                                                          kind="stable")
        seen: set = set()
        for part in (["all"] if not presplit else list(SPLITS)):
            df = cleaned[part]
            dup = df[text_column].duplicated(keep="first") | df[text_column].isin(seen)
            seen |= set(df[text_column])
            log.append(f"{part}: {int(dup.sum()):,} duplicate texts dropped"
                       + (f" (kept the copy with the highest {dedupe_keep_max})"
                          if dedupe_keep_max else "")
                       + (" (including copies of texts in earlier splits)" if presplit else ""))
            cleaned[part] = df[~dup].sort_index()

    if presplit:
        out = {s: cleaned[s] for s in SPLITS}
    else:
        df = cleaned["all"]
        strata = _strata(df, stratify, min_count=max(10, int(np.ceil(4 / min(split)))))
        train, rest = train_test_split(df, test_size=split[1] + split[2], random_state=seed,
                                       stratify=strata)
        val, test = train_test_split(rest, test_size=split[2] / (split[1] + split[2]),
                                     random_state=seed,
                                     stratify=None if strata is None else strata.loc[rest.index])
        out = {"train": train.sort_index(), "val": val.sort_index(), "test": test.sort_index()}
        log.append(f"split {split[0]:g}/{split[1]:g}/{split[2]:g}, seed {seed}"
                   + (f", stratified by {', '.join(stratify)}" if stratify else ""))

    os.makedirs(out_dir, exist_ok=True)
    for s, df in out.items():
        df.to_csv(os.path.join(out_dir, f"{name}_{s}.csv"), index=False)
        log.append(f"{s}: {len(df):,} rows -> {os.path.join(out_dir, f'{name}_{s}.csv')}")

    try:
        spec, report = detect(out_dir, name=name, text_column=text_column)
        found = "What `multitask train` will detect in these files:\n" + describe(spec, report)
    except ValueError as exc:
        found = f"Detection on the output: {exc}"
    ex = "\n".join(f"  RAW:   {str(r)[:150]!r}\n  CLEAN: {c[:150]!r}" for r, c in examples[:8])
    text = "\n\n".join(["\n".join(log), found] + ([f"Cleaning examples\n{ex}"] if ex else []))
    with open(os.path.join(out_dir, "prepare_report.txt"), "w") as f:
        f.write(text + "\n")
    return text
