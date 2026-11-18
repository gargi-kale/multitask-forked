"""What each column of a dataset is: text, source, or a label to train on.

There are two ways to get a description (a ``DatasetSpec``):

* ``detect(folder)`` looks at the train file and guesses. No file is needed.
  Command-line options (``--tasks``, ``--ignore``, ``--binarize``, ...) fix
  what the guess gets wrong.
* ``load_spec(path)`` reads a ``dataset.yaml``. Every run saves the spec it
  used as ``dataset.yaml`` next to its results, so a detected spec can be
  copied, edited (task groups, merged outputs) and passed back with
  ``--data path/to/dataset.yaml``.

``materialize`` then writes the labels in the one layout the trainer reads:
``train.csv`` / ``val.csv`` / ``test.csv`` with a ``text`` column, a
``dataset`` column (each row's source) and one column per task holding 0, 1
or blank (= not labelled).

Only yes/no (binary) tasks are trained. Scores, ratings and categories are
listed as "not used" with a hint, and become yes/no tasks with a cut-off
(``--binarize "violence>=2"`` or ``binarize:`` in the file).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")
SPLIT_WORDS = {"train": ("train", "training"), "val": ("val", "valid", "validation", "dev"),
               "test": ("test", "testing")}
DATA_EXTENSIONS = (".csv", ".tsv")
MISSING_STRINGS = ["", "NA", "na", "NaN", "nan", "N/A", "n/a", "null", "None"]
SOURCE_NAMES = ("source", "dataset", "corpus", "domain", "origin", "subset")
YES_NO = {("no", "yes"): "yes", ("false", "true"): "true", ("n", "y"): "y"}
RESERVED = ("text", "dataset")  # column names of the materialized files
OPS = (">=", "<=", "==", "!=", ">", "<")
POOL_STRATEGIES = ("max", "noisy_or", "lse")


def task_name(column: str) -> str:
    """A valid output name for a column (same rule the IAC code always used)."""
    safe = re.sub(r"[^A-Za-z0-9_.\\/>-]", "_", column)
    if not safe or not re.match(r"^[A-Za-z0-9.]", safe):
        safe = f"task_{safe}"
    return safe


# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------

@dataclass
class TaskSpec:
    name: str                 # output (head) name
    column: str               # column in the data files
    group: str = "all"        # task group: tasks of a group share a branch and a loss weight
    missing: List[float] = field(default_factory=list)  # values meaning "no label", e.g. -1
    binarize: Optional[str] = None  # cut-off making a yes/no label, e.g. ">=1" or "==yes"


@dataclass
class PoolSpec:
    name: str                 # name of the merged output
    tasks: List[str]          # tasks whose scores are merged (they keep their own heads)
    strategy: str = "max"     # max | noisy_or | lse


@dataclass
class DatasetSpec:
    name: str
    data_dir: str
    files: Dict[str, str]
    text_column: str = "text"
    source_column: Optional[str] = None
    tasks: List[TaskSpec] = field(default_factory=list)
    pools: List[PoolSpec] = field(default_factory=list)
    missing_values: List[str] = field(default_factory=lambda: list(MISSING_STRINGS))
    outputs: Optional[List[str]] = None       # scored outputs, in order (default: derived)
    consistency_skip: Dict[str, Any] = field(default_factory=dict)
    reference: Optional[Dict[str, Any]] = None  # {"title": ..., "f1": {output: F1 x100}}
    labels: Dict[str, str] = field(default_factory=dict)  # display names
    not_used: List[Dict[str, str]] = field(default_factory=list)  # detect(): skipped columns

    # -- structure ----------------------------------------------------------

    def groups(self) -> List[Tuple[str, List[str]]]:
        """[(group, [task names])] in order of first appearance."""
        out: Dict[str, List[str]] = {}
        for t in self.tasks:
            out.setdefault(t.group, []).append(t.name)
        return list(out.items())

    def task_names(self) -> List[str]:
        """Task names with each group's tasks together (the model needs that)."""
        return [t for _, ts in self.groups() for t in ts]

    def task(self, name: str) -> TaskSpec:
        for t in self.tasks:
            if t.name == name:
                return t
        raise KeyError(name)

    @property
    def pool(self) -> Optional[PoolSpec]:
        return self.pools[0] if self.pools else None

    def pooled_tasks(self) -> List[str]:
        return list(self.pool.tasks) if self.pool else []

    def scored_outputs(self) -> List[str]:
        """The outputs that are scored and reported: every task that is not
        part of a merged output, plus each merged output (at its first task)."""
        if self.outputs:
            return list(self.outputs)
        pooled = {t: p.name for p in self.pools for t in p.tasks}
        out: List[str] = []
        for t in self.task_names():
            name = pooled.get(t, t)
            if name not in out:
                out.append(name)
        return out

    def output_tasks(self, output: str) -> List[str]:
        """The task columns behind one scored output."""
        for p in self.pools:
            if p.name == output:
                return list(p.tasks)
        return [output]

    def group_of(self, output: str) -> str:
        return self.task(self.output_tasks(output)[0]).group

    def label(self, output: str) -> str:
        return self.labels.get(output, output)

    def expand(self, refs: Sequence[str]) -> List[str]:
        """Task names for references: a task or output name, a glob over task
        names or columns ("target_*"), "group:<name>" or "pool:<name>"."""
        names: List[str] = []
        for ref in refs:
            if ref.startswith("group:"):
                hit = [t.name for t in self.tasks if t.group == ref[6:]]
            elif ref.startswith("pool:") or any(p.name == ref for p in self.pools):
                hit = self.output_tasks(ref[5:] if ref.startswith("pool:") else ref)
            else:
                hit = [t.name for t in self.tasks
                       if fnmatch.fnmatchcase(t.name, ref) or fnmatch.fnmatchcase(t.column, ref)]
            if not hit:
                raise ValueError(f"{ref!r} matches no task of dataset {self.name!r} "
                                 f"(tasks: {', '.join(self.task_names())})")
            names += [h for h in hit if h not in names]
        return names

    def skip_map(self) -> Dict[str, List[str]]:
        """consistency_skip expanded to {task: [tasks]} (see LabelAwareMaskedModel)."""
        out: Dict[str, List[str]] = {}
        have = set(self.task_names())
        for key, val in self.consistency_skip.items():
            targets = [t for t in self.expand([key]) if t in have]
            refs = val if isinstance(val, list) else [val]
            skip = [t for t in self.expand(refs) if t in have]
            if skip:
                for t in targets:
                    out[t] = skip
        return out

    # -- selection ----------------------------------------------------------

    def select(self, tasks: Sequence[str] = (), ignore: Sequence[str] = ()) -> "DatasetSpec":
        """A copy with only ``tasks`` (default all) minus ``ignore``. Merged
        outputs lose dropped tasks; one left with no task is dropped."""
        keep = self.expand(tasks) if tasks else self.task_names()
        drop = set(self.expand(ignore)) if ignore else set()
        keep = [t for t in keep if t not in drop]
        if not keep:
            raise ValueError("no task left after --tasks / --ignore")
        spec = _copy(self)
        spec.tasks = [t for t in spec.tasks if t.name in keep]
        spec.pools = [PoolSpec(p.name, [t for t in p.tasks if t in keep], p.strategy)
                      for p in spec.pools]
        spec.pools = [p for p in spec.pools if p.tasks]
        if spec.outputs:
            valid = set(spec.task_names()) | {p.name for p in spec.pools}
            spec.outputs = [o for o in spec.outputs if o in valid]
        return spec

    def with_pool(self, name: str, refs: Sequence[str], strategy: str = "max") -> "DatasetSpec":
        """A copy with a merged output over the tasks ``refs`` (their scores
        merged by ``strategy``). The tasks move to one group, named after the
        pool unless they already share one."""
        if self.pools:
            raise ValueError("only one merged output per model is supported")
        if strategy not in POOL_STRATEGIES:
            raise ValueError(f"pool strategy must be one of {POOL_STRATEGIES}")
        members = self.expand(refs)
        if len(members) < 2:
            raise ValueError("a merged output needs at least two tasks")
        spec = _copy(self)
        groups = {spec.task(t).group for t in members}
        group = groups.pop() if len(groups) == 1 else name
        for t in spec.tasks:
            if t.name in members:
                t.group = group
        spec.pools = [PoolSpec(name, members, strategy)]
        spec.validate()
        return spec

    def validate(self) -> None:
        names = [t.name for t in self.tasks]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ValueError(f"task names used twice: {dup}")
        bad = [n for n in names if n in RESERVED]
        if bad:
            raise ValueError(f"task names {bad} are reserved; rename them in the spec")
        if len(self.pools) > 1:
            raise ValueError("only one merged output (pool) per model is supported")
        for p in self.pools:
            if p.strategy not in POOL_STRATEGIES:
                raise ValueError(f"pool {p.name}: strategy must be one of {POOL_STRATEGIES}")
            unknown = [t for t in p.tasks if t not in names]
            if unknown:
                raise ValueError(f"pool {p.name}: unknown tasks {unknown}")
            if len({self.task(t).group for t in p.tasks}) != 1:
                raise ValueError(f"pool {p.name}: its tasks must be in one group")
            if p.name in names:
                raise ValueError(f"pool {p.name} has the same name as a task")
        for t in self.tasks:
            if t.binarize:
                parse_cut(t.binarize)
        for split in SPLITS:
            if split not in self.files:
                raise ValueError(f"no {split} file given")
        if self.outputs:
            valid = set(names) | {p.name for p in self.pools}
            bad = [o for o in self.outputs if o not in valid]
            if bad:
                raise ValueError(f"outputs {bad} are neither tasks nor pools")
        if self.reference:
            if not isinstance(self.reference.get("f1"), dict):
                raise ValueError("reference needs an f1: {output: score} mapping")

    # -- files --------------------------------------------------------------

    def path(self, split: str) -> str:
        return os.path.join(self.data_dir, self.files[split])

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for t in d["tasks"]:
            if t["column"] == t["name"]:
                del t["column"]
            if not t["missing"]:
                del t["missing"]
            if t["binarize"] is None:
                del t["binarize"]
        for k in ("outputs", "reference", "source_column"):
            if d[k] is None:
                del d[k]
        for k in ("pools", "consistency_skip", "labels", "not_used"):
            if not d[k]:
                del d[k]
        if d["missing_values"] == MISSING_STRINGS:
            del d["missing_values"]
        return d


def _copy(spec: DatasetSpec) -> DatasetSpec:
    return spec_from_dict(json.loads(json.dumps(spec.to_dict())), base_dir=None)


def spec_from_dict(d: Dict[str, Any], base_dir: Optional[str]) -> DatasetSpec:
    d = dict(d)
    tasks = [TaskSpec(name=t["name"], column=t.get("column", t["name"]),
                      group=str(t.get("group", "all")),
                      missing=[float(x) for x in t.get("missing", [])],
                      binarize=t.get("binarize")) for t in d.pop("tasks", [])]
    pools = [PoolSpec(p["name"], list(p["tasks"]), p.get("strategy", "max"))
             for p in d.pop("pools", [])]
    data_dir = d.pop("data_dir", ".")
    if base_dir is not None and not os.path.isabs(data_dir):
        data_dir = os.path.normpath(os.path.join(base_dir, data_dir))
    known = {f.name for f in DatasetSpec.__dataclass_fields__.values()}
    extra = sorted(set(d) - known)
    if extra:
        raise ValueError(f"unknown keys in dataset spec: {extra}")
    spec = DatasetSpec(data_dir=data_dir, tasks=tasks, pools=pools, **d)
    spec.validate()
    return spec


def load_spec(path: str) -> DatasetSpec:
    import yaml

    with open(path) as f:
        d = yaml.safe_load(f)
    return spec_from_dict(d, base_dir=os.path.dirname(os.path.abspath(path)))


def save_spec(spec: DatasetSpec, path: str, header: str = "") -> None:
    import yaml

    d = spec.to_dict()
    d["data_dir"] = os.path.abspath(spec.data_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        if header:
            f.write("".join(f"# {line}\n" for line in header.splitlines()))
        yaml.safe_dump(d, f, sort_keys=False, allow_unicode=True, width=100)


# ---------------------------------------------------------------------------
# Reading files and turning a column into labels
# ---------------------------------------------------------------------------

def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".tsv"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def parse_cut(rule: str) -> Tuple[str, Any]:
    """'>=1' -> ('>=', 1.0); '==yes' -> ('==', 'yes')."""
    rule = rule.strip()
    for op in OPS:
        if rule.startswith(op):
            raw = rule[len(op):].strip()
            if not raw:
                break
            try:
                return op, float(raw)
            except ValueError:
                if op not in ("==", "!="):
                    raise ValueError(f"cut-off {rule!r}: {op} needs a number")
                return op, raw.lower()
    raise ValueError(f"cut-off must look like '>=1', '>2', '==yes' or '!=0', got {rule!r}")


def column_labels(s: pd.Series, task: TaskSpec, missing_values: Sequence[str]) -> np.ndarray:
    """float32 array of 0/1 with NaN where the row has no label."""
    s = s.copy()
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        s = s.where(~s.astype(str).str.strip().isin(list(missing_values)), np.nan)
    if task.binarize:
        op, value = parse_cut(task.binarize)
        missing = np.array(s.isna().to_numpy(), dtype=bool)
        if isinstance(value, str):
            v = s.astype(str).str.strip().str.lower()
            hit = (v == value) if op == "==" else (v != value)
            out = np.array(hit.to_numpy(), dtype=np.float32)
        else:
            num = pd.to_numeric(s, errors="coerce")
            missing |= num.isna().to_numpy()
            for code in task.missing:
                missing |= (num == code).to_numpy()
            x = np.array(num.to_numpy(dtype=np.float64))
            with np.errstate(invalid="ignore"):
                out = {">=": x >= value, "<=": x <= value, ">": x > value, "<": x < value,
                       "==": x == value, "!=": x != value}[op].astype(np.float32)
        out[missing] = np.nan
        return out
    if pd.api.types.is_bool_dtype(s):
        return np.array(s.astype(np.float32).to_numpy(), dtype=np.float32)
    out = np.array(pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32))
    for code in task.missing:
        out[out == code] = np.nan
    return out


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def split_prefix(path: str) -> Tuple[str, str]:
    """(folder, prefix) for one of a dataset's files, e.g.
    data/no_prop_overlap_train.csv -> ("data", "no_prop_overlap_")."""
    folder, name = os.path.split(os.path.abspath(path))
    stem = os.path.splitext(name)[0]
    words = [w for ws in SPLIT_WORDS.values() for w in ws]
    m = re.search(r"(?i)(?:^|[^a-z0-9])(" + "|".join(sorted(words, key=len, reverse=True))
                  + r")(?=$|[^a-z0-9])", stem)
    if not m:
        raise ValueError(f"{path}: the file name has no split word (train / val / dev / test)")
    return folder, stem[:m.start(1)]


def find_split_files(folder: str, prefix: str = "") -> Dict[str, str]:
    """{split: file name} for the train / val / test files in ``folder``,
    recognised by a word in the name (train; val, valid, dev; test). With
    ``prefix``, only files whose names start with it."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"data folder {folder!r} not found")
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(DATA_EXTENSIONS)
                   and f.startswith(prefix))
    found: Dict[str, List[str]] = {s: [] for s in SPLITS}
    for f in files:
        words = set(re.split(r"[^a-z0-9]+", os.path.splitext(f.lower())[0]))
        for split, keys in SPLIT_WORDS.items():
            if words & set(keys):
                found[split].append(f)
    missing = [s for s in SPLITS if not found[s]]
    if missing:
        raise ValueError(
            f"{folder}: no {' / '.join(missing)} file found (looked for names containing "
            f"train / val, valid, dev / test among {files or 'no .csv or .tsv files'}). "
            "If you have a single file, split it first: multitask prepare --input FILE --out DIR")
    many = {s: v for s, v in found.items() if len(v) > 1}
    if many:
        raise ValueError(f"{folder}: more than one file per split {many}; point --data at "
                         "one of the files you mean (e.g. --data FOLDER/NAME_train.csv)")
    return {s: v[0] for s, v in found.items()}


@dataclass
class ColumnReport:
    column: str
    role: str        # text | source | task | not used
    detail: str


def _is_text(s: pd.Series) -> float:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return 0.0
    nn = s.dropna().astype(str)
    return float(nn.str.len().mean()) if len(nn) else 0.0


def _classify(s: pd.Series, missing_values: Sequence[str]) -> Tuple[Optional[TaskSpec], str]:
    """(task or None, reason) for one column."""
    if pd.api.types.is_bool_dtype(s):
        return TaskSpec(name="", column=s.name), "yes/no (true/false)"
    if not pd.api.types.is_numeric_dtype(s):
        v = s.where(~s.astype(str).str.strip().isin(list(missing_values)), np.nan).dropna()
        if v.empty:
            return None, "empty"
        uniq = tuple(sorted(set(v.astype(str).str.strip().str.lower())))
        num = pd.to_numeric(v, errors="coerce")
        if num.notna().mean() < 0.99:
            if uniq in YES_NO:
                yes = YES_NO[uniq]
                return TaskSpec(name="", column=s.name, binarize=f"=={yes}"), f"yes/no ('{yes}' = 1)"
            return None, (f"text with {len(uniq)} distinct values; multi-class labels are not "
                          f"supported, use --binarize \"{s.name}==VALUE\"")
        s = pd.to_numeric(s.where(~s.astype(str).str.strip().isin(list(missing_values)),
                                  np.nan), errors="coerce")
    nn = s.dropna()
    if nn.empty:
        return None, "no values"
    u = np.unique(nn.to_numpy())
    vals = set(np.round(u, 6).tolist()) if len(u) <= 50 else None
    if vals is not None and vals == {0.0, 1.0}:
        return TaskSpec(name="", column=s.name), "yes/no"
    if vals is not None and vals == {-1.0, 0.0, 1.0}:
        return TaskSpec(name="", column=s.name, missing=[-1.0]), "yes/no, -1 = no label"
    if len(u) == 1:
        return None, f"only one value ({u[0]:g})"
    whole = bool(np.all(np.equal(np.mod(u, 1), 0)))
    if whole and len(u) == len(nn) and len(nn) > 100:
        return None, "a different whole number on every row (an ID?)"
    if whole and len(u) <= 12:
        return None, (f"whole numbers {u.min():g}..{u.max():g} ({len(u)} values: classes or a "
                      f"rating); make it yes/no with --binarize \"{s.name}>=K\"")
    return None, (f"a score from {u.min():.3g} to {u.max():.3g}; make it yes/no with "
                  f"--binarize \"{s.name}>=CUT\"")


def parse_binarize_rules(rules: Sequence[str]) -> List[Tuple[List[str], str]]:
    """'violence,insult>2' -> (['violence', 'insult'], '>2'). Column parts may be globs."""
    out = []
    for rule in rules:
        m = re.match(r"^\s*([^<>=!]+?)\s*((?:>=|<=|==|!=|>|<).+)$", rule)
        if not m:
            raise ValueError(f"--binarize {rule!r}: expected COLUMNS OP VALUE, "
                             "e.g. 'violence>=2' or 'sentiment,respect>2'")
        parse_cut(m.group(2))
        out.append(([c.strip() for c in m.group(1).split(",") if c.strip()], m.group(2).strip()))
    return out


def detect(folder: str, *, prefix: str = "", name: Optional[str] = None,
           text_column: Optional[str] = None,
           source_column: Optional[str] = None, binarize: Sequence[str] = (),
           group_by: str = "auto", missing_values: Sequence[str] = tuple(MISSING_STRINGS)
           ) -> Tuple[DatasetSpec, List[ColumnReport]]:
    """Guess a DatasetSpec from the files in ``folder`` (see module docstring).

    ``group_by``: "source" puts each task in the group of the source that
    labels most of its rows; "single" puts every task in one group; "auto"
    is "source" when there is a source column with more than one value."""
    files = find_split_files(folder, prefix)
    df = read_table(os.path.join(folder, files["train"]))
    report: List[ColumnReport] = []

    if text_column is None:
        lengths = {c: _is_text(df[c]) for c in df.columns}
        text_column = max(lengths, key=lengths.get) if lengths else None
        if not text_column or lengths[text_column] < 15:
            raise ValueError(f"{folder}: no text column found (no column of long strings); "
                             "name it with --text-column")
    elif text_column not in df.columns:
        raise ValueError(f"--text-column {text_column!r} is not a column: {list(df.columns)}")
    report.append(ColumnReport(text_column, "text",
                               f"average {_is_text(df[text_column]):.0f} characters"))

    if source_column is None:
        for c in df.columns:
            if (c != text_column and c.lower() in SOURCE_NAMES
                    and not pd.api.types.is_numeric_dtype(df[c])
                    and 1 <= df[c].nunique() <= 100 and df[c].notna().all()):
                source_column = c
                break
    elif source_column not in df.columns:
        raise ValueError(f"--source-column {source_column!r} is not a column: {list(df.columns)}")
    if source_column:
        counts = df[source_column].value_counts()
        report.append(ColumnReport(source_column, "source", ", ".join(
            f"{k} {v:,}" for k, v in counts.items())))

    rules = parse_binarize_rules(binarize)
    matched_rules = set()
    tasks: List[TaskSpec] = []
    not_used: List[Dict[str, str]] = []
    for c in df.columns:
        if c in (text_column, source_column):
            continue
        rule = next((r for pats, r in rules
                     if any(fnmatch.fnmatchcase(c, p) for p in pats)), None)
        if rule is not None:
            matched_rules.add(rule)
            t = TaskSpec(name="", column=c, binarize=rule)
            reason = f"yes/no by cut-off {rule}"
        else:
            t, reason = _classify(df[c], missing_values)
        if t is not None:
            y = column_labels(df[c], t, missing_values)
            ok = ~np.isnan(y)
            if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
                t, reason = None, f"{reason}, but only one class in train"
        if t is None:
            report.append(ColumnReport(c, "not used", reason))
            not_used.append({"column": c, "reason": reason})
            continue
        t.name = task_name(c)
        yes = float(np.nanmean(y))
        reason += f"; labelled {100 * ok.mean():.0f}% of rows, {100 * yes:.0f}% yes"
        report.append(ColumnReport(c, "task", reason))
        tasks.append(t)
    unused_rules = [r for pats, r in rules if r not in matched_rules]
    if unused_rules:
        raise ValueError(f"--binarize rules matched no column: {unused_rules}")
    if not tasks:
        raise ValueError(f"{folder}: no yes/no label column found. Turn scores into yes/no "
                         "labels with --binarize, e.g. --binarize \"score>=3\"")
    names = [t.name for t in tasks]
    clash = sorted({n for n in names if names.count(n) > 1} | {n for n in names if n in RESERVED})
    if clash:
        raise ValueError(f"columns {clash} give clashing or reserved task names; rename them")

    if group_by not in ("auto", "source", "single"):
        raise ValueError("group_by must be auto, source or single")
    use_source = source_column is not None and df[source_column].nunique() > 1 and \
        group_by in ("auto", "source")
    if group_by == "source" and not use_source:
        raise ValueError("--groups source needs a source column with more than one value")
    for t in tasks:
        if use_source:
            labelled = ~np.isnan(column_labels(df[t.column], t, missing_values))
            t.group = task_name(str(df.loc[labelled, source_column].value_counts().idxmax()))
        else:
            t.group = "all"

    spec = DatasetSpec(
        name=name or prefix.strip("_-. ") or os.path.basename(os.path.normpath(folder)),
        data_dir=folder, files=files,
        text_column=text_column, source_column=source_column, tasks=tasks,
        missing_values=list(missing_values), not_used=not_used)
    spec.validate()
    return spec, report


def describe(spec: DatasetSpec, report: Optional[List[ColumnReport]] = None) -> str:
    """Human-readable summary of what a run will train on."""
    lines = [f"Dataset {spec.name!r} ({spec.data_dir}): files "
             + ", ".join(f"{s} {spec.files[s]}" for s in SPLITS)]
    if report:
        w = min(max(len(r.column) for r in report), 40)
        left = [r for r in report if r.role == "left out"]
        for r in report:
            if r.role != "left out" or len(left) <= 5:
                lines.append(f"  {r.column[:w]:<{w}}  {r.role:<8}  {r.detail}")
        if len(left) > 5:
            lines.append(f"  ({len(left)} more columns left out by --tasks / --ignore: "
                         f"{', '.join(r.column for r in left[:4])}, ...)")
    lines.append("Task groups: " + "; ".join(f"{g} ({len(ts)}: {', '.join(ts)})"
                                             for g, ts in spec.groups()))
    for p in spec.pools:
        lines.append(f"Merged output {p.name}: {p.strategy} over {len(p.tasks)} tasks")
    lines.append(f"Scored outputs ({len(spec.scored_outputs())}): "
                 + ", ".join(spec.scored_outputs()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Materialize: the standard files the trainer reads
# ---------------------------------------------------------------------------

def _fingerprint(specs: Sequence[DatasetSpec], extra: Dict[str, Any]) -> str:
    h = hashlib.sha256()
    for spec in specs:
        h.update(json.dumps({**spec.to_dict(), "data_dir": os.path.abspath(spec.data_dir)},
                            sort_keys=True).encode())
        for split in SPLITS:
            st = os.stat(spec.path(split))
            h.update(f"{split}:{st.st_size}:{st.st_mtime_ns}".encode())
    h.update(json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()[:16]


def join_specs(specs: Sequence[DatasetSpec]) -> DatasetSpec:
    """One spec for datasets trained together in one (joint) model: the union
    of their tasks; each dataset's rows have no label for the others' tasks."""
    if len(specs) == 1:
        return specs[0]
    names = [t.name for s in specs for t in s.tasks]
    clash = sorted({n for n in names if names.count(n) > 1})
    if clash:
        raise ValueError(f"datasets share task names {clash}; rename them in one spec")
    if sum(len(s.pools) for s in specs) > 1:
        raise ValueError("only one merged output per model: at most one dataset may have a pool")
    group_owner: Dict[str, str] = {}
    tasks: List[TaskSpec] = []
    for s in specs:
        for t in s.tasks:
            g = t.group
            if group_owner.setdefault(g, s.name) != s.name:
                g = f"{s.name}_{g}"
            tasks.append(TaskSpec(t.name, t.column, g, list(t.missing), t.binarize))
    joined = DatasetSpec(
        name="+".join(s.name for s in specs), data_dir=specs[0].data_dir,
        files=dict(specs[0].files), tasks=tasks, pools=[p for s in specs for p in s.pools],
        outputs=None if not any(s.outputs for s in specs)
        else [o for s in specs for o in s.scored_outputs()],
        consistency_skip={k: v for s in specs for k, v in s.consistency_skip.items()},
        labels={k: v for s in specs for k, v in s.labels.items()})
    joined.validate()
    return joined


def materialize(specs: Sequence[DatasetSpec], cache_root: str, *,
                keep_sources: Sequence[str] = (),
                hide_source: Optional[Tuple[str, Sequence[str]]] = None,
                verbose: bool = True) -> DatasetSpec:
    """Write the standard train/val/test files for ``specs`` (several = one
    joint model) under ``cache_root`` and return the spec that describes them.

    ``keep_sources`` keeps only rows of those sources (in every split).
    ``hide_source`` = (source, tasks) removes those tasks' labels on that
    source's rows in every split (e.g. labels made by a rule rather than by
    annotators). Files are reused when nothing changed."""
    joined = join_specs(list(specs))
    extra = {"keep_sources": list(keep_sources),
             "hide_source": [hide_source[0], list(hide_source[1])] if hide_source else None}
    out_dir = os.path.join(cache_root, _fingerprint(specs, extra))
    out_spec = DatasetSpec(
        name=joined.name, data_dir=out_dir, files={s: f"{s}.csv" for s in SPLITS},
        text_column="text", source_column="dataset",
        tasks=[TaskSpec(t.name, t.name, t.group) for t in joined.tasks],
        pools=joined.pools, outputs=joined.outputs, consistency_skip=joined.consistency_skip,
        reference=specs[0].reference if len(specs) == 1 else None, labels=joined.labels)
    if hide_source:
        unknown = [t for t in hide_source[1] if t not in out_spec.task_names()]
        if unknown:
            raise ValueError(f"hide_source: unknown tasks {unknown}")
    if all(os.path.exists(os.path.join(out_dir, f"{s}.csv")) for s in SPLITS):
        if verbose:
            print(f"Using prepared data in {out_dir}", flush=True)
        return out_spec
    os.makedirs(out_dir, exist_ok=True)
    order = out_spec.task_names()
    for split in SPLITS:
        parts = []
        for spec in specs:
            df = read_table(spec.path(split))
            missing_cols = [c for c in [spec.text_column] + [t.column for t in spec.tasks]
                            if c not in df.columns]
            if missing_cols:
                raise ValueError(f"{spec.path(split)}: missing columns {missing_cols}")
            part = pd.DataFrame({"text": df[spec.text_column]})
            part["dataset"] = (df[spec.source_column].astype(str) if spec.source_column
                               else spec.name)
            for t in spec.tasks:
                part[t.name] = column_labels(df[t.column], t, spec.missing_values)
            parts.append(part)
        df = pd.concat(parts, ignore_index=True)
        for t in order:
            if t not in df:
                df[t] = np.nan
        df = df[["text", "dataset", *order]]
        if keep_sources:
            unknown = sorted(set(keep_sources) - set(df["dataset"].unique()))
            if unknown and split == "train":
                raise ValueError(f"--keep-sources: no rows of {unknown}; sources are "
                                 f"{sorted(df['dataset'].unique())}")
            df = df[df["dataset"].isin(list(keep_sources))].reset_index(drop=True)
        if hide_source:
            source, cols = hide_source
            rows = (df["dataset"] == source).to_numpy()
            n = int(df.loc[rows, list(cols)].notna().to_numpy().sum())
            df.loc[rows, list(cols)] = np.nan
            if verbose:
                print(f"{split}: removed {n:,} labels of {len(cols)} tasks on "
                      f"{int(rows.sum()):,} {source} rows", flush=True)
        tmp = os.path.join(out_dir, f".{split}.csv.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, os.path.join(out_dir, f"{split}.csv"))
        if verbose:
            print(f"{split}: {len(df):,} rows " + ", ".join(
                f"{k} {v:,}" for k, v in df["dataset"].value_counts().items()), flush=True)
    return out_spec
