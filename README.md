# multitask

Train multi-task text classifiers on your own data, using the methods and
ablations from our replication of *Multi-Task Learning Improves Performance in
Deep Argument Mining Models* (Farzam et al., 2024).

You give it a folder of CSV files with a text column and some yes/no label
columns. It works out what each column is, trains one model that predicts all
the labels at once, and reports weighted F1 and AUC for each label. You don't
need to write a configuration file. One is saved with every run, and you only
edit it if the automatic guess was wrong.

```bash
multitask detect --data data/my_data          # check what will be trained
multitask train  --data data/my_data          # train (default method: famo_consf1_m5)
multitask report runs/my_data                 # summarise the results
```

- [Install](#install)
- [Quick start](#quick-start)
- [What your data needs](#what-your-data-needs)
- [Methods](#methods)
- [Ablations](#ablations)
- [Several datasets in one model](#several-datasets-in-one-model)
- [Cleaning and splitting data](#cleaning-and-splitting-data)
- [Reproducing the paper's setup](#reproducing-the-papers-setup)
- [What a run writes](#what-a-run-writes)
- [Limits](#limits)
- [Tests](#tests)

More detail: [docs/METHODS.md](docs/METHODS.md) (each method and ablation),
[docs/DATASETS.md](docs/DATASETS.md) (detection rules and the dataset file),
[docs/LIBRARY.md](docs/LIBRARY.md) (the underlying Python library).

## Install

You need Python 3.11 or 3.12.

```bash
git clone https://github.com/gargi-kale/multitask-forked.git
cd multitask-forked
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `multitask` command and everything it needs (TensorFlow 2.19,
Keras 2 through `tf-keras`, HuggingFace `transformers`, and a CPU build of
PyTorch, which is only used to load the small-BERT checkpoint).

**GPU.** On Linux with an NVIDIA GPU, also run
`pip install "tensorflow[and-cuda]>=2.19,<2.20"`. TensorFlow then uses the GPU
automatically. On Apple Silicon, `pip install tensorflow-metal==1.2.0`. Training
also runs on a CPU, which is fine for small datasets and quick checks; for
datasets of 100,000+ texts, use a GPU.

**Keras 2.** HuggingFace's TensorFlow encoders only train under Keras 2. The
package sets `TF_USE_LEGACY_KERAS=1` itself when it is imported, so you don't
need to. (Under Keras 3 the encoder's weights silently never update.)

The first run downloads the encoder from HuggingFace (small BERT is about 20 MB).

## Quick start

**1. Put your data in a folder** as three files whose names contain `train`,
`val` (or `valid` / `dev`) and `test`:

```
data/my_data/my_data_train.csv
data/my_data/my_data_val.csv
data/my_data/my_data_test.csv
```

If you have one file, `multitask prepare` cleans and splits it (see
[below](#cleaning-and-splitting-data)). If a folder holds files for several
datasets, point `--data` at one of the files instead, e.g.
`--data data/all/my_data_train.csv`.

**2. Check what will be trained:**

```bash
multitask detect --data data/my_data
```

This prints every column and what it will be used for (text, source, a task,
or not used and why), the task groups, and the outputs that will be scored.

**3. Train:**

```bash
multitask train --data data/my_data                        # famo_consf1_m5, seed 42
multitask train --data data/my_data --method plain --seeds 42,1,2
multitask train --data data/my_data --method single_task   # one model per label
```

**4. Summarise:**

```bash
multitask report runs/my_data
multitask report runs/my_data --csv scores.csv
```

Runs that already have a result are skipped, so you can rerun a command after
an interruption, or add seeds or methods to the same folder. Use `--dry-run`
to see the plan without training, and `--overwrite` to retrain.

## What your data needs

- **A text column.** It is found automatically (the column with the longest
  strings), or you can name it with `--text-column`.
- **Yes/no label columns.** A column is a task when its values are 0/1,
  yes/no or true/false. A row can be missing some labels (a blank cell). In a
  0/1 column, `-1` also means "no label". Rows that have only some labels are
  normal: each task learns from the rows that have it.
- **Optionally, a source column**: which corpus each row came from, in a
  column called `source`, `dataset`, `corpus`, `domain`, `origin` or `subset`
  (or name it with `--source-column`). Tasks are then grouped by the source
  that labels them, and some ablations use it.

Scores, ratings and categories are listed as **not used**, with a hint. To
use one, turn it into a yes/no label with a cut-off:

```bash
--binarize "violence>=2"                 # one column
--binarize "sentiment,respect,insult>2"  # several columns, same cut-off
--binarize "label==toxic"                # a text category
```

Other options for fixing what was detected:

| Option | Effect |
|---|---|
| `--tasks LIST` | train only these (names, globs like `target_*`, `group:NAME`) |
| `--ignore LIST` | leave these out |
| `--groups single` | put all tasks in one group (default: one group per source) |
| `--pool NAME=TASKS` | add a merged output: "any of these tasks is yes" (max of their scores) |

**Task groups** decide the model's layout: tasks in a group share a branch of
layers, and the fixed loss weights are set per group. **A merged output**
scores several related tasks as one (for example 18 propaganda techniques
scored as "any propaganda"). The tasks keep their own heads.

**The dataset file.** Each run saves the description it used as
`runs/<dataset>/<run>/dataset.yaml`. To change the task groups, the merged
output, the scored outputs or the display names, copy that file, edit it, and
pass it back with `--data path/to/dataset.yaml`. `multitask detect --save
my_data.yaml` writes one without training. See [docs/DATASETS.md](docs/DATASETS.md).

## Methods

Pick one with `--method`. All share the same model (encoder, then shared
layers, then a branch per task group, then a head per task) and the same
training rules (below).

| Method | What it does |
|---|---|
| `famo_consf1_m5` (default) | FAMO task weighting, plus consistency: a copy of each text with 5% of its words masked must give the same answers as the full text, on the 10% most confident cells per side of each output |
| `famo` | FAMO: re-weights the tasks every step so all of them keep improving |
| `dbmtl` | DB-MTL: scales each task's loss by its log and balances gradient sizes |
| `famo_mlm` | FAMO plus 0.03 x a masked-word loss on the encoder |
| `plain` | fixed task weights: each group weighted by 1 / its number of labelled rows |
| `single_task` | one plain model per output, trained on the rows that output labels (the baseline every multi-task result should be compared with) |

**Training rules (all methods):** learning rate 1e-4 with AdamW (weight decay
0.01, 5% warmup), batches of 256, texts up to 512 word pieces, up to 10
epochs. After each epoch the model is scored on validation. The epoch with the
best mean validation F1 is kept, and training stops after 3 epochs without
improvement. Each output's yes/no cut-off is tuned on validation (Youden's J)
and applied to test. Scores are weighted F1 and AUC.

**Add-ons**, usable with any method:

| Option | Effect |
|---|---|
| `--stage2` | frozen stage 2: after training, freeze the shared part and re-train each output's own layers, each keeping its own best epoch |
| `--encoder NAME` | `small-bert` (default), `electra-small`, `electra-small-L2/L4/L6` (first 2/4/6 layers), `distilbert`, `openai-small` / `openai-large` (stored OpenAI embeddings; see `multitask embed`) |
| `--arch NAME` | layer sizes: `wide` (default), `paper`, `medium`, `legacy`, or custom `SHARED/GROUP/TASK` widths, e.g. `512,256/128,64/64,32` |
| `--fast-batches 8` | put texts of similar length in the same batch; about 1.7x faster, same scores |
| `--max-length`, `--batch-size`, `--epochs`, `--lr`, `--patience`, `--stop-metric` | the training rules above |
| `--set KEY=VALUE` | any other setting, e.g. `--set consistency_keep=0.2` (all settings: `multitask/train/methods.py`) |

## Ablations

Each can be combined with any method. One command can launch several runs
(one per share, per label or per seed). `multitask report` lists them side by
side.

| Option | What it tests |
|---|---|
| `--hide-label OUTPUT` | hide one output's labels from every training row; its head is still scored. Add `--probes hidden` to fit a linear probe that reads the hidden label from the trained model |
| `--hide-label all` | a reference run with every label, then one run per output with that output hidden, with probes (the leave-one-label-out study) |
| `--hide-share 0.5` | hide a random share of every label's training cells (`0.25,0.5,0.9` for several runs); also scores the model on the hidden cells |
| `--label-budget TASK=0.05,0.1,0.25` | keep only this share of one task's training labels (label-scarcity curve; compare with `--method single_task`) |
| `--hide-source SOURCE[:TASKS]` | remove these tasks' labels on one source's rows in every split, e.g. labels made by a rule rather than annotators |
| `--keep-sources A,B` | train and score on these sources' rows only |
| `--unweighted-loss` | no task or class weights: a plain mean loss over labelled cells |
| `--no-grouping` | weight each output on its own instead of per task group |
| `--arch ...` | depth and width of the shared and group layers |
| `--probes all --probe-untrained` | control for the probes: the same probes on the untrained model |

Example: the leave-one-label-out study with three seeds:

```bash
multitask train --data data/my_data --method plain --hide-label all --seeds 42,1,2
```

## Several datasets in one model

Give `--data` more than once to train the datasets together in one model. Each
dataset's rows have no labels for the other dataset's tasks, and each row's
source is its dataset (or its own source column). Task names must not clash.

```bash
multitask train --data iac=/path/to/iac --data hatespeech=/path/to/hs \
    --keep-sources IAC,hatespeech --tasks "group:discourse,group:hatespeech"
```

## Cleaning and splitting data

```bash
multitask prepare --input raw.csv --out data/my_data --style cmp \
    --stratify "platform,hatespeech>=1" --dedupe-keep-max n_annotators
```

- `--style cmp`: plain ASCII, links and `@handles` removed, `@`/`*` inside
  words deleted (`f@ggot` becomes `fggot`), apostrophes deleted (`don't`
  becomes `dont`), other punctuation becomes a space. This matches the
  cleaned CMP and hate speech data. The default `--style light` only
  collapses spaces. Single steps: `--clean ascii,urls,handles,punct,lower`;
  placeholder words: `--remove-token URL`.
- Duplicate texts (after cleaning) are kept once, so a text can't appear in
  two splits. `--dedupe-keep-max COLUMN` keeps the copy with the highest value
  in that column.
- `--split 0.8,0.1,0.1 --seed 42`. `--stratify` keeps the shares of these
  columns (or cut-offs) equal across splits.
- Input can be CSV, TSV, Excel or JSON lines. A folder that already has
  train/val/test files is cleaned without re-splitting.

It writes the three files and `prepare_report.txt`, which includes what
`multitask train` will detect in them.

## Reproducing the paper's setup

The paper's data (IAC + Propaganda + IBM; files
`no_prop_overlap_{train,val,test}.csv`) and the Measuring Hate Speech data
come with ready-made dataset files in [datasets/](datasets/):

```bash
multitask train --data iac=/path/to/folder --method famo_consf1_m5 --seeds 42,1,2
multitask train --data iac=/path/to/folder --method single_task --seeds 42,1,2
multitask report runs/iac          # includes the paper's Table 3 F1 as a column
```

The `iac` file reproduces the paper's ten outputs: the 18 propaganda
techniques merged into one Propaganda output, Argument Quality, and the 8 IAC
discourse labels, in three task groups. The Propaganda rows in this data also
carry IAC labels made by a rule from the technique tags, not by annotators.
To train and score without them:

```bash
multitask train --data iac=/path/to/folder --hide-source Propaganda:group:discourse
```

## What a run writes

```
runs/<dataset>/
  <run>/result.json      settings, per-epoch scores, val/test F1 and AUC per output,
                         stage 2, probes, hidden-cell scores, class weights
  <run>/predictions.npz  val/test scores and labels per output (row i = row i of the file)
  <run>/dataset.yaml     the dataset description the run used
  .prepared/             the labels in the standard layout (reused between runs)
```

Run folders are named after the method, the non-default options and the seed,
e.g. `famo_consf1_m5_stage2_s42` or `plain_hide-violence_s1`.

## Limits

- Only yes/no tasks are trained. Multi-class labels and scores need a cut-off
  (`--binarize`). The underlying library supports other task types
  ([docs/LIBRARY.md](docs/LIBRARY.md)), but the methods here are built for yes/no.
- One merged output per model.
- `openai-*` encoders read stored embeddings. `multitask embed --data ...
  --embedding-cache DIR` shows the token count and list-price cost, and only
  calls the API with `--yes` (needs `OPENAI_API_KEY`). Methods that mask words
  (`famo_consf1_m5`, `famo_mlm`) and probes need a text encoder.

## Tests

```bash
python -m pytest tests/test_toolkit_dataset.py -o addopts=   # seconds
python -m pytest tests/test_toolkit_train.py -o addopts=     # ~5 min on a CPU: every method and ablation
```

## Repository layout

```
multitask/              the library (models, trainers, encoders)
multitask/train/        the toolkit: data, methods, runner, report
multitask/cli.py        the `multitask` command
multitask/prepare.py    cleaning and splitting
datasets/               ready-made dataset files (iac, hatespeech)
replication/            the research scripts and write-ups behind the methods
hatespeech/             the hate speech research scripts
```

## Citation

```bibtex
@inproceedings{Farzammultitask,
  title = {Multi-{{Task Learning Improves Performance}} in {{Deep Argument Mining Models}}},
  author = {Farzam, Amirhossein and Shekhar, Shashank and Mehlhaff, Isaac D. and Morucci, Marco},
  year = {2024},
  month = aug,
  booktitle = {Proceedings of the 11th Workshop on Argument Mining},
  publisher = {Association for Computational Linguistics},
  address = {Bangkok, Thailand},
  pages = {46-58},
}
```

See [LICENSE](LICENSE).
