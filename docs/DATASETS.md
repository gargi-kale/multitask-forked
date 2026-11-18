# Datasets: detection and the dataset file

## How columns are detected

`multitask detect --data FOLDER` (and `train`) read the train file and decide,
column by column:

| Column looks like | Used as |
|---|---|
| the longest strings (15+ characters on average) | the **text** |
| a text column named source / dataset / corpus / domain / origin / subset, with up to 100 values and no blanks | the **source** |
| only 0 and 1 (blanks allowed) | a yes/no **task** |
| only -1, 0 and 1 | a yes/no task, with -1 = no label |
| yes/no, true/false or y/n text | a yes/no task |
| a different whole number on every row | not used (an ID) |
| a few whole numbers (classes or a rating) | not used: add a cut-off with `--binarize` |
| any other numbers (a score) | not used: add a cut-off with `--binarize` |
| other text with a few values (categories) | not used: `--binarize "col==value"` |
| only one value, or one class in train | not used |

Blank cells, and the strings `NA`, `N/A`, `NaN`, `null`, `None`, mean "no
label". Task names are the column names, with characters a layer name can't
hold replaced by `_`.

**Task groups.** If there is a source column with more than one value, each
task joins the group of the source that labels most of its rows (on the
paper's data this finds the three groups Propaganda, IAC and IBM).
Otherwise, or with `--groups single`, all tasks form one group.

**Files.** In the folder, the files whose names contain `train`, `val` /
`valid` / `dev` and `test` (CSV or TSV). If a folder has several sets, pass one
file: `--data data/all/name_train.csv` uses the files starting with `name_`.

## The dataset file

A YAML file describing the same things. Every run saves one; `multitask
detect --save FILE` writes one; `datasets/iac/dataset.yaml` is a full example.

```yaml
name: my_data
data_dir: .                  # folder with the files, relative to this file
files:
  train: my_data_train.csv
  val: my_data_val.csv
  test: my_data_test.csv
text_column: text
source_column: dataset       # optional
tasks:
- name: toxic                # output name
  column: is_toxic           # column in the files (default: the name)
  group: harm                # task group
- name: violence
  group: harm
  binarize: '>=2'            # yes/no by a cut-off ('>=', '>', '<=', '<', '==', '!=')
- name: rated_flag
  group: other
  missing: [-1]              # values that mean "no label"
pools:                       # optional, at most one: a merged output
- name: any_harm
  tasks: [toxic, violence]
  strategy: max              # max | noisy_or | lse
outputs: [any_harm, rated_flag]      # optional: scored outputs and their order
consistency_skip:            # optional (famo_consf1_m5): keep rows labelled for
  group:other: pool:any_harm #   these tasks out of those outputs' targets
reference:                   # optional: published scores, shown by `report`
  title: Paper Table 3
  f1: {any_harm: 61.7, rated_flag: 70.1}
labels: {any_harm: Any harm}  # optional display names
```

Pass it with `--data my_data.yaml`. `--tasks` and `--ignore` still work on
top of it; `--binarize`, `--text-column` and `--source-column` only apply to
detected folders (edit the file instead).

## Bundled datasets

`--data iac` and `--data hatespeech` use the files in `datasets/`. The data
itself isn't in the repository. Put the CSVs in that folder, or give their
location: `--data iac=/path/to/folder`.

- **iac**: `no_prop_overlap_{train,val,test}.csv`, the paper's IAC +
  Propaganda + IBM data, with its ten outputs and the paper's Table 3 F1 as
  the reference column.
- **hatespeech**: `hatespeech_{train,val,test}.csv` from
  `hatespeech/prepare_hatespeech.py`. The 10 survey items become yes/no tasks:
  > 2 for the 0-4 items, >= 1 for `hatespeech`.

## The standard layout

Before training, the labels are written once to
`runs/<dataset>/.prepared/<id>/{train,val,test}.csv` with the columns `text`,
`dataset` (each row's source) and one column per task holding 0, 1 or blank.
The trainer reads only these. `<id>` changes when the dataset file, the data
files, `--keep-sources` or `--hide-source` change, so the files are rebuilt
only when needed.
