# Methods and ablations

What each `--method` and ablation option does, in more detail than the
README. The code is in `multitask/train/` (`methods.py` lists every setting
and its default).

## The model

Every method trains the same model:

1. **Encoder**: a pretrained transformer (small BERT by default) that turns
   a text into a vector. It is fine-tuned, not frozen.
2. **Shared layers**: dense layers every task uses.
3. **Group branches**: one stack of dense layers per task group.
4. **Heads**: one yes/no output per task. A merged output (`pool`) combines
   several heads' scores (max by default) and is scored as one output.

Layer sizes come from `--arch`. The default `wide` is 256, 128 shared, then
128, 64 per group and 64, 32 per task.

## The loss

For each task, binary cross-entropy is computed on the rows that have its
label, with class weights `N / (2 x count of class)` from the training file,
so rare "yes" labels count more. A merged output is trained on its pooled
score, with the pooled label ("any of its tasks is yes").

How the task losses are combined is what separates the methods.

## Methods

### `plain`: fixed task weights
Each task group gets weight `1 / (rows with a label in the group)`, the
weights are scaled to average 1, and a group's weight is split equally among
its tasks. Tasks with many labelled rows don't dominate just because they
have more rows.

### `famo`: FAMO
FAMO (Fast Adaptive Multitask Optimization; Liu et al., NeurIPS 2023)
replaces the fixed weights. After every step it compares how much each task's
loss fell and shifts weight toward the tasks that are improving least, so
every task keeps making progress. A merged output counts as one task.

### `dbmtl`: DB-MTL
DB-MTL (Lin et al., 2023) takes the log of each task's loss so that tasks on
different scales are comparable, then rescales each task's gradient on the
shared layers to the same size.

### `famo_mlm`: FAMO + masked-word loss
FAMO, plus 0.03 x a masked-word loss: 15% of word pieces are hidden and the
encoder must guess them, which keeps it close to its pretraining.

### `famo_consf1_m5`: FAMO + consistency (the default)
FAMO, plus a consistency term. Each batch also goes through the model with 5%
of its word pieces masked, and the masked copy's answers are pushed toward the
full text's answers (binary KL divergence; the full text's answers are held
fixed). This term also uses **unlabelled** cells, so a task learns something
from rows that only have other tasks' labels. To avoid copying noise, only the
10% most confident cells per side (lowest and highest scores) of each output
in each batch are used. The weight is 1, rising linearly from 0 over the first
epoch. No masked-word loss is added.

A dataset file can keep some rows out of an output's consistency targets
(`consistency_skip`). The bundled IAC file uses this for the Propaganda rows,
whose IAC labels come from a rule.

### `single_task`: one model per output
One `plain` model per scored output, each with only that output's head (or a
merged output's heads), trained on the rows that have its label. This is the
baseline: multi-task learning helps only if it beats this.

## Training rules (every method)

| Setting | Default | Option |
|---|---|---|
| optimizer | AdamW, weight decay 0.01, 5% linear warmup | `--set weight_decay=...` |
| learning rate | 1e-4 | `--lr` |
| batch size | 256 | `--batch-size` |
| text length | 512 word pieces | `--max-length` |
| epochs | up to 10 | `--epochs` |
| dropout | 0 | `--set dropout=0.1` |
| epoch kept | best mean validation F1 over the scored outputs | `--stop-metric f1/auc/loss` |
| stopping | after 3 epochs without improvement | `--patience` |
| cut-off | per output, Youden's J on validation, applied to test | |
| scores | weighted F1 and AUC per output, and their mean | |

## Add-ons

### `--stage2`: frozen stage 2
After training, the encoder, shared layers and group branches are frozen.
Their outputs are computed once per text, and then only each output's own
layers keep training, on its own labels. With nothing shared left to train,
each output keeps the epoch that is best for its own validation F1, so no
output has to accept the epoch that suits the average. It tries the learning
rates in `stage2_lrs` (default 1e-4 and 3e-5, 10 epochs each) and reports the
one with the best validation F1. The result is still one model.

### `--encoder`
`small-bert` (2 layers, 128 wide, 4.4M parameters), `electra-small` (12
layers, 256 wide, 13.5M), `electra-small-L2/L4/L6` (ELECTRA-small cut to its
first 2/4/6 layers, which keep their pretrained weights), `distilbert`.
`openai-small` / `openai-large` read OpenAI embeddings stored by `multitask
embed`. The embedding model is frozen, so only the dense layers train.

### `--fast-batches N`
Texts of similar length go into the same batch (within chunks of N batches,
keeping each batch's mix of sources), and each batch is padded only to its
longest text. Same scores, less wasted compute on padding.

### Fine-tuning options (`--set`)
`layer_lr_decay=0.8` (lower learning rates for lower encoder layers),
`clipnorm=1.0`, `no_decay_bias_norm=true`, `evals_per_epoch=4` (score and
keep the best point several times per epoch).

## Ablations

### `--hide-label OUTPUT` / `--hide-label all`
The output's labels are removed from every training row. Its head stays in
the model and is scored, but it is left out of epoch selection. It gets no
training signal of its own, so its raw scores are close to chance. The real
question is whether learning the *other* tasks built a representation from
which the hidden label can be read. `--probes hidden` answers it: the trained
model is frozen, and a logistic regression is fit for the hidden label on up
to 20,000 labelled training rows (labels the network never saw), from three
representations: the encoder output, the shared layers' output and the
label's group branch. Its cut-off is tuned on validation; F1 and AUC are on
test. `all` runs a reference with every label (probes on every output) and
one hide run per output.

`--probe-untrained` (with `--probes all`) runs the same probes on the model
before any training: a control for how much is already in the pretrained
encoder.

### `--hide-share P`
In every task, a random share P of the labelled training cells is hidden,
drawn per task from one generator seeded by the run's seed (the same cells
for the same seed). Validation and test keep every label. The run also
scores the model on the hidden training cells (`hidden_cells` in
result.json), which shows how well the model fills in labels it never saw.

### `--label-budget TASK=SHARES`
Only this share of one task's training labels is kept (a seeded order, so a
smaller share is a subset of a larger one). Other tasks keep every label. Run
the same budgets with `--method single_task --tasks TASK` for the comparison.
`--repeat-train N` gives single-task runs on few rows as many steps per
epoch as a full run.

### `--hide-source SOURCE[:TASKS]`
The tasks' labels (default: every task) are removed on that source's rows in
train, validation and test. Use it for labels you don't trust on one source.
`TASKS` takes names, globs or `group:NAME`.

### `--keep-sources`, `--unweighted-loss`, `--no-grouping`, `--arch`
Train on some sources only; drop task and class weights (a plain mean over
labelled cells); weight each output on its own rather than per group; change
the depth and width of the shared and group layers.
