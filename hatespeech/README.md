# Measuring Hate Speech data: random-masking experiment

This folder is kept apart from the main model code. Nothing in `multitask/` or `replication/` was changed for it, and nothing there imports this folder.

`run_hatespeech.py` *imports* (read-only) the model, loss, data loader and per-epoch scorer from `replication/test_no_prop_overlap.py`, so training matches the main re-tuned runs exactly:
- small BERT, inputs up to 512 word pieces;
- the "wide" layer sizes;
- learning rate 1e-4, weight decay 0.01, 5% warmup, dropout 0, batch 256;
- up to 10 epochs, keeping the epoch with the best validation F1 and stopping after 3 epochs without improvement;
- yes/no cut-offs tuned on validation; scored with weighted F1.

## Running the masking experiment

Run these from `multitask-forked/`:

    modal run hatespeech/modal_hatespeech.py::upload_data      # once
    modal run hatespeech/modal_hatespeech.py::runs --seeds 42,1,2 --shares 0,0.25,0.5,0.75
    modal run hatespeech/modal_hatespeech.py::report

- `--mask-share p` hides a random share p of each item's training labels.
- The hidden cells depend only on the seed. A masked run and the unmasked run with the same seed can therefore be compared on the same cells.
- Results are written to `multitask-outputs:/hatespeech_mask/`.

**Purpose:** test whether a semi-supervised loss helps on data where *every* label is known. The plan is to hide a random share of training labels, then score the model on those hidden cells as well as on test.

## Files

- `raw/hatespeech_dlab.csv`: a copy of `multitask-data:/raw/hatespeech_dlab.csv` on Modal. It has 39,565 comments, one row per comment, and each item score is the average over that comment's annotators.
- `prepare_hatespeech.py`: cleans the text and splits it into train, validation and test (run from `multitask-forked/`).
- `data/hatespeech_{train,val,test}.csv`: the output. The 10 survey items are kept as continuous scores. The yes/no cut-off has not been applied yet.
- `data/prepare_report.txt`: row counts, the "yes" share at each cut-off, label correlations, and before/after cleaning examples.

## Cleaning (matches the CMP style)

- Plain ASCII, so emoji, zero-width characters and accents are removed or folded.
- Case kept.
- Apostrophes deleted ("don't" becomes "dont").
- @handles at the start of a word removed.
- An @ or * inside a word deleted ("f@ggot" becomes "fggot").
- The dataset's "URL" placeholder removed.
- All other punctuation becomes a space, then repeated spaces are collapsed.

The cleaning dropped 1 comment that was empty afterwards. It also found 74 texts that were identical after cleaning. Only one copy of each was kept, the one with the most annotators. 39,490 rows remain.

## Split

80/10/10 with seed 42, stratified by platform and by hate speech yes/no (hatespeech ≥ 1):

| Split | Rows |
|---|---|
| train | 31,592 |
| val | 3,949 |
| test | 3,949 |

No comment ID or text appears in more than one split.

## Items

Higher means more hateful on every item.

- `hatespeech` is on a 0–2 scale (no / unclear / yes).
- The other nine items are on a 0–4 scale: sentiment, respect, insult, humiliate, status, dehumanize, violence, genocide, attack_defend.

The yes/no cut-off is still to be chosen; see `data/prepare_report.txt`.
