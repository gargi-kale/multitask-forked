# multitask repo — status (7 Oct 2026)

## 1. Synthetic smoke tests (no data needed)

Same test list as `modal_common.SYNTHETIC_TESTS`, run on CPU with TF 2.19.1 / Python 3.12.

| Branch | Result |
|---|---|
| `main` | **215 passed, 4 skipped** (the 4 are GPU-only tests), 0 failed |
| `testing` | **310 passed, 7 failed** |

The 7 failures on `testing`:

- **4 tests are out of date with the code:**
  - multilabel thresholds now come back as a list, not a float (3 tests)
  - `dropout_rate` moved off `ModelConfig` (1 test)
- **1 test checks an object type:** the learning rate comes back as a Keras Variable, not a float.
- **1 early-stopping test:** training never stopped early and ran all 30 epochs.
- **1 real issue:** `test_regression_predictions_match_label_scale`. Regression predictions stay near 0 while the labels are around 300. The cause is that the loss divides both the label and the prediction by the label's standard deviation, which shrinks the gradient by std². This matters for loss design.

## 2. Paper-shaped model on simulated data

Script: `simulate_paper_setup.py`. It needs no CSVs and no downloads.

- **Same structure as `run_paper_replication`:**
  - 27 binary tasks (18 propaganda techniques + WA_bi + 8 IAC discourse tasks)
  - hierarchy `[[3],[18,1,8]]`, shared layers `[64,32]`, branch layers `[[32,16],[16,8]]`
  - a max-pool output over the 18 propaganda heads
  - the paper's hyperparameters (AdamW, lr 3e-4, weight decay 0.01, dropout 0.4, batch 256, 5% warmup)
- **Realistic labels:** each row is labelled only for its own source corpus, and everything else is masked. Class imbalance is taken from paper Table 1 and from the class weights logged in the cluster run.
- **Inputs:** 128-d simulated embeddings standing in for small BERT, so it runs on a laptop CPU. The signal is easy to learn: a plain logistic regression gets **mean AUC 0.97**.
- **Split:** 24k train, 3k val, 3k test. In the main script, thresholds are tuned on val and metrics are reported on test.

**What happened** (one seed, 10 epochs, validation AUC by task group):

| Setting | Propaganda | WA_bi | Discourse |
|---|---|---|---|
| Paper settings (main) | 0.53 | 0.88 | 0.54 |
| Paper settings (testing branch loss) | 0.54 | 0.88 | 0.54 |
| Dropout 0.1 instead of 0.4 | **0.76** | **0.97** | **0.85** |
| lr 1e-3 | 0.60 | 0.97 | 0.70 |
| Flat (no hierarchy) | 0.63 | 0.93 | 0.63 |
| No class weights | 0.63 | 0.78 | 0.55 |
| No task-type grouping | 0.55 | 0.66 | 0.52 |

With paper settings, the model barely learns. Most of the gap comes from 40% dropout applied after *every* layer, including the 16- and 8-unit task heads (this is what the paper describes). This is a hypothesis, not a conclusion: the data is simulated and it's one seed. Still, the near-chance AUC matches the cluster run (overall AUC 0.541).

**Also found:** with paper settings and default early stopping, `main` stopped at epoch 4 and restored epoch 1. The loss on `main` is an average (about 0.09), so the absolute `min_delta=0.001` treats real improvements as "no change". The paper also says patience 2, but the library default is 3.

## 3. Evaluation issues worth fixing

- **F1, precision and recall use `average='weighted'`.**
  - On the simulated run, weighted F1 was about 0.60, while macro F1 was 0.40 and positive-class F1 was 0.22.
  - Weighted recall always equals accuracy for binary tasks.
- **Overall AUC becomes `nan`** if any task has only one class in the eval split (for example Bandwagon), because NaN isn't filtered out before averaging.
- **The replication script tunes thresholds on val and reports on that same val set.** The test split is loaded but never used.
- **`run_experiment` ignores `--epochs`** when `PAPER_TRAINING_CONFIG` is passed, so it always uses the default of 20.

## 4. Data

- No CSVs exist in the repo or its history.
- The real dataset combines IAC 2.0 (8 discourse tasks), IBM-Rank-30k (WA) and the Da San Martino et al. 2019 propaganda corpus.
- The Hugging Face item in the code is the small-BERT **encoder** (`google/bert_uncased_L-2_H-128_A-2`), not a dataset.
