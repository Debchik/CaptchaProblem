# CAPTCHA Behavioural Bot Detection

Feature-engineering project for the [Kaggle CAPTCHA Behavioural Bot Detection competition](https://www.kaggle.com/competitions/captcha-behavioural-bot-detection/leaderboard).

**Result:** 11th place out of 24 participants. The final submission used a pruned `HistGradientBoostingClassifier` on hand-crafted behavioural features; neural sequence models were explored but did not provide a validated improvement.

## Problem

Classify CAPTCHA interaction sessions from browser telemetry. Each row represents one session and contains pointer timestamps, click and hover coordinates, viewport metadata, and nested mouse/touch event streams. The goal is to rank sessions by the probability of the positive class supplied by the competition.

The practical modeling question was: can aggregated behavioural signals distinguish natural pointer movement from scripted or synthetic interaction when there are only 1,000 labelled sessions?

## Metric and validation

The competition target is partial ROC AUC at a constrained false-positive rate:

```python
roc_auc_score(y_true, y_score, max_fpr=0.1)
```

This rewards strong ranking in the low-false-positive region, which is the relevant operating regime for bot detection.

Historic experiments in this repository were tracked with a stricter local proxy, `pAUC@0.035`, using a fixed 5-fold stratified split (`random_state=42`). The two values should not be compared as if they were identical metrics. Future iterations should standardise selection directly on the official `max_fpr=0.1` metric.

## Dataset

| Split | Rows | Label |
| --- | ---: | --- |
| `train.parquet` | 1,000 | Balanced binary `target` |
| `unlabelled.parquet` | 200,000 | No target |
| `test.parquet` | 100,000 | No target |

Important characteristics:

- The labelled set is very small relative to the event complexity.
- Mouse and touch telemetry are nested event lists, not ready-to-model columns.
- Event histories are capped at 100 stored points.
- Train, unlabeled, and test have noticeable distribution shift, especially in touch usage and event counts.
- The high-value signal is behavioural: timing, path geometry, movement regularity, and hover/click consistency.

## Final model

The selected model was `HistGradientBoostingClassifier` trained on a compact engineered feature matrix. It was chosen because it gave the best validated result without the extra variance and compute cost of the neural branch.

Feature families retained in the final approach:

- Session timing: CAPTCHA initialisation to hover/down/up, click duration, and pointer speed.
- Pointer geometry: down/up displacement, hover-to-click distances, and viewport-normalised coordinates.
- Trajectory aggregates for mouse and touch streams: duration, path length, displacement, straightness/tortuosity, step size, speed, turns, repeated positions, and pauses.
- Data-quality and modality signals: missing hover telemetry, invalid viewport, outside-viewport events, mouse/touch shares, and truncation flags.

The final pruning removed low-value or shifted families such as coordinate quantiles (`q10`/`q90`), pause-ratio features, and `hover_to_center`. This recovered the strongest local result while reducing feature noise.

| Model / feature set | Local validation result |
| --- | ---: |
| LogisticRegression on 16 session features | pAUC@0.035 = 0.5879 |
| HGB on rich behavioural features | pAUC@0.035 = 0.6798 |
| HGB on the full reusable feature matrix | pAUC@0.035 = 0.6662 |
| **Pruned HGB final model** | **pAUC@0.035 = 0.6798** |

The submitted predictions are in [`outputs/submission.csv`](outputs/submission.csv).

## Solution flow

```text
Raw parquet sessions
    -> robust parsing of nested mouse/touch events
    -> timing, geometry, path-shape, modality and consistency features
    -> shift audit and feature ablation
    -> 5-fold stratified validation
    -> prune noisy feature families
    -> HistGradientBoosting final fit
    -> submission.csv
```

### What was tried

1. A small logistic-regression baseline established a trustworthy floor.
2. Manual mouse/touch trajectory summaries created the main performance gain.
3. A train-versus-unlabeled shift audit identified unstable proxies, particularly touch usage and some raw positions.
4. A broader reusable feature matrix was easier to reproduce but initially worse, showing that more features were not automatically better.
5. Pruning noisy or shifted features recovered the best cross-validation result and produced the final submission.
6. Template-mining and self-supervised sequence experiments were explored separately. They were not selected because the simpler tree model was the better validated option for this small-labelled-data setting.

## Repository layout

```text
data/                         Competition parquet files
notebooks/
  CAPTCHA_competition_baseline.ipynb  Initial baseline
  competition_main.ipynb              Original inference pipeline
  complex_features.ipynb              Template / SSL exploration
src/
  features.py                   Reusable behavioural feature extraction
  validation.py                 Stratified CV and partial-AUC helpers
reports/
  experiment_log.md             Hypotheses, experiments, and results
  findings.md                   Data and modelling conclusions
docs/
  interview_notes_ru.md         Short Russian interview walkthrough
outputs/submission.csv          Final submitted predictions
```

## Reproduce the feature extraction

```bash
python -m pip install -r requirements.txt
```

```python
import pandas as pd
from src.features import build_features

train = pd.read_parquet("data/train.parquet")
X = build_features(train.drop(columns="target"))
y = train["target"]
```

The final submission file is retained as the competition artifact. The exact fitted HGB configuration from the original run was not saved as a standalone model file, so this repository documents the validated approach and preserves its feature code and experiment trail rather than claiming bit-for-bit model replay.

## Key lessons

- With only 1,000 labels, carefully chosen domain features can beat higher-capacity sequence models.
- Validation has to reflect the false-positive constraint; full AUC alone is not enough here.
- Unlabelled data is useful for shift detection, but weak transductive features should not be trusted without leakage checks.
- Feature pruning was a modelling decision, not cleanup: removing unstable features was as valuable as adding new ones.

For a concise Russian walkthrough suitable for interviews, see [`docs/interview_notes_ru.md`](docs/interview_notes_ru.md).
