# Experiment Log

## Overview
Log of hypothesis-driven experiments for the CAPTCHA competition.

## Leaderboard

| Exp | Hypothesis | Model / Features | CV pAUC@0.035 | Notes |
| --- | --- | --- | --- | --- |
| 1 | Small session-level features are enough for a first trusted baseline. | LogisticRegression on 16 tiny features | 0.5879 | Matches the saved baseline notebook. |
| 2 | Mouse/touch sequence summaries will capture the real signal better than the tiny baseline. | HistGradientBoosting on 122 engineered features | 0.6798 | Best validated result so far. |
| 3 | Some strong train signals are also stable on unlabeled/test-like data, while others are heavily shifted. | EDA / shift audit on train vs unlabeled | n/a | Stable candidates: `hover_ts`, `mouse_speed_max`, `mouse_tortuosity`, `mouse_events_total`. |
| 4 | A broader feature matrix plus stronger regularization may beat the current HGB baseline. | Reusable `src/features.py` matrix, HGB | 0.6662 | Reproducible in the new src pipeline, but weaker than Exp 2. |
| 5 | Pruning shifted and low-value engineered features from the reusable matrix will recover most of the scratch-model gain. | Pruned `src/features.py` matrix, HGB | 0.6798 | Final submission model exported to `outputs/submission.csv`. |

## Experiments

### Experiment 1
- Date: 2026-04-23
- Hypothesis: a small number of session-level fields is enough to establish a trustworthy baseline.
- Implementation summary: extracted the tiny feature set from the baseline notebook and fit LogisticRegression with stratified 5-fold CV using standardized partial AUC.
- Validation result: CV pAUC@0.035 = **0.5879**.
- Conclusion: the baseline is valid but weak; it confirms the task needs richer behavior features.
- Next action: expand features around mouse/touch dynamics and timing interactions.

### Experiment 2
- Date: 2026-04-23
- Hypothesis: explicit mouse-path and touch summaries will improve ranking quality more than linear modeling alone.
- Implementation summary: engineered sequence statistics from `mouse_events` and `touch_events`, plus timing, geometry, and path-shape features; evaluated with HistGradientBoosting under the same 5-fold split.
- Validation result: CV pAUC@0.035 = **0.6798**.
- Conclusion: this is a meaningful uplift over the tiny baseline and the strongest validated result so far.
- Next action: reduce feature shift, test unlabeled-aware clustering, and compare against a few carefully tuned tree models.

### Experiment 3
- Date: 2026-04-23
- Hypothesis: the unlabeled set reflects the test distribution and can reveal which features are stable enough to keep.
- Implementation summary: compared engineered feature distributions between train and a sample of unlabeled rows.
- Validation result: no score, but the shift audit showed large movement in `has_touch`, event counts, and some position features.
- Conclusion: train-only gains are not enough; the next iteration should favor stable mouse-dynamics features and avoid over-weighting shifted touch-count proxies.
- Next action: test cluster-based or density-based features learned from unlabeled rows, then re-evaluate HGB and XGBoost on the pruned feature set.

### Experiment 4
- Date: 2026-04-23
- Hypothesis: the reusable `src/features.py` matrix should still support a competitive tree model, but some added feature families may be noisy.
- Implementation summary: built the full reusable engineered matrix and evaluated HistGradientBoosting with the same 5-fold pAUC split.
- Validation result: CV pAUC@0.035 = **0.6662**.
- Conclusion: the reusable pipeline worked, but the extra quantile/pause-style features were likely too noisy.
- Next action: prune the low-value columns and retune HGB on the cleaner matrix.

### Experiment 5
- Date: 2026-04-23
- Hypothesis: removing the shifted / noisy engineered columns will recover the strongest scratch-model behavior on the reusable matrix.
- Implementation summary: dropped the `q10`, `q90`, `pause_`, and `hover_to_center` features from the reusable matrix and tuned HistGradientBoosting to a slightly deeper configuration.
- Validation result: CV pAUC@0.035 = **0.6798**.
- Conclusion: this is the final submission model, and it matches the best validated score we have in the repo.
- Next action: submit `outputs/submission.csv` to Kaggle and iterate on unlabeled-derived features if more lift is needed.
