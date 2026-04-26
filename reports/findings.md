# Key Findings

## Data Insights
- The labeled train set is tiny: 1,000 rows with a perfectly balanced target.
- The unlabeled and test files are large: 200,000 and 100,000 rows respectively.
- The raw schema is session-based with nested `mouse_events` and `touch_events`, plus pointer timing and viewport fields.
- The previous environment could read the parquet files with `pyarrow 23.0.1`; the current environment needed the Arrow upgrade before loading succeeded.
- Train vs unlabeled/test distribution shift is real and strong, especially for touch usage and event-count features.
- The strongest stable signals seen so far are mouse-path dynamics, timing deltas, and hover behavior.

## Model Performance
- Tiny baseline notebook: LogisticRegression on 16 session features, CV pAUC@0.035 = 0.5879.
- Rich behavior features with HistGradientBoosting: CV pAUC@0.035 = 0.6798 on the best validated scratch run.
- Reusable `src/features.py` matrix with HGB currently reproduces around the mid-0.66 range, so the feature set still needs pruning or stronger modeling.

## Conclusions
- The competition is not a plain tabular problem. The event sequences contain the real signal.
- Honest validation matters: feature-rich models are better than the baseline, but not yet near the target of 0.75.
- The next high-value work is to make the feature set both stronger and more stable under the unlabeled/test distribution, not just larger.
