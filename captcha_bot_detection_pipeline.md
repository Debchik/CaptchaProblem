# CAPTCHA Bot Detection Pipeline

This document collects the full feature-engineering and ML pipeline that was described for producing bot-probability predictions from CAPTCHA-solving logs.

The goal is to predict:

```text
probability = P(row was produced by a bot)
```

The expected final output format is:

```csv
id,probability
0,0.123456
1,0.987654
...
```

## Important reproducibility note

The original run did not save a separate fitted model artifact or an exact executable training script. This document is therefore a reproducible reconstruction of the pipeline that was described:

- same raw columns,
- same engineered feature families,
- same model families,
- same validation logic,
- same calibration method,
- same test-output logic.

Because the previous fitted estimators were not persisted, rerunning this pipeline should reproduce the approach, but exact probabilities may differ slightly depending on library versions, CPU parallelism, and implementation details. Welcome to ML, where determinism goes to retire.

---

# 1. Input files

Expected files:

```text
train: /mnt/data/train.parquet
test:  /mnt/data/test.parquet
out:   /mnt/data/predictions.csv
```

Training data contains a label column:

```text
target
```

Assumption:

```text
target = 1 -> bot
target = 0 -> human
```

The test file has no explicit `id` column in the described run, so the test row index was used as the output id:

```text
id = 0, 1, 2, ..., len(test) - 1
```

---

# 2. Raw dataset columns

The columns observed / used in the pipeline were:

```text
target
relative_captcha_init_time
mouse_events_total
mouse_events
touch_events_total
touch_events
pointerdown_timestamp
pointerdown_x
pointerdown_y
pointerup_timestamp
pointerup_x
pointerup_y
hover_timestamp
hover_x
hover_y
viewport_width
viewport_height
```

In test, `target` is absent.

Nested columns:

```text
mouse_events = list of events with fields:
- timestamp_
- x_
- y_
```

```text
touch_events = list of events with fields:
- force_
- radiusX_
- radiusY_
- rotationAngle_
- timestamp_
- x_
- y_
```

The described model used only the timestamp and coordinate fields from `touch_events`:

```text
timestamp_, x_, y_
```

---

# 3. Complete feature list

The pipeline generated 126 candidate features:

- 14 raw scalar features kept from the original table,
- 112 engineered features.

Constant or fully missing columns were dropped before model training, leaving about 124 usable model features in the described run.

## 3.1 Raw features kept

```text
relative_captcha_init_time
mouse_events_total
touch_events_total
pointerdown_timestamp
pointerdown_x
pointerdown_y
pointerup_timestamp
pointerup_x
pointerup_y
hover_timestamp
hover_x
hover_y
viewport_width
viewport_height
```

## 3.2 Timing features

```text
click_duration
time_init_down
time_init_up
time_hover_down
time_hover_up
```

## 3.3 Pointer geometry features

```text
pointer_dx
pointer_dy
pointer_dist
click_speed
hover_down_dist
hover_up_dist
```

## 3.4 Hover / missingness feature

```text
hover_zero
```

## 3.5 Viewport features

```text
viewport_area
viewport_aspect
viewport_invalid
vp_w_log
vp_h_log
vp_area_log
mobile_like
desktop_like
wide_screen
```

## 3.6 Normalized coordinate features

```text
pd_x_norm
pd_y_norm
pu_x_norm
pu_y_norm
h_x_norm
h_y_norm
```

## 3.7 Outside-viewport flags

```text
pd_outside
pu_outside
h_outside
```

## 3.8 Mouse/touch count features

```text
events_sum
mouse_touch_ratio
has_mouse
has_touch
only_mouse
only_touch
mouse_total_minus_len
touch_total_minus_len
mouse_truncated
touch_truncated
```

## 3.9 Mouse trajectory aggregate features

```text
mouse_len
mouse_dur
mouse_x_rng
mouse_y_rng
mouse_x_std
mouse_y_std
mouse_unique_ratio
mouse_x_first
mouse_y_first
mouse_x_last
mouse_y_last
mouse_t_first
mouse_t_last
mouse_path
mouse_disp
mouse_straight
mouse_dt_mean
mouse_dt_std
mouse_dt_min
mouse_dt_max
mouse_step_mean
mouse_step_std
mouse_step_max
mouse_speed_mean
mouse_speed_std
mouse_speed_max
mouse_stationary_ratio
mouse_last_to_down
mouse_last_to_up
mouse_last_to_hover
mouse_first_to_down
mouse_first_to_up
```

## 3.10 Touch trajectory aggregate features

```text
touch_len
touch_dur
touch_x_rng
touch_y_rng
touch_x_std
touch_y_std
touch_unique_ratio
touch_x_first
touch_y_first
touch_x_last
touch_y_last
touch_t_first
touch_t_last
touch_path
touch_disp
touch_straight
touch_dt_mean
touch_dt_std
touch_dt_min
touch_dt_max
touch_step_mean
touch_step_std
touch_step_max
touch_speed_mean
touch_speed_std
touch_speed_max
touch_stationary_ratio
touch_last_to_down
touch_last_to_up
touch_last_to_hover
touch_first_to_down
touch_first_to_up
```

## 3.11 Frequency and smoothed target-encoding features

```text
te_vp
freq_vp
te_counts
freq_counts
te_vp_counts
freq_vp_counts
te_vp_hover
freq_vp_hover
```

---

# 4. Feature formulas and explanations

## 4.1 Raw scalar features

These are copied directly from the input table:

```python
relative_captcha_init_time
mouse_events_total
touch_events_total
pointerdown_timestamp
pointerdown_x
pointerdown_y
pointerup_timestamp
pointerup_x
pointerup_y
hover_timestamp
hover_x
hover_y
viewport_width
viewport_height
```

These preserve direct timing, coordinate, event-count, and screen-size information.

---

## 4.2 Timing features

```python
click_duration = pointerup_timestamp - pointerdown_timestamp
```

Measures how long the pointer was held down.

```python
time_init_down = pointerdown_timestamp - relative_captcha_init_time
```

Measures time from CAPTCHA initialization to pointer down.

```python
time_init_up = pointerup_timestamp - relative_captcha_init_time
```

Measures time from CAPTCHA initialization to pointer up.

```python
time_hover_down = pointerdown_timestamp - hover_timestamp
```

Measures time between hover and pointer down.

```python
time_hover_up = pointerup_timestamp - hover_timestamp
```

Measures time between hover and pointer up.

Suspicious bot behavior may include extremely low values, impossible ordering, or repeated identical values.

---

## 4.3 Pointer geometry features

```python
pointer_dx = pointerup_x - pointerdown_x
pointer_dy = pointerup_y - pointerdown_y
pointer_dist = sqrt(pointer_dx ** 2 + pointer_dy ** 2)
```

These measure movement between pointer down and pointer up.

```python
click_speed = pointer_dist / click_duration
```

Approximate pointer movement speed during the click. In implementation, division by zero is guarded.

```python
hover_down_dist = sqrt((pointerdown_x - hover_x) ** 2 + (pointerdown_y - hover_y) ** 2)
hover_up_dist   = sqrt((pointerup_x   - hover_x) ** 2 + (pointerup_y   - hover_y) ** 2)
```

These measure geometric consistency between hover and click/tap location.

---

## 4.4 Hover missingness

```python
hover_zero = int(hover_timestamp == 0 and hover_x == 0 and hover_y == 0)
```

This catches missing, defaulted, or synthetic hover telemetry.

---

## 4.5 Viewport features

```python
viewport_area = viewport_width * viewport_height
viewport_aspect = viewport_width / viewport_height
viewport_invalid = int(viewport_width <= 0 or viewport_height <= 0)
```

Viewport size can act as a weak device / environment fingerprint.

```python
vp_w_log = log1p(viewport_width)
vp_h_log = log1p(viewport_height)
vp_area_log = log1p(viewport_area)
```

Log transforms reduce scale dominance.

```python
mobile_like  = int(0 < viewport_width <= 600)
desktop_like = int(viewport_width >= 1200 and viewport_height >= 700)
wide_screen  = int(viewport_width >= 1800)
```

Coarse screen-type flags.

---

## 4.6 Normalized coordinate features

```python
pd_x_norm = pointerdown_x / viewport_width
pd_y_norm = pointerdown_y / viewport_height
pu_x_norm = pointerup_x / viewport_width
pu_y_norm = pointerup_y / viewport_height
h_x_norm  = hover_x / viewport_width
h_y_norm  = hover_y / viewport_height
```

These make coordinates comparable across different screen sizes.

---

## 4.7 Outside-viewport flags

```python
pd_outside = int(pointerdown_x < 0 or pointerdown_x > viewport_width or
                 pointerdown_y < 0 or pointerdown_y > viewport_height)

pu_outside = int(pointerup_x < 0 or pointerup_x > viewport_width or
                 pointerup_y < 0 or pointerup_y > viewport_height)

h_outside = int(hover_x < 0 or hover_x > viewport_width or
                hover_y < 0 or hover_y > viewport_height)
```

These identify impossible or suspicious coordinate values.

---

## 4.8 Mouse/touch count features

```python
events_sum = mouse_events_total + touch_events_total
mouse_touch_ratio = (mouse_events_total + 1) / (touch_events_total + 1)
```

The ratio is smoothed by `+1` to avoid division by zero.

```python
has_mouse = int(mouse_events_total > 0)
has_touch = int(touch_events_total > 0)
only_mouse = int(mouse_events_total > 0 and touch_events_total == 0)
only_touch = int(touch_events_total > 0 and mouse_events_total == 0)
```

These describe the interaction modality.

```python
mouse_total_minus_len = mouse_events_total - len(mouse_events)
touch_total_minus_len = touch_events_total - len(touch_events)
mouse_truncated = int(mouse_events_total > len(mouse_events))
touch_truncated = int(touch_events_total > len(touch_events))
```

These compare declared event counts with stored event-list lengths.

---

## 4.9 Trajectory features

For both `mouse_events` and `touch_events`, events are reduced to:

```text
(timestamp, x, y)
```

For a sequence:

```python
events = [(t0, x0, y0), (t1, x1, y1), ..., (tn, xn, yn)]
```

Basic features:

```python
prefix_len = len(events)
prefix_dur = last_timestamp - first_timestamp
prefix_x_rng = max(x) - min(x)
prefix_y_rng = max(y) - min(y)
prefix_x_std = std(x)
prefix_y_std = std(y)
prefix_unique_ratio = number_of_unique_points / len(events)
prefix_x_first = first x
prefix_y_first = first y
prefix_x_last = last x
prefix_y_last = last y
prefix_t_first = first timestamp
prefix_t_last = last timestamp
```

Movement features:

```python
dx = diff(x)
dy = diff(y)
dt = diff(t)
step = sqrt(dx ** 2 + dy ** 2)
```

```python
prefix_path = sum(step)
prefix_disp = sqrt((x_last - x_first) ** 2 + (y_last - y_first) ** 2)
prefix_straight = prefix_disp / prefix_path
```

Timing-delta features:

```python
prefix_dt_mean = mean(dt)
prefix_dt_std = std(dt)
prefix_dt_min = min(dt)
prefix_dt_max = max(dt)
```

Step-distance features:

```python
prefix_step_mean = mean(step)
prefix_step_std = std(step)
prefix_step_max = max(step)
```

Speed features:

```python
speed = step / dt
prefix_speed_mean = mean(speed)
prefix_speed_std = std(speed)
prefix_speed_max = max(speed)
```

Stationary feature:

```python
prefix_stationary_ratio = mean(step == 0)
```

Trajectory-to-click consistency:

```python
prefix_last_to_down = distance(last_event_point, pointerdown_point)
prefix_last_to_up = distance(last_event_point, pointerup_point)
prefix_last_to_hover = distance(last_event_point, hover_point)
prefix_first_to_down = distance(first_event_point, pointerdown_point)
prefix_first_to_up = distance(first_event_point, pointerup_point)
```

`prefix` is either:

```text
mouse
```

or:

```text
touch
```

---

## 4.10 Frequency and smoothed target encoding

Four grouping signatures were used:

```python
KEY_GROUPS = {
    "vp": ["viewport_width", "viewport_height"],
    "counts": ["mouse_events_total", "touch_events_total"],
    "vp_counts": ["viewport_width", "viewport_height", "mouse_events_total", "touch_events_total"],
    "vp_hover": ["viewport_width", "viewport_height", "hover_zero"],
}
```

For each group, two features were generated:

```text
freq_<group>
te_<group>
```

Frequency:

```python
freq_group = count of same exact signature in training data
```

Smoothed target encoding:

```python
te = (bot_count_for_group + global_bot_rate * smoothing) / (group_count + smoothing)
```

Used:

```python
smoothing = 8.0
```

During cross-validation, target encodings for validation rows must be computed using only the training folds. Otherwise the model gets to peek at the answer, because apparently cheating is bad even when computers do it.

---

# 5. Model pipeline

## 5.1 Cross-validation

Used stratified 5-fold CV:

```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

This preserves the label balance in each fold.

For every fold:

1. Generate base features.
2. Fit target-encoding maps on the training fold only.
3. Transform training fold and validation fold.
4. Train each model on training fold.
5. Predict probabilities for validation fold.
6. Store out-of-fold predictions.

---

## 5.2 Models

Three sklearn models were used.

### Model 1: Logistic Regression

```python
make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    LogisticRegression(C=0.2, max_iter=3000)
)
```

Purpose:

- simple linear baseline,
- stable on small training data,
- usually decent probability behavior.

### Model 2: ExtraTreesClassifier

```python
make_pipeline(
    SimpleImputer(strategy="median"),
    ExtraTreesClassifier(
        n_estimators=450,
        max_features="sqrt",
        min_samples_leaf=3,
        random_state=1,
        n_jobs=-1
    )
)
```

Purpose:

- nonlinear feature interactions,
- strong performance on tabular behavioral features,
- robust to monotonic and non-monotonic patterns.

### Model 3: RandomForestClassifier

```python
make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=5,
        random_state=2,
        n_jobs=-1
    )
)
```

Purpose:

- second tree-based model with different bias,
- useful for ensembling,
- smoother than ExtraTrees.

---

## 5.3 Ensemble

Each model produced out-of-fold probabilities.

Model weights were computed from validation log loss:

```python
weight_i = 1 / (logloss_i ** 2)
```

Then normalized:

```python
weight_i = weight_i / sum(weights)
```

The described run produced approximately:

```text
ExtraTrees:          0.386
LogisticRegression:  0.377
RandomForest:        0.238
```

Ensemble probability:

```python
raw_ensemble_probability = weighted_average(model_probabilities)
```

---

## 5.4 Calibration

The raw ensemble was calibrated using Platt scaling.

First clip raw probabilities:

```python
p = clip(raw_ensemble_probability, 1e-6, 1 - 1e-6)
```

Convert to logit:

```python
logit_score = log(p / (1 - p))
```

Fit a logistic regression calibrator:

```python
calibrator = LogisticRegression(C=1.0)
calibrator.fit(logit_score.reshape(-1, 1), y)
```

For test predictions:

```python
calibrated_probability = calibrator.predict_proba(test_logit_score.reshape(-1, 1))[:, 1]
```

Final conservative blend:

```python
final_probability = 0.7 * calibrated_probability + 0.3 * raw_ensemble_probability
```

Final clipping:

```python
final_probability = clip(final_probability, 0.001, 0.999)
```

---

# 6. Fully reproducible Python script

Save this as:

```text
train_predict_captcha_bot.py
```

Install dependencies:

```bash
pip install numpy pandas pyarrow scikit-learn
```

Run:

```bash
python train_predict_captcha_bot.py \
  --train /mnt/data/train.parquet \
  --test /mnt/data/test.parquet \
  --out /mnt/data/predictions.csv
```

```python
import argparse
import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

EPS = 1e-9
SMOOTHING = 8.0

KEY_GROUPS = {
    "vp": ["viewport_width", "viewport_height"],
    "counts": ["mouse_events_total", "touch_events_total"],
    "vp_counts": ["viewport_width", "viewport_height", "mouse_events_total", "touch_events_total"],
    "vp_hover": ["viewport_width", "viewport_height", "hover_zero"],
}


def safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > EPS, a / b, np.nan)


def dist_xy(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def is_missing_obj(x):
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def event_get(ev, key, default=np.nan):
    if ev is None:
        return default
    if isinstance(ev, dict):
        return ev.get(key, default)
    try:
        return getattr(ev, key)
    except Exception:
        pass
    return default


def normalize_event_list(obj):
    """Return list-like event container as a Python list.

    Parquet nested columns may come back as list, numpy array, pandas object,
    or None depending on engine/version.
    """
    if obj is None:
        return []
    if isinstance(obj, float) and math.isnan(obj):
        return []
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    try:
        if pd.isna(obj):
            return []
    except Exception:
        pass
    try:
        return list(obj)
    except Exception:
        return []


def extract_sequence_stats(events_col, prefix, df):
    rows = []

    for obj in events_col:
        events = normalize_event_list(obj)
        ts, xs, ys = [], [], []

        for ev in events:
            t = event_get(ev, "timestamp_", np.nan)
            x = event_get(ev, "x_", np.nan)
            y = event_get(ev, "y_", np.nan)
            ts.append(t)
            xs.append(x)
            ys.append(y)

        t = np.asarray(ts, dtype=float)
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)

        valid = ~(np.isnan(t) | np.isnan(x) | np.isnan(y))
        t, x, y = t[valid], x[valid], y[valid]
        n = len(t)

        d = {}
        d[f"{prefix}_len"] = n

        if n == 0:
            base_names = [
                "dur", "x_rng", "y_rng", "x_std", "y_std", "unique_ratio",
                "x_first", "y_first", "x_last", "y_last", "t_first", "t_last",
                "path", "disp", "straight", "dt_mean", "dt_std", "dt_min", "dt_max",
                "step_mean", "step_std", "step_max", "speed_mean", "speed_std",
                "speed_max", "stationary_ratio"
            ]
            for name in base_names:
                d[f"{prefix}_{name}"] = np.nan
            rows.append(d)
            continue

        d[f"{prefix}_dur"] = t[-1] - t[0]
        d[f"{prefix}_x_rng"] = np.nanmax(x) - np.nanmin(x)
        d[f"{prefix}_y_rng"] = np.nanmax(y) - np.nanmin(y)
        d[f"{prefix}_x_std"] = np.nanstd(x)
        d[f"{prefix}_y_std"] = np.nanstd(y)
        d[f"{prefix}_unique_ratio"] = len(set(zip(x, y))) / max(n, 1)
        d[f"{prefix}_x_first"] = x[0]
        d[f"{prefix}_y_first"] = y[0]
        d[f"{prefix}_x_last"] = x[-1]
        d[f"{prefix}_y_last"] = y[-1]
        d[f"{prefix}_t_first"] = t[0]
        d[f"{prefix}_t_last"] = t[-1]

        if n >= 2:
            dx = np.diff(x)
            dy = np.diff(y)
            dt = np.diff(t)
            step = np.sqrt(dx ** 2 + dy ** 2)
            positive_dt = np.where(np.abs(dt) > EPS, dt, np.nan)
            speed = step / positive_dt

            path = np.nansum(step)
            disp = math.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

            d[f"{prefix}_path"] = path
            d[f"{prefix}_disp"] = disp
            d[f"{prefix}_straight"] = disp / path if path > EPS else np.nan
            d[f"{prefix}_dt_mean"] = np.nanmean(dt)
            d[f"{prefix}_dt_std"] = np.nanstd(dt)
            d[f"{prefix}_dt_min"] = np.nanmin(dt)
            d[f"{prefix}_dt_max"] = np.nanmax(dt)
            d[f"{prefix}_step_mean"] = np.nanmean(step)
            d[f"{prefix}_step_std"] = np.nanstd(step)
            d[f"{prefix}_step_max"] = np.nanmax(step)
            d[f"{prefix}_speed_mean"] = np.nanmean(speed)
            d[f"{prefix}_speed_std"] = np.nanstd(speed)
            d[f"{prefix}_speed_max"] = np.nanmax(speed)
            d[f"{prefix}_stationary_ratio"] = np.nanmean(step == 0)
        else:
            for name in [
                "path", "disp", "straight", "dt_mean", "dt_std", "dt_min", "dt_max",
                "step_mean", "step_std", "step_max", "speed_mean", "speed_std",
                "speed_max", "stationary_ratio"
            ]:
                d[f"{prefix}_{name}"] = np.nan

        rows.append(d)

    seq = pd.DataFrame(rows, index=df.index)

    # Distances from first/last event point to click/hover points.
    for side in ["last", "first"]:
        x_col = f"{prefix}_x_{side}"
        y_col = f"{prefix}_y_{side}"
        seq[f"{prefix}_{side}_to_down"] = dist_xy(
            seq[x_col], seq[y_col], df["pointerdown_x"], df["pointerdown_y"]
        )
        seq[f"{prefix}_{side}_to_up"] = dist_xy(
            seq[x_col], seq[y_col], df["pointerup_x"], df["pointerup_y"]
        )

    seq[f"{prefix}_last_to_hover"] = dist_xy(
        seq[f"{prefix}_x_last"], seq[f"{prefix}_y_last"], df["hover_x"], df["hover_y"]
    )

    return seq


def extract_base_features(df):
    out = pd.DataFrame(index=df.index)

    raw_cols = [
        "relative_captcha_init_time",
        "mouse_events_total",
        "touch_events_total",
        "pointerdown_timestamp",
        "pointerdown_x",
        "pointerdown_y",
        "pointerup_timestamp",
        "pointerup_x",
        "pointerup_y",
        "hover_timestamp",
        "hover_x",
        "hover_y",
        "viewport_width",
        "viewport_height",
    ]

    for col in raw_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = np.nan

    out["click_duration"] = out["pointerup_timestamp"] - out["pointerdown_timestamp"]
    out["time_init_down"] = out["pointerdown_timestamp"] - out["relative_captcha_init_time"]
    out["time_init_up"] = out["pointerup_timestamp"] - out["relative_captcha_init_time"]
    out["time_hover_down"] = out["pointerdown_timestamp"] - out["hover_timestamp"]
    out["time_hover_up"] = out["pointerup_timestamp"] - out["hover_timestamp"]

    out["pointer_dx"] = out["pointerup_x"] - out["pointerdown_x"]
    out["pointer_dy"] = out["pointerup_y"] - out["pointerdown_y"]
    out["pointer_dist"] = dist_xy(
        out["pointerdown_x"], out["pointerdown_y"], out["pointerup_x"], out["pointerup_y"]
    )
    out["click_speed"] = safe_div(out["pointer_dist"], out["click_duration"])

    out["hover_down_dist"] = dist_xy(
        out["hover_x"], out["hover_y"], out["pointerdown_x"], out["pointerdown_y"]
    )
    out["hover_up_dist"] = dist_xy(
        out["hover_x"], out["hover_y"], out["pointerup_x"], out["pointerup_y"]
    )

    out["hover_zero"] = (
        (out["hover_timestamp"] == 0) & (out["hover_x"] == 0) & (out["hover_y"] == 0)
    ).astype(int)

    out["viewport_area"] = out["viewport_width"] * out["viewport_height"]
    out["viewport_aspect"] = safe_div(out["viewport_width"], out["viewport_height"])
    out["viewport_invalid"] = ((out["viewport_width"] <= 0) | (out["viewport_height"] <= 0)).astype(int)

    out["vp_w_log"] = np.log1p(np.maximum(out["viewport_width"], 0))
    out["vp_h_log"] = np.log1p(np.maximum(out["viewport_height"], 0))
    out["vp_area_log"] = np.log1p(np.maximum(out["viewport_area"], 0))

    out["mobile_like"] = ((out["viewport_width"] > 0) & (out["viewport_width"] <= 600)).astype(int)
    out["desktop_like"] = ((out["viewport_width"] >= 1200) & (out["viewport_height"] >= 700)).astype(int)
    out["wide_screen"] = (out["viewport_width"] >= 1800).astype(int)

    out["pd_x_norm"] = safe_div(out["pointerdown_x"], out["viewport_width"])
    out["pd_y_norm"] = safe_div(out["pointerdown_y"], out["viewport_height"])
    out["pu_x_norm"] = safe_div(out["pointerup_x"], out["viewport_width"])
    out["pu_y_norm"] = safe_div(out["pointerup_y"], out["viewport_height"])
    out["h_x_norm"] = safe_div(out["hover_x"], out["viewport_width"])
    out["h_y_norm"] = safe_div(out["hover_y"], out["viewport_height"])

    out["pd_outside"] = (
        (out["pointerdown_x"] < 0) | (out["pointerdown_x"] > out["viewport_width"]) |
        (out["pointerdown_y"] < 0) | (out["pointerdown_y"] > out["viewport_height"])
    ).astype(int)
    out["pu_outside"] = (
        (out["pointerup_x"] < 0) | (out["pointerup_x"] > out["viewport_width"]) |
        (out["pointerup_y"] < 0) | (out["pointerup_y"] > out["viewport_height"])
    ).astype(int)
    out["h_outside"] = (
        (out["hover_x"] < 0) | (out["hover_x"] > out["viewport_width"]) |
        (out["hover_y"] < 0) | (out["hover_y"] > out["viewport_height"])
    ).astype(int)

    out["events_sum"] = out["mouse_events_total"] + out["touch_events_total"]
    out["mouse_touch_ratio"] = (out["mouse_events_total"] + 1) / (out["touch_events_total"] + 1)
    out["has_mouse"] = (out["mouse_events_total"] > 0).astype(int)
    out["has_touch"] = (out["touch_events_total"] > 0).astype(int)
    out["only_mouse"] = ((out["mouse_events_total"] > 0) & (out["touch_events_total"] == 0)).astype(int)
    out["only_touch"] = ((out["touch_events_total"] > 0) & (out["mouse_events_total"] == 0)).astype(int)

    mouse_lens = df["mouse_events"].apply(lambda x: len(normalize_event_list(x))) if "mouse_events" in df.columns else 0
    touch_lens = df["touch_events"].apply(lambda x: len(normalize_event_list(x))) if "touch_events" in df.columns else 0
    out["mouse_total_minus_len"] = out["mouse_events_total"] - mouse_lens
    out["touch_total_minus_len"] = out["touch_events_total"] - touch_lens
    out["mouse_truncated"] = (out["mouse_events_total"] > mouse_lens).astype(int)
    out["touch_truncated"] = (out["touch_events_total"] > touch_lens).astype(int)

    if "mouse_events" in df.columns:
        out = pd.concat([out, extract_sequence_stats(df["mouse_events"], "mouse", out)], axis=1)
    if "touch_events" in df.columns:
        out = pd.concat([out, extract_sequence_stats(df["touch_events"], "touch", out)], axis=1)

    # Replace infinite values with NaN so the imputer handles them.
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def make_keys(X, cols):
    # Convert rows into stable tuple keys. NaN is normalized to string sentinel.
    tmp = X[cols].copy()
    tmp = tmp.where(~tmp.isna(), "__NA__")
    return list(map(tuple, tmp.astype(str).values))


def fit_te_maps(X, y, key_groups=KEY_GROUPS, smoothing=SMOOTHING):
    y = np.asarray(y, dtype=float)
    global_rate = float(np.mean(y))
    maps = {"global_rate": global_rate, "groups": {}}

    for name, cols in key_groups.items():
        keys = make_keys(X, cols)
        stat = defaultdict(lambda: [0.0, 0.0])
        for k, target in zip(keys, y):
            stat[k][0] += float(target)
            stat[k][1] += 1.0

        te_map = {}
        freq_map = {}
        for k, (bot_count, count) in stat.items():
            te_map[k] = (bot_count + global_rate * smoothing) / (count + smoothing)
            freq_map[k] = count

        maps["groups"][name] = {
            "cols": cols,
            "te": te_map,
            "freq": freq_map,
        }

    return maps


def apply_te_maps(X, maps):
    X = X.copy()
    global_rate = maps["global_rate"]

    for name, cfg in maps["groups"].items():
        cols = cfg["cols"]
        keys = make_keys(X, cols)
        X[f"te_{name}"] = [cfg["te"].get(k, global_rate) for k in keys]
        X[f"freq_{name}"] = [cfg["freq"].get(k, 0.0) for k in keys]

    return X


def get_models():
    return [
        (
            "logit",
            make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=0.2, max_iter=3000)
            ),
        ),
        (
            "extra_trees",
            make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesClassifier(
                    n_estimators=450,
                    max_features="sqrt",
                    min_samples_leaf=3,
                    random_state=1,
                    n_jobs=-1,
                )
            ),
        ),
        (
            "random_forest",
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=300,
                    max_features="sqrt",
                    min_samples_leaf=5,
                    random_state=2,
                    n_jobs=-1,
                )
            ),
        ),
    ]


def choose_usable_columns(X):
    usable = []
    for col in X.columns:
        s = X[col]
        if s.isna().all():
            continue
        if s.nunique(dropna=False) <= 1:
            continue
        usable.append(col)
    return usable


def fit_pipeline(train_path):
    train = pd.read_parquet(train_path)
    if "target" not in train.columns:
        raise ValueError("Training file must contain target column")

    y = train["target"].astype(int).values
    base = extract_base_features(train)

    # Full TE only to determine full candidate columns and final training matrix.
    full_maps = fit_te_maps(base, y)
    X_full_candidate = apply_te_maps(base, full_maps)
    usable_cols = choose_usable_columns(X_full_candidate)

    models = get_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof = {name: np.zeros(len(train), dtype=float) for name, _ in models}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(base, y), 1):
        X_tr_base = base.iloc[tr_idx]
        X_val_base = base.iloc[val_idx]
        y_tr = y[tr_idx]

        fold_maps = fit_te_maps(X_tr_base, y_tr)
        X_tr = apply_te_maps(X_tr_base, fold_maps).reindex(columns=usable_cols)
        X_val = apply_te_maps(X_val_base, fold_maps).reindex(columns=usable_cols)

        for name, model in models:
            m = clone(model)
            m.fit(X_tr, y_tr)
            oof[name][val_idx] = m.predict_proba(X_val)[:, 1]

    model_scores = {}
    for name, preds in oof.items():
        p = np.clip(preds, 1e-6, 1 - 1e-6)
        model_scores[name] = {
            "auc": roc_auc_score(y, p),
            "logloss": log_loss(y, p),
            "brier": brier_score_loss(y, p),
        }

    inv = {name: 1.0 / (score["logloss"] ** 2) for name, score in model_scores.items()}
    total_inv = sum(inv.values())
    weights = {name: val / total_inv for name, val in inv.items()}

    raw_oof = np.zeros(len(train), dtype=float)
    for name in oof:
        raw_oof += weights[name] * oof[name]
    raw_oof = np.clip(raw_oof, 1e-6, 1 - 1e-6)

    logit_oof = np.log(raw_oof / (1 - raw_oof)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=1000)
    calibrator.fit(logit_oof, y)

    calibrated_oof = calibrator.predict_proba(logit_oof)[:, 1]
    final_oof = np.clip(0.7 * calibrated_oof + 0.3 * raw_oof, 0.001, 0.999)

    print("OOF model scores:")
    for name, score in model_scores.items():
        print(name, score, "weight=", weights[name])

    print("Raw ensemble:", {
        "auc": roc_auc_score(y, raw_oof),
        "logloss": log_loss(y, raw_oof),
        "brier": brier_score_loss(y, raw_oof),
    })
    print("Final calibrated blend:", {
        "auc": roc_auc_score(y, final_oof),
        "logloss": log_loss(y, final_oof),
        "brier": brier_score_loss(y, final_oof),
    })

    # Train final models on full training set with TE maps fitted on full train.
    X_full = X_full_candidate.reindex(columns=usable_cols)
    fitted_models = []
    for name, model in models:
        m = clone(model)
        m.fit(X_full, y)
        fitted_models.append((name, m))

    return {
        "te_maps": full_maps,
        "usable_cols": usable_cols,
        "weights": weights,
        "models": fitted_models,
        "calibrator": calibrator,
    }


def predict_batch(df, state):
    base = extract_base_features(df)
    X = apply_te_maps(base, state["te_maps"]).reindex(columns=state["usable_cols"])

    raw = np.zeros(len(df), dtype=float)
    for name, model in state["models"]:
        raw += state["weights"][name] * model.predict_proba(X)[:, 1]

    raw = np.clip(raw, 1e-6, 1 - 1e-6)
    logit = np.log(raw / (1 - raw)).reshape(-1, 1)
    cal = state["calibrator"].predict_proba(logit)[:, 1]
    final = np.clip(0.7 * cal + 0.3 * raw, 0.001, 0.999)
    return final


def predict_test_to_csv(test_path, out_path, state, batch_size=5000):
    pf = pq.ParquetFile(test_path)
    row_offset = 0
    first = True

    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        probs = predict_batch(df, state)
        ids = np.arange(row_offset, row_offset + len(df))
        row_offset += len(df)

        out = pd.DataFrame({"id": ids, "probability": probs})
        out.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
        first = False

    print(f"Wrote {row_offset} predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    state = fit_pipeline(args.train)
    predict_test_to_csv(args.test, args.out, state, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
```

---

# 7. Output validation checklist

After generating predictions, validate the file:

```python
import pandas as pd

pred = pd.read_csv("/mnt/data/predictions.csv")
test_rows = 100000  # replace with actual len(test) if needed

assert list(pred.columns) == ["id", "probability"]
assert len(pred) == test_rows
assert pred["id"].is_unique
assert pred["probability"].notna().all()
assert pred["probability"].between(0, 1).all()
```

If test has a real id column in another version of the dataset, replace row-index ids with that column.

---

# 8. Summary of the approach

The full pipeline is:

```text
Load train/test parquet
-> Extract raw scalar features
-> Aggregate mouse/touch trajectories
-> Generate timing, geometry, viewport, modality, and consistency features
-> Add frequency features
-> Add out-of-fold smoothed target encodings
-> Train Logistic Regression, ExtraTrees, RandomForest
-> Weight models by inverse squared validation log loss
-> Blend probabilities
-> Fit Platt calibrator on out-of-fold blend
-> Predict test in batches
-> Apply conservative calibrated/raw blend
-> Clip probabilities
-> Save CSV with id,probability
```

This is a normal tabular behavioral ML pipeline: no magic, no mystical cyber-aura, just feature engineering, cross-validation, ensembling, and calibration. A tiny miracle by modern standards.
