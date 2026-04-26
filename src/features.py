from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

EPS = 1e-9


def _safe_div(a: float, b: float) -> float:
    try:
        b = float(b)
        if not np.isfinite(b) or abs(b) < EPS:
            return np.nan
        return float(a) / b
    except Exception:
        return np.nan


def _to_event_frame(events: Any) -> pd.DataFrame:
    if events is None:
        return pd.DataFrame()
    if isinstance(events, np.ndarray):
        events = events.tolist()
    if not isinstance(events, (list, tuple)) or len(events) == 0:
        return pd.DataFrame()

    rows = []
    for item in events:
        if isinstance(item, dict):
            rows.append(item.copy())
        else:
            try:
                rows.append(dict(item))
            except Exception:
                pass

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.columns = [c[:-1] if isinstance(c, str) and c.endswith("_") else c for c in df.columns]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _series_stats(values: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
        }

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def _summarize_sequence(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    feats: Dict[str, float] = {f"{prefix}n": float(len(df))}

    scalar_cols = [c for c in ["x", "y", "timestamp"] if c in df.columns]
    if len(df) == 0 or not scalar_cols:
        for key in [
            "duration",
            "path_length",
            "net_dx",
            "net_dy",
            "straight",
            "tortuosity",
            "bbox_w",
            "bbox_h",
            "bbox_area",
            "step_mean",
            "step_std",
            "step_min",
            "step_max",
            "step_median",
            "dt_mean",
            "dt_std",
            "dt_min",
            "dt_max",
            "speed_mean",
            "speed_std",
            "speed_min",
            "speed_max",
            "turn_mean",
            "turn_std",
            "turn_abs_mean",
            "unique_xy_ratio",
            "zero_step_ratio",
            "pause_100_ratio",
            "pause_250_ratio",
            "pause_500_ratio",
            "pause_1000_ratio",
        ]:
            feats[f"{prefix}{key}"] = np.nan
        return feats

    x = pd.to_numeric(df.get("x"), errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df.get("y"), errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(df.get("timestamp"), errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    t = t[valid] if len(t) == len(valid) else np.asarray(t, dtype=float)

    if len(x) == 0:
        for key in [
            "duration",
            "path_length",
            "net_dx",
            "net_dy",
            "straight",
            "tortuosity",
            "bbox_w",
            "bbox_h",
            "bbox_area",
            "step_mean",
            "step_std",
            "step_min",
            "step_max",
            "step_median",
            "dt_mean",
            "dt_std",
            "dt_min",
            "dt_max",
            "speed_mean",
            "speed_std",
            "speed_min",
            "speed_max",
            "turn_mean",
            "turn_std",
            "turn_abs_mean",
            "unique_xy_ratio",
            "zero_step_ratio",
            "pause_100_ratio",
            "pause_250_ratio",
            "pause_500_ratio",
            "pause_1000_ratio",
        ]:
            feats[f"{prefix}{key}"] = np.nan
        return feats

    feats[f"{prefix}first_x"] = float(x[0])
    feats[f"{prefix}first_y"] = float(y[0])
    feats[f"{prefix}last_x"] = float(x[-1])
    feats[f"{prefix}last_y"] = float(y[-1])
    feats[f"{prefix}mean_x"] = float(np.mean(x))
    feats[f"{prefix}mean_y"] = float(np.mean(y))
    feats[f"{prefix}std_x"] = float(np.std(x))
    feats[f"{prefix}std_y"] = float(np.std(y))
    feats[f"{prefix}median_x"] = float(np.median(x))
    feats[f"{prefix}median_y"] = float(np.median(y))
    feats[f"{prefix}q10_x"] = float(np.quantile(x, 0.10))
    feats[f"{prefix}q90_x"] = float(np.quantile(x, 0.90))
    feats[f"{prefix}q10_y"] = float(np.quantile(y, 0.10))
    feats[f"{prefix}q90_y"] = float(np.quantile(y, 0.90))

    dx = np.diff(x)
    dy = np.diff(y)
    step = np.sqrt(dx * dx + dy * dy)
    dt = np.diff(t) if len(t) == len(x) and len(x) > 1 else np.array([], dtype=float)

    net_dx = float(x[-1] - x[0]) if len(x) > 1 else 0.0
    net_dy = float(y[-1] - y[0]) if len(y) > 1 else 0.0
    straight = float(math.hypot(net_dx, net_dy))
    bbox_w = float(np.max(x) - np.min(x))
    bbox_h = float(np.max(y) - np.min(y))
    bbox_area = bbox_w * bbox_h
    unique_xy_ratio = len({(float(a), float(b)) for a, b in zip(x, y)}) / max(len(x), 1)
    zero_step_ratio = float(np.mean(step < EPS)) if len(step) else np.nan

    feats[f"{prefix}duration"] = float(t[-1] - t[0]) if len(dt) else np.nan
    feats[f"{prefix}path_length"] = float(np.sum(step)) if len(step) else np.nan
    feats[f"{prefix}net_dx"] = net_dx
    feats[f"{prefix}net_dy"] = net_dy
    feats[f"{prefix}straight"] = straight
    feats[f"{prefix}tortuosity"] = _safe_div(feats[f"{prefix}path_length"], straight)
    feats[f"{prefix}bbox_w"] = bbox_w
    feats[f"{prefix}bbox_h"] = bbox_h
    feats[f"{prefix}bbox_area"] = bbox_area
    feats[f"{prefix}step_mean"] = float(np.mean(step)) if len(step) else np.nan
    feats[f"{prefix}step_std"] = float(np.std(step)) if len(step) else np.nan
    feats[f"{prefix}step_min"] = float(np.min(step)) if len(step) else np.nan
    feats[f"{prefix}step_max"] = float(np.max(step)) if len(step) else np.nan
    feats[f"{prefix}step_median"] = float(np.median(step)) if len(step) else np.nan
    feats[f"{prefix}dt_mean"] = float(np.mean(dt)) if len(dt) else np.nan
    feats[f"{prefix}dt_std"] = float(np.std(dt)) if len(dt) else np.nan
    feats[f"{prefix}dt_min"] = float(np.min(dt)) if len(dt) else np.nan
    feats[f"{prefix}dt_max"] = float(np.max(dt)) if len(dt) else np.nan

    if len(step) and len(dt) == len(step):
        speed = np.divide(step, dt, out=np.full_like(step, np.nan), where=np.isfinite(dt) & (np.abs(dt) > EPS))
        feats[f"{prefix}speed_mean"] = float(np.nanmean(speed)) if np.isfinite(speed).any() else np.nan
        feats[f"{prefix}speed_std"] = float(np.nanstd(speed)) if np.isfinite(speed).any() else np.nan
        feats[f"{prefix}speed_min"] = float(np.nanmin(speed)) if np.isfinite(speed).any() else np.nan
        feats[f"{prefix}speed_max"] = float(np.nanmax(speed)) if np.isfinite(speed).any() else np.nan
    else:
        feats[f"{prefix}speed_mean"] = np.nan
        feats[f"{prefix}speed_std"] = np.nan
        feats[f"{prefix}speed_min"] = np.nan
        feats[f"{prefix}speed_max"] = np.nan

    if len(dx) > 1:
        ang = np.arctan2(dy, dx)
        d_ang = (np.diff(ang) + np.pi) % (2 * np.pi) - np.pi
        feats[f"{prefix}turn_mean"] = float(np.mean(d_ang))
        feats[f"{prefix}turn_std"] = float(np.std(d_ang))
        feats[f"{prefix}turn_abs_mean"] = float(np.mean(np.abs(d_ang)))
    else:
        feats[f"{prefix}turn_mean"] = np.nan
        feats[f"{prefix}turn_std"] = np.nan
        feats[f"{prefix}turn_abs_mean"] = np.nan

    feats[f"{prefix}unique_xy_ratio"] = unique_xy_ratio
    feats[f"{prefix}zero_step_ratio"] = zero_step_ratio
    feats[f"{prefix}pause_100_ratio"] = float(np.mean(dt > 100.0)) if len(dt) else np.nan
    feats[f"{prefix}pause_250_ratio"] = float(np.mean(dt > 250.0)) if len(dt) else np.nan
    feats[f"{prefix}pause_500_ratio"] = float(np.mean(dt > 500.0)) if len(dt) else np.nan
    feats[f"{prefix}pause_1000_ratio"] = float(np.mean(dt > 1000.0)) if len(dt) else np.nan
    return feats


def _row_features(row: Dict[str, Any]) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    vw = float(row.get("viewport_width", np.nan)) if pd.notna(row.get("viewport_width", np.nan)) else np.nan
    vh = float(row.get("viewport_height", np.nan)) if pd.notna(row.get("viewport_height", np.nan)) else np.nan
    vw_safe = max(vw if np.isfinite(vw) else 1.0, 1.0)
    vh_safe = max(vh if np.isfinite(vh) else 1.0, 1.0)

    feats["viewport_width"] = vw_safe
    feats["viewport_height"] = vh_safe
    feats["viewport_aspect"] = _safe_div(vw_safe, vh_safe)
    feats["relative_captcha_init_time"] = float(row.get("relative_captcha_init_time", np.nan))
    feats["mouse_events_total"] = float(row.get("mouse_events_total", np.nan))
    feats["touch_events_total"] = float(row.get("touch_events_total", np.nan))
    feats["has_mouse"] = int((row.get("mouse_events_total", 0) or 0) > 0)
    feats["has_touch"] = int((row.get("touch_events_total", 0) or 0) > 0)
    feats["mouse_share"] = _safe_div(feats["mouse_events_total"], feats["mouse_events_total"] + feats["touch_events_total"])
    feats["touch_share"] = _safe_div(feats["touch_events_total"], feats["mouse_events_total"] + feats["touch_events_total"])
    feats["init_time_log"] = np.log1p(max(feats["relative_captcha_init_time"], 0)) if np.isfinite(feats["relative_captcha_init_time"]) else np.nan
    feats["mouse_log"] = np.log1p(max(feats["mouse_events_total"], 0)) if np.isfinite(feats["mouse_events_total"]) else np.nan
    feats["touch_log"] = np.log1p(max(feats["touch_events_total"], 0)) if np.isfinite(feats["touch_events_total"]) else np.nan

    down_ts = float(row.get("pointerdown_timestamp", np.nan))
    up_ts = float(row.get("pointerup_timestamp", np.nan))
    hover_ts = float(row.get("hover_timestamp", np.nan))

    feats["down_ts"] = down_ts
    feats["up_ts"] = up_ts
    feats["hover_ts"] = hover_ts
    feats["down_to_up_ms"] = up_ts - down_ts if np.isfinite(up_ts) and np.isfinite(down_ts) else np.nan
    feats["init_to_down_ms"] = down_ts - feats["relative_captcha_init_time"] if np.isfinite(down_ts) and np.isfinite(feats["relative_captcha_init_time"]) else np.nan
    feats["init_to_up_ms"] = up_ts - feats["relative_captcha_init_time"] if np.isfinite(up_ts) and np.isfinite(feats["relative_captcha_init_time"]) else np.nan
    feats["hover_to_down_ms"] = down_ts - hover_ts if np.isfinite(hover_ts) and np.isfinite(down_ts) else np.nan

    feats["down_x"] = float(row.get("pointerdown_x", np.nan))
    feats["down_y"] = float(row.get("pointerdown_y", np.nan))
    feats["up_x"] = float(row.get("pointerup_x", np.nan))
    feats["up_y"] = float(row.get("pointerup_y", np.nan))
    feats["hover_x"] = float(row.get("hover_x", np.nan))
    feats["hover_y"] = float(row.get("hover_y", np.nan))

    feats["down_x_norm"] = _safe_div(feats["down_x"], vw_safe)
    feats["down_y_norm"] = _safe_div(feats["down_y"], vh_safe)
    feats["up_x_norm"] = _safe_div(feats["up_x"], vw_safe)
    feats["up_y_norm"] = _safe_div(feats["up_y"], vh_safe)
    feats["hover_x_norm"] = _safe_div(feats["hover_x"], vw_safe)
    feats["hover_y_norm"] = _safe_div(feats["hover_y"], vh_safe)

    feats["down_to_center"] = math.hypot(feats["down_x_norm"] - 0.5, feats["down_y_norm"] - 0.5) if np.isfinite(feats["down_x_norm"]) and np.isfinite(feats["down_y_norm"]) else np.nan
    feats["up_to_center"] = math.hypot(feats["up_x_norm"] - 0.5, feats["up_y_norm"] - 0.5) if np.isfinite(feats["up_x_norm"]) and np.isfinite(feats["up_y_norm"]) else np.nan
    feats["hover_to_center"] = math.hypot(feats["hover_x_norm"] - 0.5, feats["hover_y_norm"] - 0.5) if np.isfinite(feats["hover_x_norm"]) and np.isfinite(feats["hover_y_norm"]) else np.nan

    mouse_df = _to_event_frame(row.get("mouse_events", []))
    touch_df = _to_event_frame(row.get("touch_events", []))

    feats.update(_summarize_sequence(mouse_df, "mouse_"))
    feats.update(_summarize_sequence(touch_df, "touch_"))

    for col in ["force", "radiusX", "radiusY", "rotationAngle"]:
        if col in touch_df.columns:
            stats = _series_stats(touch_df[col].to_numpy(dtype=float))
        else:
            stats = {k: np.nan for k in ["mean", "std", "min", "max", "median", "p10", "p90"]}
        for key, value in stats.items():
            feats[f"touch_{col}_{key}"] = value

    return feats


def extract_features(row: Any) -> Dict[str, float]:
    if isinstance(row, pd.Series):
        row = row.to_dict()
    elif not isinstance(row, dict):
        row = dict(row)
    return _row_features(row)


def build_features(df: pd.DataFrame, n_jobs: int = -1, backend: str = "loky") -> pd.DataFrame:
    rows = df.to_dict("records")
    feats = Parallel(n_jobs=n_jobs, backend=backend, batch_size=256)(
        delayed(extract_features)(row) for row in rows
    )
    out = pd.DataFrame(feats)
    return out
