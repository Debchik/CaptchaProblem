from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from pynndescent import NNDescent
from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

EPS = 1e-9
SEQ_LEN = 100
SEQ_CHANNELS = 21
TRAJ_POINTS = 32
SSL_EMBED_DIM = 64
SEED = 42
KEY_GROUPS = {
    "vp": ["viewport_width", "viewport_height"],
    "counts": ["mouse_events_total", "touch_events_total"],
    "vp_counts": ["viewport_width", "viewport_height", "mouse_events_total", "touch_events_total"],
    "vp_hover": ["viewport_width", "viewport_height", "hover_zero"],
}
SMOOTHING = 8.0

SEQ_COLS = [
    "x_norm",
    "y_norm",
    "t_norm",
    "dx",
    "dy",
    "log_dt",
    "distance",
    "log_speed",
    "acceleration",
    "jerk",
    "sin_angle",
    "cos_angle",
    "sin_turn",
    "cos_turn",
    "is_mouse",
    "is_touch",
    "is_padding",
    "force",
    "radiusX",
    "radiusY",
    "rotationAngle",
]
SEQ_IDX = {name: idx for idx, name in enumerate(SEQ_COLS)}

BASE_NUMERIC_COLS = [
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


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def partial_auc_score(y_true: np.ndarray, y_pred: np.ndarray, max_fpr: float = 0.1) -> float:
    return float(roc_auc_score(y_true, y_pred, max_fpr=max_fpr))


def safe_div(a: Any, b: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > EPS, a / b, np.nan)


def dist_xy(x1: Any, y1: Any, x2: Any, y2: Any) -> Any:
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def is_missing_obj(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def event_get(ev: Any, key: str, default: Any = np.nan) -> Any:
    if ev is None:
        return default
    if isinstance(ev, dict):
        if key in ev:
            return ev.get(key, default)
        alt = key[:-1] if key.endswith("_") else f"{key}_"
        return ev.get(alt, default)
    try:
        return getattr(ev, key)
    except Exception:
        pass
    try:
        alt = key[:-1] if key.endswith("_") else f"{key}_"
        return getattr(ev, alt)
    except Exception:
        return default


def normalize_event_list(obj: Any) -> list[Any]:
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


def extract_sequence_stats(events_col: pd.Series, prefix: str, df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for obj in events_col:
        events = normalize_event_list(obj)
        ts, xs, ys = [], [], []

        for ev in events:
            ts.append(event_get(ev, "timestamp_", np.nan))
            xs.append(event_get(ev, "x_", np.nan))
            ys.append(event_get(ev, "y_", np.nan))

        t = np.asarray(ts, dtype=float)
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)

        valid = ~(np.isnan(t) | np.isnan(x) | np.isnan(y))
        t, x, y = t[valid], x[valid], y[valid]
        n = len(t)

        d: dict[str, float] = {f"{prefix}_len": float(n)}
        if n == 0:
            base_names = [
                "dur",
                "x_rng",
                "y_rng",
                "x_std",
                "y_std",
                "unique_ratio",
                "x_first",
                "y_first",
                "x_last",
                "y_last",
                "t_first",
                "t_last",
                "path",
                "disp",
                "straight",
                "dt_mean",
                "dt_std",
                "dt_min",
                "dt_max",
                "step_mean",
                "step_std",
                "step_max",
                "speed_mean",
                "speed_std",
                "speed_max",
                "stationary_ratio",
            ]
            for name in base_names:
                d[f"{prefix}_{name}"] = np.nan
            rows.append(d)
            continue

        d[f"{prefix}_dur"] = float(t[-1] - t[0])
        d[f"{prefix}_x_rng"] = float(np.nanmax(x) - np.nanmin(x))
        d[f"{prefix}_y_rng"] = float(np.nanmax(y) - np.nanmin(y))
        d[f"{prefix}_x_std"] = float(np.nanstd(x))
        d[f"{prefix}_y_std"] = float(np.nanstd(y))
        d[f"{prefix}_unique_ratio"] = float(len(set(zip(x, y))) / max(n, 1))
        d[f"{prefix}_x_first"] = float(x[0])
        d[f"{prefix}_y_first"] = float(y[0])
        d[f"{prefix}_x_last"] = float(x[-1])
        d[f"{prefix}_y_last"] = float(y[-1])
        d[f"{prefix}_t_first"] = float(t[0])
        d[f"{prefix}_t_last"] = float(t[-1])

        if n >= 2:
            dx = np.diff(x)
            dy = np.diff(y)
            dt = np.diff(t)
            step = np.sqrt(dx ** 2 + dy ** 2)
            positive_dt = np.where(np.abs(dt) > EPS, dt, np.nan)
            speed = step / positive_dt
            path = np.nansum(step)
            disp = math.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)

            d[f"{prefix}_path"] = float(path)
            d[f"{prefix}_disp"] = float(disp)
            d[f"{prefix}_straight"] = float(disp / path) if path > EPS else np.nan
            d[f"{prefix}_dt_mean"] = float(np.nanmean(dt))
            d[f"{prefix}_dt_std"] = float(np.nanstd(dt))
            d[f"{prefix}_dt_min"] = float(np.nanmin(dt))
            d[f"{prefix}_dt_max"] = float(np.nanmax(dt))
            d[f"{prefix}_step_mean"] = float(np.nanmean(step))
            d[f"{prefix}_step_std"] = float(np.nanstd(step))
            d[f"{prefix}_step_max"] = float(np.nanmax(step))
            d[f"{prefix}_speed_mean"] = float(np.nanmean(speed))
            d[f"{prefix}_speed_std"] = float(np.nanstd(speed))
            d[f"{prefix}_speed_max"] = float(np.nanmax(speed))
            d[f"{prefix}_stationary_ratio"] = float(np.nanmean(step == 0))
        else:
            for name in [
                "path",
                "disp",
                "straight",
                "dt_mean",
                "dt_std",
                "dt_min",
                "dt_max",
                "step_mean",
                "step_std",
                "step_max",
                "speed_mean",
                "speed_std",
                "speed_max",
                "stationary_ratio",
            ]:
                d[f"{prefix}_{name}"] = np.nan

        rows.append(d)

    seq = pd.DataFrame(rows, index=df.index)
    for side in ["last", "first"]:
        x_col = f"{prefix}_x_{side}"
        y_col = f"{prefix}_y_{side}"
        seq[f"{prefix}_{side}_to_down"] = dist_xy(
            seq[x_col],
            seq[y_col],
            df["pointerdown_x"],
            df["pointerdown_y"],
        )
        seq[f"{prefix}_{side}_to_up"] = dist_xy(
            seq[x_col],
            seq[y_col],
            df["pointerup_x"],
            df["pointerup_y"],
        )

    seq[f"{prefix}_last_to_hover"] = dist_xy(
        seq[f"{prefix}_x_last"],
        seq[f"{prefix}_y_last"],
        df["hover_x"],
        df["hover_y"],
    )
    return seq


def extract_competition_base_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in BASE_NUMERIC_COLS:
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
        out["pointerdown_x"],
        out["pointerdown_y"],
        out["pointerup_x"],
        out["pointerup_y"],
    )
    out["click_speed"] = safe_div(out["pointer_dist"], out["click_duration"])

    out["hover_down_dist"] = dist_xy(
        out["hover_x"],
        out["hover_y"],
        out["pointerdown_x"],
        out["pointerdown_y"],
    )
    out["hover_up_dist"] = dist_xy(
        out["hover_x"],
        out["hover_y"],
        out["pointerup_x"],
        out["pointerup_y"],
    )
    out["hover_zero"] = (
        (out["hover_timestamp"] == 0) & (out["hover_x"] == 0) & (out["hover_y"] == 0)
    ).astype(int)

    out["viewport_area"] = out["viewport_width"] * out["viewport_height"]
    out["viewport_aspect"] = safe_div(out["viewport_width"], out["viewport_height"])
    out["viewport_invalid"] = (
        (out["viewport_width"] <= 0) | (out["viewport_height"] <= 0)
    ).astype(int)
    out["vp_w_log"] = np.log1p(np.maximum(out["viewport_width"], 0))
    out["vp_h_log"] = np.log1p(np.maximum(out["viewport_height"], 0))
    out["vp_area_log"] = np.log1p(np.maximum(out["viewport_area"], 0))

    out["mobile_like"] = (
        (out["viewport_width"] > 0) & (out["viewport_width"] <= 600)
    ).astype(int)
    out["desktop_like"] = (
        (out["viewport_width"] >= 1200) & (out["viewport_height"] >= 700)
    ).astype(int)
    out["wide_screen"] = (out["viewport_width"] >= 1800).astype(int)

    out["pd_x_norm"] = safe_div(out["pointerdown_x"], out["viewport_width"])
    out["pd_y_norm"] = safe_div(out["pointerdown_y"], out["viewport_height"])
    out["pu_x_norm"] = safe_div(out["pointerup_x"], out["viewport_width"])
    out["pu_y_norm"] = safe_div(out["pointerup_y"], out["viewport_height"])
    out["h_x_norm"] = safe_div(out["hover_x"], out["viewport_width"])
    out["h_y_norm"] = safe_div(out["hover_y"], out["viewport_height"])

    out["pd_outside"] = (
        (out["pointerdown_x"] < 0)
        | (out["pointerdown_x"] > out["viewport_width"])
        | (out["pointerdown_y"] < 0)
        | (out["pointerdown_y"] > out["viewport_height"])
    ).astype(int)
    out["pu_outside"] = (
        (out["pointerup_x"] < 0)
        | (out["pointerup_x"] > out["viewport_width"])
        | (out["pointerup_y"] < 0)
        | (out["pointerup_y"] > out["viewport_height"])
    ).astype(int)
    out["h_outside"] = (
        (out["hover_x"] < 0)
        | (out["hover_x"] > out["viewport_width"])
        | (out["hover_y"] < 0)
        | (out["hover_y"] > out["viewport_height"])
    ).astype(int)

    out["events_sum"] = out["mouse_events_total"] + out["touch_events_total"]
    out["mouse_touch_ratio"] = (out["mouse_events_total"] + 1) / (out["touch_events_total"] + 1)
    out["has_mouse"] = (out["mouse_events_total"] > 0).astype(int)
    out["has_touch"] = (out["touch_events_total"] > 0).astype(int)
    out["only_mouse"] = (
        (out["mouse_events_total"] > 0) & (out["touch_events_total"] == 0)
    ).astype(int)
    out["only_touch"] = (
        (out["touch_events_total"] > 0) & (out["mouse_events_total"] == 0)
    ).astype(int)

    mouse_lens = (
        df["mouse_events"].apply(lambda x: len(normalize_event_list(x)))
        if "mouse_events" in df.columns
        else 0
    )
    touch_lens = (
        df["touch_events"].apply(lambda x: len(normalize_event_list(x)))
        if "touch_events" in df.columns
        else 0
    )
    out["mouse_total_minus_len"] = out["mouse_events_total"] - mouse_lens
    out["touch_total_minus_len"] = out["touch_events_total"] - touch_lens
    out["mouse_truncated"] = (out["mouse_events_total"] > mouse_lens).astype(int)
    out["touch_truncated"] = (out["touch_events_total"] > touch_lens).astype(int)

    if "mouse_events" in df.columns:
        out = pd.concat([out, extract_sequence_stats(df["mouse_events"], "mouse", out)], axis=1)
    if "touch_events" in df.columns:
        out = pd.concat([out, extract_sequence_stats(df["touch_events"], "touch", out)], axis=1)

    return out.replace([np.inf, -np.inf], np.nan)


def make_keys(X: pd.DataFrame, cols: list[str]) -> list[tuple[str, ...]]:
    tmp = X[cols].copy()
    tmp = tmp.where(~tmp.isna(), "__NA__")
    return list(map(tuple, tmp.astype(str).values))


def fit_te_maps(
    X: pd.DataFrame,
    y: np.ndarray,
    key_groups: dict[str, list[str]] = KEY_GROUPS,
    smoothing: float = SMOOTHING,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    global_rate = float(np.mean(y))
    maps: dict[str, Any] = {"global_rate": global_rate, "groups": {}}

    for name, cols in key_groups.items():
        keys = make_keys(X, cols)
        stat: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])
        for key, target in zip(keys, y):
            stat[key][0] += float(target)
            stat[key][1] += 1.0

        te_map: dict[tuple[str, ...], float] = {}
        freq_map: dict[tuple[str, ...], float] = {}
        for key, (human_count, count) in stat.items():
            te_map[key] = (human_count + global_rate * smoothing) / (count + smoothing)
            freq_map[key] = count

        maps["groups"][name] = {"cols": cols, "te": te_map, "freq": freq_map}
    return maps


def apply_te_maps(X: pd.DataFrame, maps: dict[str, Any]) -> pd.DataFrame:
    X = X.copy()
    global_rate = maps["global_rate"]
    for name, cfg in maps["groups"].items():
        cols = cfg["cols"]
        keys = make_keys(X, cols)
        X[f"te_{name}"] = [cfg["te"].get(key, global_rate) for key in keys]
        X[f"freq_{name}"] = [cfg["freq"].get(key, 0.0) for key in keys]
    return X


def choose_usable_columns(X: pd.DataFrame) -> list[str]:
    usable = []
    for col in X.columns:
        s = X[col]
        if s.isna().all():
            continue
        if s.nunique(dropna=False) <= 1:
            continue
        usable.append(col)
    return usable


def _as_float(x: Any, default: float = np.nan) -> float:
    try:
        value = float(x)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def _combine_events(row: dict[str, Any]) -> dict[str, np.ndarray]:
    items: list[tuple[float, float, float, int, float, float, float, float]] = []

    for ev in normalize_event_list(row.get("mouse_events")):
        t = _as_float(event_get(ev, "timestamp_"))
        x = _as_float(event_get(ev, "x_"))
        y = _as_float(event_get(ev, "y_"))
        if np.isfinite(t) and np.isfinite(x) and np.isfinite(y):
            items.append((t, x, y, 0, 0.0, 0.0, 0.0, 0.0))

    for ev in normalize_event_list(row.get("touch_events")):
        t = _as_float(event_get(ev, "timestamp_"))
        x = _as_float(event_get(ev, "x_"))
        y = _as_float(event_get(ev, "y_"))
        force = _as_float(event_get(ev, "force_", 0.0), 0.0)
        radius_x = _as_float(event_get(ev, "radiusX_", 0.0), 0.0)
        radius_y = _as_float(event_get(ev, "radiusY_", 0.0), 0.0)
        rotation = _as_float(event_get(ev, "rotationAngle_", 0.0), 0.0)
        if np.isfinite(t) and np.isfinite(x) and np.isfinite(y):
            items.append((t, x, y, 1, force, radius_x, radius_y, rotation))

    if not items:
        return {
            "t": np.array([], dtype=np.float32),
            "x": np.array([], dtype=np.float32),
            "y": np.array([], dtype=np.float32),
            "src": np.array([], dtype=np.int8),
            "force": np.array([], dtype=np.float32),
            "radiusX": np.array([], dtype=np.float32),
            "radiusY": np.array([], dtype=np.float32),
            "rotationAngle": np.array([], dtype=np.float32),
        }

    items.sort(key=lambda x: x[0])
    arr = np.asarray(items, dtype=np.float32)
    return {
        "t": arr[:, 0],
        "x": arr[:, 1],
        "y": arr[:, 2],
        "src": arr[:, 3].astype(np.int8),
        "force": arr[:, 4],
        "radiusX": arr[:, 5],
        "radiusY": arr[:, 6],
        "rotationAngle": arr[:, 7],
    }


def _normalize_xy(
    x: np.ndarray,
    y: np.ndarray,
    viewport_width: float,
    viewport_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = viewport_width if np.isfinite(viewport_width) and viewport_width > 0 else max(float(np.nanmax(np.abs(x))) if len(x) else 1.0, 1.0)
    height = viewport_height if np.isfinite(viewport_height) and viewport_height > 0 else max(float(np.nanmax(np.abs(y))) if len(y) else 1.0, 1.0)
    return x / width, y / height


def _interp_1d(source_pos: np.ndarray, values: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(len(target_pos), dtype=np.float32)
    if len(values) == 1:
        return np.repeat(values.astype(np.float32), len(target_pos))
    source_pos = np.asarray(source_pos, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if np.allclose(source_pos[0], source_pos[-1]):
        return np.repeat(values[-1], len(target_pos)).astype(np.float32)
    return np.interp(target_pos, source_pos, values).astype(np.float32)


def _resample_by_time(t: np.ndarray, x: np.ndarray, y: np.ndarray, n_points: int = TRAJ_POINTS) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(n_points * 2, dtype=np.float32)
    if len(x) == 1:
        vec = np.column_stack([np.repeat(x[0], n_points), np.repeat(y[0], n_points)]).reshape(-1)
        return vec.astype(np.float32)

    t0 = float(t[0])
    duration = float(t[-1] - t0)
    if duration <= EPS:
        pos = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    else:
        pos = ((t - t0) / duration).astype(np.float32)
    target = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    x_rs = _interp_1d(pos, x, target)
    y_rs = _interp_1d(pos, y, target)
    return np.column_stack([x_rs, y_rs]).reshape(-1).astype(np.float32)


def _resample_by_arc(x: np.ndarray, y: np.ndarray, n_points: int = TRAJ_POINTS) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(n_points * 2, dtype=np.float32)
    if len(x) == 1:
        vec = np.column_stack([np.repeat(x[0], n_points), np.repeat(y[0], n_points)]).reshape(-1)
        return vec.astype(np.float32)

    step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    total = float(arc[-1])
    if total <= EPS:
        target = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
        pos = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    else:
        pos = (arc / total).astype(np.float32)
        target = np.linspace(0.0, 1.0, n_points, dtype=np.float32)
    x_rs = _interp_1d(pos, x, target)
    y_rs = _interp_1d(pos, y, target)
    return np.column_stack([x_rs, y_rs]).reshape(-1).astype(np.float32)


def _normalize_start_end(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return x, y
    x0 = x - x[0]
    y0 = y - y[0]
    if len(x0) < 2:
        return x0, y0
    straight = float(math.hypot(x0[-1], y0[-1]))
    path = float(np.sum(np.sqrt(np.diff(x0) ** 2 + np.diff(y0) ** 2)))
    scale = straight if straight > EPS else path
    scale = scale if scale > EPS else 1.0
    return (x0 / scale).astype(np.float32), (y0 / scale).astype(np.float32)


def _bin_direction(angle: np.ndarray, n_bins: int = 8) -> np.ndarray:
    wrapped = (angle + 2.0 * np.pi) % (2.0 * np.pi)
    bins = np.floor(wrapped / (2.0 * np.pi / n_bins)).astype(np.int16)
    return np.clip(bins, 0, n_bins - 1)


def _bin_turn(turn: np.ndarray, n_bins: int = 8) -> np.ndarray:
    wrapped = (turn + np.pi) % (2.0 * np.pi)
    bins = np.floor(wrapped / (2.0 * np.pi / n_bins)).astype(np.int16)
    return np.clip(bins, 0, n_bins - 1)


def _digitize(values: np.ndarray, bins: list[float]) -> np.ndarray:
    return np.digitize(values, bins=bins, right=False).astype(np.int16)


def _stable_hash(parts: list[str]) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _state_from_step(src: np.ndarray, dt: np.ndarray, dist: np.ndarray) -> np.ndarray:
    is_pause = dt > 250.0
    is_stationary = dist < 1e-3
    return (src.astype(np.int16) * 4 + is_pause.astype(np.int16) * 2 + is_stationary.astype(np.int16)).astype(
        np.int16
    )


def _build_symbolic_features(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    src: np.ndarray,
) -> dict[str, Any]:
    if len(x) < 2:
        empty_hash = _stable_hash(["empty"])
        return {
            "text": "pad",
            "dir_hash": empty_hash,
            "speed_hash": empty_hash,
            "turn_hash": empty_hash,
            "state_hash": empty_hash,
            "dir_speed_hash": empty_hash,
            "dir_turn_dt_hash": empty_hash,
        }

    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    safe_dt = np.where(np.abs(dt) > EPS, dt, np.nan)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    speed = np.divide(dist, safe_dt, out=np.zeros_like(dist), where=np.isfinite(safe_dt))
    log_speed = np.log1p(np.maximum(speed, 0))
    log_dt = np.log1p(np.maximum(dt, 0))
    angle = np.arctan2(dy, dx)
    turn = np.zeros_like(angle)
    if len(angle) > 1:
        turn[1:] = (np.diff(angle) + np.pi) % (2.0 * np.pi) - np.pi

    direction_bin = _bin_direction(angle)
    speed_bin = _digitize(log_speed, [-1.5, -0.5, 0.2, 0.8, 1.5, 2.5])
    turn_bin = _bin_turn(turn)
    dt_bin = _digitize(log_dt, [1.0, 2.0, 3.0, 4.0, 5.0])
    state = _state_from_step(src[1:], dt, dist)

    step_tokens = [
        f"d{d}_s{s}_t{trn}_q{st}"
        for d, s, trn, st in zip(direction_bin.tolist(), speed_bin.tolist(), turn_bin.tolist(), state.tolist())
    ]
    text = " ".join(step_tokens) if step_tokens else "pad"

    dir_tokens = [f"d{d}" for d in direction_bin.tolist()]
    speed_tokens = [f"s{s}" for s in speed_bin.tolist()]
    turn_tokens = [f"t{trn}" for trn in turn_bin.tolist()]
    state_tokens = [f"q{st}" for st in state.tolist()]
    dir_speed_tokens = [f"d{d}s{s}" for d, s in zip(direction_bin.tolist(), speed_bin.tolist())]
    dir_turn_dt_tokens = [
        f"d{d}t{trn}u{dtb}"
        for d, trn, dtb in zip(direction_bin.tolist(), turn_bin.tolist(), dt_bin.tolist())
    ]

    return {
        "text": text,
        "dir_hash": _stable_hash(dir_tokens),
        "speed_hash": _stable_hash(speed_tokens),
        "turn_hash": _stable_hash(turn_tokens),
        "state_hash": _stable_hash(state_tokens),
        "dir_speed_hash": _stable_hash(dir_speed_tokens),
        "dir_turn_dt_hash": _stable_hash(dir_turn_dt_tokens),
    }


def _build_sequence_tensor(
    t: np.ndarray,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    src: np.ndarray,
    force: np.ndarray,
    radius_x: np.ndarray,
    radius_y: np.ndarray,
    rotation: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    tensor = np.zeros((seq_len, SEQ_CHANNELS), dtype=np.float16)
    tensor[:, SEQ_IDX["is_padding"]] = 1.0
    if len(x_norm) == 0:
        return tensor

    if len(x_norm) > seq_len:
        start = len(x_norm) - seq_len
        t = t[start:]
        x_norm = x_norm[start:]
        y_norm = y_norm[start:]
        src = src[start:]
        force = force[start:]
        radius_x = radius_x[start:]
        radius_y = radius_y[start:]
        rotation = rotation[start:]

    n = len(x_norm)
    duration = float(t[-1] - t[0]) if n > 1 else 0.0
    t_norm = ((t - t[0]) / duration).astype(np.float32) if duration > EPS else np.zeros(n, dtype=np.float32)

    dx = np.zeros(n, dtype=np.float32)
    dy = np.zeros(n, dtype=np.float32)
    dt = np.zeros(n, dtype=np.float32)
    if n > 1:
        dx[1:] = np.diff(x_norm)
        dy[1:] = np.diff(y_norm)
        dt[1:] = np.diff(t)
    log_dt = np.log1p(np.maximum(dt, 0))
    dist = np.sqrt(dx ** 2 + dy ** 2)
    speed = np.divide(dist, dt, out=np.zeros_like(dist), where=np.abs(dt) > EPS)
    log_speed = np.log1p(np.maximum(speed, 0))
    acceleration = np.zeros(n, dtype=np.float32)
    jerk = np.zeros(n, dtype=np.float32)
    if n > 1:
        acceleration[1:] = np.diff(speed, n=1)
    if n > 2:
        jerk[2:] = np.diff(acceleration[1:], n=1)
    angle = np.arctan2(dy, dx)
    turn = np.zeros(n, dtype=np.float32)
    if n > 1:
        turn[1:] = (np.diff(angle) + np.pi) % (2.0 * np.pi) - np.pi

    tensor[:n, SEQ_IDX["x_norm"]] = x_norm.astype(np.float16)
    tensor[:n, SEQ_IDX["y_norm"]] = y_norm.astype(np.float16)
    tensor[:n, SEQ_IDX["t_norm"]] = t_norm.astype(np.float16)
    tensor[:n, SEQ_IDX["dx"]] = dx.astype(np.float16)
    tensor[:n, SEQ_IDX["dy"]] = dy.astype(np.float16)
    tensor[:n, SEQ_IDX["log_dt"]] = log_dt.astype(np.float16)
    tensor[:n, SEQ_IDX["distance"]] = dist.astype(np.float16)
    tensor[:n, SEQ_IDX["log_speed"]] = log_speed.astype(np.float16)
    tensor[:n, SEQ_IDX["acceleration"]] = acceleration.astype(np.float16)
    tensor[:n, SEQ_IDX["jerk"]] = jerk.astype(np.float16)
    tensor[:n, SEQ_IDX["sin_angle"]] = np.sin(angle).astype(np.float16)
    tensor[:n, SEQ_IDX["cos_angle"]] = np.cos(angle).astype(np.float16)
    tensor[:n, SEQ_IDX["sin_turn"]] = np.sin(turn).astype(np.float16)
    tensor[:n, SEQ_IDX["cos_turn"]] = np.cos(turn).astype(np.float16)
    tensor[:n, SEQ_IDX["is_mouse"]] = (src == 0).astype(np.float16)
    tensor[:n, SEQ_IDX["is_touch"]] = (src == 1).astype(np.float16)
    tensor[:n, SEQ_IDX["is_padding"]] = 0.0
    tensor[:n, SEQ_IDX["force"]] = force.astype(np.float16)
    tensor[:n, SEQ_IDX["radiusX"]] = radius_x.astype(np.float16)
    tensor[:n, SEQ_IDX["radiusY"]] = radius_y.astype(np.float16)
    tensor[:n, SEQ_IDX["rotationAngle"]] = rotation.astype(np.float16)
    return tensor


def _build_row_artifacts(row: dict[str, Any]) -> dict[str, Any]:
    events = _combine_events(row)
    t = events["t"]
    x = events["x"]
    y = events["y"]
    src = events["src"]
    x_norm, y_norm = _normalize_xy(
        x,
        y,
        _as_float(row.get("viewport_width"), 1.0),
        _as_float(row.get("viewport_height"), 1.0),
    )

    view_vec = _resample_by_time(t, x_norm, y_norm, TRAJ_POINTS)
    geo_x, geo_y = _normalize_start_end(x_norm, y_norm)
    geo_vec = _resample_by_time(t, geo_x, geo_y, TRAJ_POINTS)
    arc_vec = _resample_by_arc(x_norm, y_norm, TRAJ_POINTS)
    sym = _build_symbolic_features(t, x_norm, y_norm, src)
    seq = _build_sequence_tensor(
        t=t,
        x_norm=x_norm,
        y_norm=y_norm,
        src=src,
        force=events["force"],
        radius_x=events["radiusX"],
        radius_y=events["radiusY"],
        rotation=events["rotationAngle"],
        seq_len=SEQ_LEN,
    )
    return {
        "traj_view": view_vec,
        "traj_geo": geo_vec,
        "traj_arc": arc_vec,
        "symbolic_text": sym["text"],
        "hashes": {k: v for k, v in sym.items() if k != "text"},
        "seq_tensor": seq,
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


@dataclass
class SplitCache:
    name: str
    n_rows: int
    base_path: Path
    hash_path: Path
    text_path: Path
    traj_view_path: Path
    traj_geo_path: Path
    traj_arc_path: Path
    seq_path: Path


def prepare_split_cache(
    name: str,
    data_path: Path,
    cache_dir: Path,
    batch_size: int = 2048,
) -> SplitCache:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / f"{name}_meta.json"
    base_path = cache_dir / f"{name}_base.parquet"
    hash_path = cache_dir / f"{name}_hashes.parquet"
    text_path = cache_dir / f"{name}_symbolic.txt"
    traj_view_path = cache_dir / f"{name}_traj_view.npy"
    traj_geo_path = cache_dir / f"{name}_traj_geo.npy"
    traj_arc_path = cache_dir / f"{name}_traj_arc.npy"
    seq_path = cache_dir / f"{name}_seq.npy"

    required = [meta_path, base_path, hash_path, text_path, traj_view_path, traj_geo_path, traj_arc_path, seq_path]
    if all(path.exists() for path in required):
        meta = _load_json(meta_path)
        return SplitCache(
            name=name,
            n_rows=int(meta["n_rows"]),
            base_path=base_path,
            hash_path=hash_path,
            text_path=text_path,
            traj_view_path=traj_view_path,
            traj_geo_path=traj_geo_path,
            traj_arc_path=traj_arc_path,
            seq_path=seq_path,
        )

    pf = pq.ParquetFile(data_path)
    n_rows = pf.metadata.num_rows

    traj_view = np.lib.format.open_memmap(traj_view_path, mode="w+", dtype=np.float32, shape=(n_rows, TRAJ_POINTS * 2))
    traj_geo = np.lib.format.open_memmap(traj_geo_path, mode="w+", dtype=np.float32, shape=(n_rows, TRAJ_POINTS * 2))
    traj_arc = np.lib.format.open_memmap(traj_arc_path, mode="w+", dtype=np.float32, shape=(n_rows, TRAJ_POINTS * 2))
    seq_tensor = np.lib.format.open_memmap(seq_path, mode="w+", dtype=np.float16, shape=(n_rows, SEQ_LEN, SEQ_CHANNELS))

    text_rows: list[str] = [""] * n_rows
    hash_rows: list[dict[str, Any]] = []
    base_batches: list[pd.DataFrame] = []
    row_offset = 0

    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        base_batches.append(extract_competition_base_features(df).reset_index(drop=True))
        records = df.to_dict("records")
        batch_hash_rows: list[dict[str, Any]] = []
        for i, row in enumerate(records):
            artifacts = _build_row_artifacts(row)
            idx = row_offset + i
            traj_view[idx] = artifacts["traj_view"]
            traj_geo[idx] = artifacts["traj_geo"]
            traj_arc[idx] = artifacts["traj_arc"]
            seq_tensor[idx] = artifacts["seq_tensor"]
            text_rows[idx] = artifacts["symbolic_text"]
            batch_hash_rows.append(artifacts["hashes"])
        hash_rows.extend(batch_hash_rows)
        row_offset += len(df)

    pd.concat(base_batches, axis=0, ignore_index=True).to_parquet(base_path, index=False)
    pd.DataFrame(hash_rows).to_parquet(hash_path, index=False)
    text_path.write_text("\n".join(text_rows), encoding="utf-8")
    traj_view.flush()
    traj_geo.flush()
    traj_arc.flush()
    seq_tensor.flush()
    _dump_json(meta_path, {"name": name, "n_rows": n_rows})
    del traj_view, traj_geo, traj_arc, seq_tensor
    gc.collect()

    return SplitCache(
        name=name,
        n_rows=n_rows,
        base_path=base_path,
        hash_path=hash_path,
        text_path=text_path,
        traj_view_path=traj_view_path,
        traj_geo_path=traj_geo_path,
        traj_arc_path=traj_arc_path,
        seq_path=seq_path,
    )


def load_memmap(path: Path, dtype: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.lib.format.open_memmap(path, mode="r", dtype=dtype, shape=shape)


def load_texts(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _combine_hash_counts(
    split_frames: dict[str, pd.DataFrame],
    hash_col: str,
) -> dict[str, Counter]:
    counters = {}
    for name, frame in split_frames.items():
        counters[name] = Counter(frame[hash_col].fillna("__NA__").astype(str).tolist())
    return counters


def _entropy_from_mean(p: float) -> float:
    if not np.isfinite(p):
        return np.nan
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def build_hash_feature_frames(
    train_hash: pd.DataFrame,
    unlabeled_hash: pd.DataFrame,
    test_hash: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hash_cols = list(train_hash.columns)
    split_frames = {"train": train_hash.copy(), "unlabeled": unlabeled_hash.copy(), "test": test_hash.copy()}
    all_counts: dict[str, dict[str, Counter]] = {}
    for hash_col in hash_cols:
        all_counts[hash_col] = _combine_hash_counts(split_frames, hash_col)

    def basic_counts(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for hash_col in hash_cols:
            values = frame[hash_col].fillna("__NA__").astype(str)
            counters = all_counts[hash_col]
            out[f"{hash_col}_count_all"] = values.map(
                lambda v: counters["train"][v] + counters["unlabeled"][v] + counters["test"][v]
            ).astype(float)
            out[f"{hash_col}_count_train"] = values.map(counters["train"]).astype(float)
            out[f"{hash_col}_count_unlabeled"] = values.map(counters["unlabeled"]).astype(float)
            out[f"{hash_col}_count_test"] = values.map(counters["test"]).astype(float)
        return out

    train_out = basic_counts(train_hash, "train")
    unlabeled_out = basic_counts(unlabeled_hash, "unlabeled")
    test_out = basic_counts(test_hash, "test")

    global_mean = float(np.mean(y))
    global_std = float(np.std(y))
    global_entropy = _entropy_from_mean(global_mean)

    for hash_col in hash_cols:
        values = train_hash[hash_col].fillna("__NA__").astype(str)
        oof_cols = {
            "human_count": np.zeros(len(train_hash), dtype=np.float32),
            "bot_count": np.zeros(len(train_hash), dtype=np.float32),
            "target_mean": np.zeros(len(train_hash), dtype=np.float32),
            "target_std": np.zeros(len(train_hash), dtype=np.float32),
            "label_entropy": np.zeros(len(train_hash), dtype=np.float32),
        }

        for tr_idx, val_idx in cv.split(train_hash, y):
            tr_values = values.iloc[tr_idx].tolist()
            tr_y = y[tr_idx]
            stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
            for key, target in zip(tr_values, tr_y):
                stats[key][0] += float(target)
                stats[key][1] += 1.0
                stats[key][2] += float(target) ** 2

            for idx in val_idx:
                key = values.iat[idx]
                if key not in stats:
                    oof_cols["human_count"][idx] = global_mean * len(tr_idx)
                    oof_cols["bot_count"][idx] = (1 - global_mean) * len(tr_idx)
                    oof_cols["target_mean"][idx] = global_mean
                    oof_cols["target_std"][idx] = global_std
                    oof_cols["label_entropy"][idx] = global_entropy
                    continue
                human_sum, count, sq_sum = stats[key]
                mean = human_sum / count if count else global_mean
                var = max(sq_sum / count - mean ** 2, 0.0) if count else global_std ** 2
                oof_cols["human_count"][idx] = human_sum
                oof_cols["bot_count"][idx] = count - human_sum
                oof_cols["target_mean"][idx] = mean
                oof_cols["target_std"][idx] = math.sqrt(var)
                oof_cols["label_entropy"][idx] = _entropy_from_mean(mean)

        for suffix, values_arr in oof_cols.items():
            train_out[f"{hash_col}_{suffix}_oof"] = values_arr

        full_stats: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        for key, target in zip(values.tolist(), y.tolist()):
            full_stats[key][0] += float(target)
            full_stats[key][1] += 1.0
            full_stats[key][2] += float(target) ** 2

        for frame_name, frame, out in [
            ("unlabeled", unlabeled_hash, unlabeled_out),
            ("test", test_hash, test_out),
        ]:
            frame_values = frame[hash_col].fillna("__NA__").astype(str)
            human_count = []
            bot_count = []
            target_mean = []
            target_std = []
            label_entropy = []
            for key in frame_values.tolist():
                if key not in full_stats:
                    human_count.append(global_mean * len(y))
                    bot_count.append((1 - global_mean) * len(y))
                    target_mean.append(global_mean)
                    target_std.append(global_std)
                    label_entropy.append(global_entropy)
                    continue
                human_sum, count, sq_sum = full_stats[key]
                mean = human_sum / count if count else global_mean
                var = max(sq_sum / count - mean ** 2, 0.0) if count else global_std ** 2
                human_count.append(human_sum)
                bot_count.append(count - human_sum)
                target_mean.append(mean)
                target_std.append(math.sqrt(var))
                label_entropy.append(_entropy_from_mean(mean))
            out[f"{hash_col}_human_count_train"] = human_count
            out[f"{hash_col}_bot_count_train"] = bot_count
            out[f"{hash_col}_target_mean_train"] = target_mean
            out[f"{hash_col}_target_std_train"] = target_std
            out[f"{hash_col}_label_entropy_train"] = label_entropy

    return train_out, unlabeled_out, test_out


def fit_text_reducer(
    train_text: list[str],
    unlabeled_text: list[str],
    test_text: list[str],
    max_features: int = 4000,
    n_components: int = 16,
) -> tuple[TfidfVectorizer, TruncatedSVD, np.ndarray, np.ndarray, np.ndarray]:
    texts = train_text + unlabeled_text + test_text
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(2, 4),
        max_features=max_features,
        min_df=3,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    reduced = svd.fit_transform(tfidf).astype(np.float32)
    n_train = len(train_text)
    n_unlabeled = len(unlabeled_text)
    train_arr = reduced[:n_train]
    unlabeled_arr = reduced[n_train : n_train + n_unlabeled]
    test_arr = reduced[n_train + n_unlabeled :]
    return vectorizer, svd, train_arr, unlabeled_arr, test_arr


def fit_dense_reducer(
    train_arr: np.ndarray,
    unlabeled_arr: np.ndarray,
    test_arr: np.ndarray,
    n_components: int,
) -> tuple[SimpleImputer, StandardScaler, PCA, np.ndarray, np.ndarray, np.ndarray]:
    all_arr = np.vstack([train_arr, unlabeled_arr, test_arr]).astype(np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    reduced_n = min(n_components, all_arr.shape[1], all_arr.shape[0] - 1)
    reduced_n = max(reduced_n, 2)
    pca = PCA(n_components=reduced_n, random_state=SEED, svd_solver="randomized")
    all_arr = imputer.fit_transform(all_arr)
    all_arr = scaler.fit_transform(all_arr)
    reduced = pca.fit_transform(all_arr).astype(np.float32)
    n_train = len(train_arr)
    n_unlabeled = len(unlabeled_arr)
    return (
        imputer,
        scaler,
        pca,
        reduced[:n_train],
        reduced[n_train : n_train + n_unlabeled],
        reduced[n_train + n_unlabeled :],
    )


class SequenceMemmapDataset(Dataset):
    def __init__(
        self,
        memmaps: list[np.ndarray],
        lengths: list[int],
        include_lengths: list[int] | None = None,
    ) -> None:
        self.memmaps = memmaps
        self.lengths = lengths
        self.cumulative = np.cumsum(lengths)
        self.include_lengths = include_lengths

    def __len__(self) -> int:
        return int(self.cumulative[-1]) if len(self.cumulative) else 0

    def _locate(self, idx: int) -> tuple[int, int]:
        file_idx = int(np.searchsorted(self.cumulative, idx, side="right"))
        start = 0 if file_idx == 0 else int(self.cumulative[file_idx - 1])
        return file_idx, idx - start

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, row_idx = self._locate(idx)
        arr = self.memmaps[file_idx][row_idx].astype(np.float32)
        return torch.from_numpy(arr)


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.gelu(x + residual)


class SSLTrajectoryEncoder(nn.Module):
    def __init__(self, in_channels: int = SEQ_CHANNELS, hidden: int = 64, embed_dim: int = SSL_EMBED_DIM) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResidualTCNBlock(hidden, dilation=1),
            ResidualTCNBlock(hidden, dilation=2),
            ResidualTCNBlock(hidden, dilation=4),
        )
        self.proj = nn.Linear(hidden * 2, embed_dim)
        self.reg_head = nn.Linear(hidden, 3)
        self.speed_head = nn.Linear(hidden, 7)
        self.dir_head = nn.Linear(hidden, 8)
        self.turn_head = nn.Linear(hidden, 8)
        self.next_state_head = nn.Linear(hidden, 8)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        h = self.blocks(h)
        avg = h.mean(dim=-1)
        mx = h.max(dim=-1).values
        embedding = self.proj(torch.cat([avg, mx], dim=1))
        return h, embedding

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h, embedding = self.encode(x)
        seq = h.transpose(1, 2)
        return {
            "hidden": h,
            "embedding": embedding,
            "reg": self.reg_head(seq),
            "speed_logits": self.speed_head(seq),
            "dir_logits": self.dir_head(seq),
            "turn_logits": self.turn_head(seq),
            "next_state_logits": self.next_state_head(seq),
        }


def _derive_ssl_targets(batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
    dx = batch[:, :, SEQ_IDX["dx"]]
    dy = batch[:, :, SEQ_IDX["dy"]]
    log_speed = batch[:, :, SEQ_IDX["log_speed"]]
    sin_angle = batch[:, :, SEQ_IDX["sin_angle"]]
    cos_angle = batch[:, :, SEQ_IDX["cos_angle"]]
    sin_turn = batch[:, :, SEQ_IDX["sin_turn"]]
    cos_turn = batch[:, :, SEQ_IDX["cos_turn"]]
    dt = torch.expm1(batch[:, :, SEQ_IDX["log_dt"]])
    dist = batch[:, :, SEQ_IDX["distance"]]
    src = batch[:, :, SEQ_IDX["is_touch"]].round().long()
    state = (src * 4 + (dt > 250).long() * 2 + (dist < 1e-3).long()).clamp(0, 7)

    angle = torch.atan2(sin_angle, cos_angle)
    turn = torch.atan2(sin_turn, cos_turn)
    direction_bin = torch.clamp(((angle + 2 * math.pi) % (2 * math.pi) / (2 * math.pi / 8)).floor().long(), 0, 7)
    turn_bin = torch.clamp(((turn + math.pi) % (2 * math.pi) / (2 * math.pi / 8)).floor().long(), 0, 7)
    speed_edges = torch.tensor([-1.5, -0.5, 0.2, 0.8, 1.5, 2.5], device=batch.device)
    speed_bin = torch.bucketize(log_speed, speed_edges)
    reg_target = batch[:, :, [SEQ_IDX["dx"], SEQ_IDX["dy"], SEQ_IDX["log_dt"]]]
    return reg_target, speed_bin, direction_bin, turn_bin, state


def _masked_ssl_loss(
    batch: torch.Tensor,
    model_out: dict[str, torch.Tensor],
    mask_rate: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = batch.device
    valid_mask = batch[:, :, SEQ_IDX["is_padding"]] < 0.5
    rand_mask = (torch.rand(valid_mask.shape, device=device) < mask_rate) & valid_mask

    reg_target, speed_bin, direction_bin, turn_bin, state = _derive_ssl_targets(batch)

    reg_loss = F.huber_loss(
        model_out["reg"][rand_mask],
        reg_target[rand_mask],
        reduction="mean",
        delta=0.2,
    ) if rand_mask.any() else torch.tensor(0.0, device=device)

    speed_loss = (
        F.cross_entropy(model_out["speed_logits"][rand_mask], speed_bin[rand_mask])
        if rand_mask.any()
        else torch.tensor(0.0, device=device)
    )
    dir_loss = (
        F.cross_entropy(model_out["dir_logits"][rand_mask], direction_bin[rand_mask])
        if rand_mask.any()
        else torch.tensor(0.0, device=device)
    )
    turn_loss = (
        F.cross_entropy(model_out["turn_logits"][rand_mask], turn_bin[rand_mask])
        if rand_mask.any()
        else torch.tensor(0.0, device=device)
    )
    masked_loss = reg_loss + speed_loss + dir_loss + turn_loss

    next_mask = valid_mask.clone()
    next_mask[:, -1] = False
    next_state = state[:, 1:]
    next_logits = model_out["next_state_logits"][:, :-1, :]
    next_valid = next_mask[:, :-1]
    next_loss = (
        F.cross_entropy(next_logits[next_valid], next_state[next_valid], reduction="mean")
        if next_valid.any()
        else torch.tensor(0.0, device=device)
    )
    total_loss = masked_loss + next_loss
    return total_loss, masked_loss.detach(), next_loss.detach()


def _per_sample_ssl_losses(batch: torch.Tensor, model_out: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    valid_mask = batch[:, :, SEQ_IDX["is_padding"]] < 0.5
    reg_target, speed_bin, direction_bin, turn_bin, state = _derive_ssl_targets(batch)

    reg_token = F.huber_loss(model_out["reg"], reg_target, reduction="none", delta=0.2).mean(dim=-1)
    speed_token = F.cross_entropy(
        model_out["speed_logits"].transpose(1, 2),
        speed_bin,
        reduction="none",
    )
    dir_token = F.cross_entropy(
        model_out["dir_logits"].transpose(1, 2),
        direction_bin,
        reduction="none",
    )
    turn_token = F.cross_entropy(
        model_out["turn_logits"].transpose(1, 2),
        turn_bin,
        reduction="none",
    )
    token_loss = reg_token + speed_token + dir_token + turn_token
    masked_like = (token_loss * valid_mask.float()).sum(dim=1) / valid_mask.float().sum(dim=1).clamp_min(1.0)

    next_logits = model_out["next_state_logits"][:, :-1, :].transpose(1, 2)
    next_state = state[:, 1:]
    next_valid = valid_mask[:, 1:]
    next_token_loss = F.cross_entropy(next_logits, next_state, reduction="none")
    next_loss = (next_token_loss * next_valid.float()).sum(dim=1) / next_valid.float().sum(dim=1).clamp_min(1.0)
    return masked_like.detach(), next_loss.detach()


def pretrain_ssl_encoder(
    train_seq: np.ndarray,
    unlabeled_seq: np.ndarray,
    model_path: Path,
    device: str = "cpu",
    batch_size: int = 256,
    epochs: int = 3,
    lr: float = 1e-3,
) -> SSLTrajectoryEncoder:
    device_t = torch.device(device)
    model = SSLTrajectoryEncoder().to(device_t)
    if model_path.exists():
        state = torch.load(model_path, map_location=device_t)
        model.load_state_dict(state)
        model.eval()
        return model

    dataset = SequenceMemmapDataset(
        memmaps=[train_seq, unlabeled_seq],
        lengths=[len(train_seq), len(unlabeled_seq)],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    for _ in range(epochs):
        for batch in loader:
            batch = batch.to(device_t).transpose(1, 2)
            masked_input = batch.clone()
            valid_mask = masked_input[:, SEQ_IDX["is_padding"], :] < 0.5
            random_mask = (torch.rand(valid_mask.shape, device=device_t) < 0.2) & valid_mask
            feature_mask = torch.ones_like(masked_input, dtype=torch.bool)
            feature_mask[:, SEQ_IDX["is_mouse"], :] = False
            feature_mask[:, SEQ_IDX["is_touch"], :] = False
            feature_mask[:, SEQ_IDX["is_padding"], :] = False
            masked_input = masked_input.masked_fill(random_mask.unsqueeze(1) & feature_mask, 0.0)

            optimizer.zero_grad(set_to_none=True)
            out = model(masked_input)
            total_loss, _, _ = _masked_ssl_loss(batch.transpose(1, 2), out)
            total_loss.backward()
            optimizer.step()

    model.eval()
    torch.save(model.state_dict(), model_path)
    return model


@torch.no_grad()
def infer_ssl_embeddings(
    model: SSLTrajectoryEncoder,
    seq_arr: np.ndarray,
    batch_size: int = 512,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device_t = torch.device(device)
    dataset = SequenceMemmapDataset(memmaps=[seq_arr], lengths=[len(seq_arr)])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeds = np.zeros((len(seq_arr), SSL_EMBED_DIM), dtype=np.float32)
    masked_losses = np.zeros(len(seq_arr), dtype=np.float32)
    next_losses = np.zeros(len(seq_arr), dtype=np.float32)
    offset = 0
    model.eval()

    for batch in loader:
        batch = batch.to(device_t)
        out = model(batch.transpose(1, 2))
        masked_loss, next_loss = _per_sample_ssl_losses(batch, out)
        emb = out["embedding"].cpu().numpy().astype(np.float32)
        n = len(emb)
        embeds[offset : offset + n] = emb
        masked_losses[offset : offset + n] = masked_loss.cpu().numpy().astype(np.float32)
        next_losses[offset : offset + n] = next_loss.cpu().numpy().astype(np.float32)
        offset += n
    return embeds, masked_losses, next_losses


def build_cluster_features(
    train_space: np.ndarray,
    unlabeled_space: np.ndarray,
    test_space: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
    n_clusters_list: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if n_clusters_list is None:
        n_clusters_list = [50, 100]

    out_train = pd.DataFrame(index=np.arange(len(train_space)))
    out_unlabeled = pd.DataFrame(index=np.arange(len(unlabeled_space)))
    out_test = pd.DataFrame(index=np.arange(len(test_space)))

    combined = np.vstack([train_space, unlabeled_space, test_space]).astype(np.float32)
    n_train = len(train_space)
    n_unlabeled = len(unlabeled_space)

    for n_clusters in n_clusters_list:
        model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, random_state=SEED, n_init="auto")
        labels_all = model.fit_predict(combined)
        dist_all = np.linalg.norm(combined - model.cluster_centers_[labels_all], axis=1)

        train_labels = labels_all[:n_train]
        unlabeled_labels = labels_all[n_train : n_train + n_unlabeled]
        test_labels = labels_all[n_train + n_unlabeled :]
        train_dist = dist_all[:n_train]
        unlabeled_dist = dist_all[n_train : n_train + n_unlabeled]
        test_dist = dist_all[n_train + n_unlabeled :]

        counts_all = Counter(labels_all.tolist())
        counts_train = Counter(train_labels.tolist())
        counts_unlabeled = Counter(unlabeled_labels.tolist())
        counts_test = Counter(test_labels.tolist())

        prefix = f"cluster_{n_clusters}"
        for frame, labels, dist in [
            (out_train, train_labels, train_dist),
            (out_unlabeled, unlabeled_labels, unlabeled_dist),
            (out_test, test_labels, test_dist),
        ]:
            frame[f"{prefix}_id"] = labels.astype(np.float32)
            frame[f"{prefix}_size_all"] = [counts_all[int(x)] for x in labels]
            frame[f"{prefix}_size_train"] = [counts_train[int(x)] for x in labels]
            frame[f"{prefix}_size_unlabeled"] = [counts_unlabeled[int(x)] for x in labels]
            frame[f"{prefix}_size_test"] = [counts_test[int(x)] for x in labels]
            frame[f"{prefix}_distance_to_center"] = dist.astype(np.float32)
            frame[f"{prefix}_density_proxy"] = 1.0 / (dist + 1e-3)

        global_mean = float(np.mean(y))
        global_std = float(np.std(y))
        global_entropy = _entropy_from_mean(global_mean)
        oof_mean = np.zeros(n_train, dtype=np.float32)
        oof_std = np.zeros(n_train, dtype=np.float32)
        oof_entropy = np.zeros(n_train, dtype=np.float32)
        oof_human = np.zeros(n_train, dtype=np.float32)
        oof_bot = np.zeros(n_train, dtype=np.float32)

        train_labels_series = pd.Series(train_labels)
        for tr_idx, val_idx in cv.split(train_space, y):
            tr_lab = train_labels_series.iloc[tr_idx].tolist()
            tr_y = y[tr_idx]
            stats: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
            for cluster_id, target in zip(tr_lab, tr_y):
                stats[int(cluster_id)][0] += float(target)
                stats[int(cluster_id)][1] += 1.0
                stats[int(cluster_id)][2] += float(target) ** 2
            for idx in val_idx:
                cluster_id = int(train_labels_series.iat[idx])
                if cluster_id not in stats:
                    oof_mean[idx] = global_mean
                    oof_std[idx] = global_std
                    oof_entropy[idx] = global_entropy
                    oof_human[idx] = global_mean * len(tr_idx)
                    oof_bot[idx] = (1 - global_mean) * len(tr_idx)
                    continue
                human_sum, count, sq_sum = stats[cluster_id]
                mean = human_sum / count if count else global_mean
                var = max(sq_sum / count - mean ** 2, 0.0) if count else global_std ** 2
                oof_mean[idx] = mean
                oof_std[idx] = math.sqrt(var)
                oof_entropy[idx] = _entropy_from_mean(mean)
                oof_human[idx] = human_sum
                oof_bot[idx] = count - human_sum

        out_train[f"{prefix}_target_mean_oof"] = oof_mean
        out_train[f"{prefix}_target_std_oof"] = oof_std
        out_train[f"{prefix}_label_entropy_oof"] = oof_entropy
        out_train[f"{prefix}_human_count_oof"] = oof_human
        out_train[f"{prefix}_bot_count_oof"] = oof_bot

        full_stats: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        for cluster_id, target in zip(train_labels.tolist(), y.tolist()):
            full_stats[int(cluster_id)][0] += float(target)
            full_stats[int(cluster_id)][1] += 1.0
            full_stats[int(cluster_id)][2] += float(target) ** 2

        for frame, labels, tag in [
            (out_unlabeled, unlabeled_labels, "train"),
            (out_test, test_labels, "train"),
        ]:
            means, stds, ents, humans, bots = [], [], [], [], []
            for cluster_id in labels.tolist():
                if int(cluster_id) not in full_stats:
                    means.append(global_mean)
                    stds.append(global_std)
                    ents.append(global_entropy)
                    humans.append(global_mean * len(y))
                    bots.append((1 - global_mean) * len(y))
                    continue
                human_sum, count, sq_sum = full_stats[int(cluster_id)]
                mean = human_sum / count if count else global_mean
                var = max(sq_sum / count - mean ** 2, 0.0) if count else global_std ** 2
                means.append(mean)
                stds.append(math.sqrt(var))
                ents.append(_entropy_from_mean(mean))
                humans.append(human_sum)
                bots.append(count - human_sum)
            frame[f"{prefix}_target_mean_train"] = means
            frame[f"{prefix}_target_std_train"] = stds
            frame[f"{prefix}_label_entropy_train"] = ents
            frame[f"{prefix}_human_count_train"] = humans
            frame[f"{prefix}_bot_count_train"] = bots

    return out_train, out_unlabeled, out_test


def _query_self_knn(space: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    index = NNDescent(space, n_neighbors=max(k + 5, 20), metric="euclidean", random_state=SEED)
    indices, distances = index.query(space, k=k + 1)
    return indices[:, 1:], distances[:, 1:]


def _query_knn(reference: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    index = NNDescent(reference, n_neighbors=max(k + 5, 20), metric="euclidean", random_state=SEED)
    indices, distances = index.query(query, k=k)
    return indices, distances


def build_knn_features(
    train_space: np.ndarray,
    unlabeled_space: np.ndarray,
    test_space: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out_train = pd.DataFrame(index=np.arange(len(train_space)))
    out_unlabeled = pd.DataFrame(index=np.arange(len(unlabeled_space)))
    out_test = pd.DataFrame(index=np.arange(len(test_space)))

    combined = np.vstack([train_space, unlabeled_space, test_space]).astype(np.float32)
    n_train = len(train_space)
    n_unlabeled = len(unlabeled_space)
    split_ranges = {
        "train": np.arange(0, n_train),
        "unlabeled": np.arange(n_train, n_train + n_unlabeled),
        "test": np.arange(n_train + n_unlabeled, len(combined)),
    }

    _, all_dist = _query_self_knn(combined, k=20)
    knn_source = {
        "train": pd.DataFrame(index=np.arange(n_train)),
        "unlabeled": pd.DataFrame(index=np.arange(n_unlabeled)),
        "test": pd.DataFrame(index=np.arange(len(test_space))),
    }
    split_dists = {
        "train": all_dist[split_ranges["train"]],
        "unlabeled": all_dist[split_ranges["unlabeled"]],
        "test": all_dist[split_ranges["test"]],
    }
    for tag, frame in knn_source.items():
        dist = split_dists[tag]
        frame["knn_dist_1"] = dist[:, 0]
        frame["knn_dist_3_mean"] = dist[:, :3].mean(axis=1)
        frame["knn_dist_5_mean"] = dist[:, :5].mean(axis=1)
        frame["knn_dist_10_mean"] = dist[:, :10].mean(axis=1)
        frame["knn_dist_20_mean"] = dist[:, :20].mean(axis=1)
        frame["knn_dist_std_10"] = dist[:, :10].std(axis=1)

    out_train = pd.concat([out_train, knn_source["train"]], axis=1)
    out_unlabeled = pd.concat([out_unlabeled, knn_source["unlabeled"]], axis=1)
    out_test = pd.concat([out_test, knn_source["test"]], axis=1)

    _, train_dist = _query_self_knn(train_space, k=10)
    _, unlabeled_self_dist = _query_self_knn(unlabeled_space, k=1)
    _, test_self_dist = _query_self_knn(test_space, k=1)
    train_to_unlabeled_dist = _query_knn(unlabeled_space, train_space, k=1)[1][:, 0]
    test_to_unlabeled_dist = _query_knn(unlabeled_space, test_space, k=1)[1][:, 0]
    train_to_test_dist = _query_knn(test_space, train_space, k=1)[1][:, 0]
    unlabeled_to_test_dist = _query_knn(test_space, unlabeled_space, k=1)[1][:, 0]
    train_to_train_other = train_dist[:, 0]
    others_to_train = _query_knn(train_space, combined[n_train:], k=1)[1][:, 0]
    out_train["knn_train_dist_min"] = train_to_train_other
    out_unlabeled["knn_train_dist_min"] = others_to_train[:n_unlabeled]
    out_test["knn_train_dist_min"] = others_to_train[n_unlabeled:]
    out_train["knn_unlabeled_dist_min"] = train_to_unlabeled_dist
    out_unlabeled["knn_unlabeled_dist_min"] = unlabeled_self_dist[:, 0]
    out_test["knn_unlabeled_dist_min"] = test_to_unlabeled_dist
    out_train["knn_test_dist_min"] = train_to_test_dist
    out_unlabeled["knn_test_dist_min"] = unlabeled_to_test_dist
    out_test["knn_test_dist_min"] = test_self_dist[:, 0]

    oof_label_mean_3 = np.zeros(n_train, dtype=np.float32)
    oof_label_mean_5 = np.zeros(n_train, dtype=np.float32)
    oof_label_mean_10 = np.zeros(n_train, dtype=np.float32)
    oof_label_std_10 = np.zeros(n_train, dtype=np.float32)
    oof_bot_dist_min = np.zeros(n_train, dtype=np.float32)
    oof_human_dist_min = np.zeros(n_train, dtype=np.float32)

    for tr_idx, val_idx in cv.split(train_space, y):
        tr_space = train_space[tr_idx]
        val_space = train_space[val_idx]
        tr_y = y[tr_idx]
        idx, dist = _query_knn(tr_space, val_space, k=min(10, len(tr_space)))
        neigh_y = tr_y[idx]
        oof_label_mean_3[val_idx] = neigh_y[:, : min(3, neigh_y.shape[1])].mean(axis=1)
        oof_label_mean_5[val_idx] = neigh_y[:, : min(5, neigh_y.shape[1])].mean(axis=1)
        oof_label_mean_10[val_idx] = neigh_y.mean(axis=1)
        oof_label_std_10[val_idx] = neigh_y.std(axis=1)
        bot_dist = np.full(len(val_idx), np.nan, dtype=np.float32)
        human_dist = np.full(len(val_idx), np.nan, dtype=np.float32)
        for i in range(len(val_idx)):
            d = dist[i]
            yy = neigh_y[i]
            bot_pos = np.where(yy < 0.5)[0]
            human_pos = np.where(yy >= 0.5)[0]
            if len(bot_pos):
                bot_dist[i] = d[bot_pos[0]]
            if len(human_pos):
                human_dist[i] = d[human_pos[0]]
        oof_bot_dist_min[val_idx] = np.nan_to_num(bot_dist, nan=float(np.nanmax(dist) if dist.size else 1.0))
        oof_human_dist_min[val_idx] = np.nan_to_num(human_dist, nan=float(np.nanmax(dist) if dist.size else 1.0))

    out_train["knn_train_label_mean_3"] = oof_label_mean_3
    out_train["knn_train_label_mean_5"] = oof_label_mean_5
    out_train["knn_train_label_mean_10"] = oof_label_mean_10
    out_train["knn_train_label_std_10"] = oof_label_std_10
    out_train["knn_train_bot_dist_min"] = oof_bot_dist_min
    out_train["knn_train_human_dist_min"] = oof_human_dist_min
    out_train["knn_bot_human_dist_ratio"] = oof_bot_dist_min / (oof_human_dist_min + 1e-3)

    unl_train_idx, unl_train_dist = _query_knn(train_space, unlabeled_space, k=min(10, len(train_space)))
    neigh_y = y[unl_train_idx]
    out_unlabeled["knn_train_label_mean_3"] = neigh_y[:, : min(3, neigh_y.shape[1])].mean(axis=1)
    out_unlabeled["knn_train_label_mean_5"] = neigh_y[:, : min(5, neigh_y.shape[1])].mean(axis=1)
    out_unlabeled["knn_train_label_mean_10"] = neigh_y.mean(axis=1)
    out_unlabeled["knn_train_label_std_10"] = neigh_y.std(axis=1)

    test_train_idx, test_train_dist = _query_knn(train_space, test_space, k=min(10, len(train_space)))
    neigh_y = y[test_train_idx]
    out_test["knn_train_label_mean_3"] = neigh_y[:, : min(3, neigh_y.shape[1])].mean(axis=1)
    out_test["knn_train_label_mean_5"] = neigh_y[:, : min(5, neigh_y.shape[1])].mean(axis=1)
    out_test["knn_train_label_mean_10"] = neigh_y.mean(axis=1)
    out_test["knn_train_label_std_10"] = neigh_y.std(axis=1)

    for out_frame, idx_arr, dist_arr in [
        (out_unlabeled, unl_train_idx, unl_train_dist),
        (out_test, test_train_idx, test_train_dist),
    ]:
        bot_dist = []
        human_dist = []
        for neigh_idx, neigh_dist in zip(idx_arr, dist_arr):
            yy = y[neigh_idx]
            bot_pos = np.where(yy < 0.5)[0]
            human_pos = np.where(yy >= 0.5)[0]
            bot_dist.append(float(neigh_dist[bot_pos[0]]) if len(bot_pos) else float(neigh_dist[-1]))
            human_dist.append(float(neigh_dist[human_pos[0]]) if len(human_pos) else float(neigh_dist[-1]))
        out_frame["knn_train_bot_dist_min"] = bot_dist
        out_frame["knn_train_human_dist_min"] = human_dist
        out_frame["knn_bot_human_dist_ratio"] = out_frame["knn_train_bot_dist_min"] / (
            out_frame["knn_train_human_dist_min"] + 1e-3
        )

    return out_train.reset_index(drop=True), out_unlabeled.reset_index(drop=True), out_test.reset_index(drop=True)


@dataclass
class FeatureBundle:
    train_base: pd.DataFrame
    test_base: pd.DataFrame
    train_template: pd.DataFrame
    test_template: pd.DataFrame
    train_ssl: pd.DataFrame
    test_ssl: pd.DataFrame
    train_hash: pd.DataFrame
    train_text_reduced: pd.DataFrame
    test_text_reduced: pd.DataFrame


def _feature_family_columns(
    train_base: pd.DataFrame,
    train_template: pd.DataFrame,
    train_ssl: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, list[str]]:
    full_maps = fit_te_maps(train_base, y)
    manual = apply_te_maps(train_base, full_maps)
    template = pd.concat([manual.reset_index(drop=True), train_template.reset_index(drop=True)], axis=1)
    full = pd.concat([template.reset_index(drop=True), train_ssl.reset_index(drop=True)], axis=1)
    return {
        "manual": choose_usable_columns(manual),
        "template": choose_usable_columns(template),
        "full": choose_usable_columns(full),
        "ssl": choose_usable_columns(pd.concat([manual.reset_index(drop=True), train_ssl.reset_index(drop=True)], axis=1)),
    }


def _build_feature_family(
    family: str,
    train_base: pd.DataFrame,
    other_train: dict[str, pd.DataFrame],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    y: np.ndarray,
    columns_map: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_tr = train_base.iloc[train_idx].reset_index(drop=True)
    base_val = train_base.iloc[val_idx].reset_index(drop=True)
    maps = fit_te_maps(base_tr, y[train_idx])
    manual_tr = apply_te_maps(base_tr, maps)
    manual_val = apply_te_maps(base_val, maps)

    if family == "manual":
        return (
            manual_tr.reindex(columns=columns_map["manual"]),
            manual_val.reindex(columns=columns_map["manual"]),
        )

    if family == "template":
        X_tr = pd.concat([manual_tr, other_train["template"].iloc[train_idx].reset_index(drop=True)], axis=1)
        X_val = pd.concat([manual_val, other_train["template"].iloc[val_idx].reset_index(drop=True)], axis=1)
        return X_tr.reindex(columns=columns_map["template"]), X_val.reindex(columns=columns_map["template"])

    if family == "full":
        X_tr = pd.concat(
            [
                manual_tr,
                other_train["template"].iloc[train_idx].reset_index(drop=True),
                other_train["ssl"].iloc[train_idx].reset_index(drop=True),
            ],
            axis=1,
        )
        X_val = pd.concat(
            [
                manual_val,
                other_train["template"].iloc[val_idx].reset_index(drop=True),
                other_train["ssl"].iloc[val_idx].reset_index(drop=True),
            ],
            axis=1,
        )
        return X_tr.reindex(columns=columns_map["full"]), X_val.reindex(columns=columns_map["full"])

    if family == "ssl":
        X_tr = pd.concat([manual_tr, other_train["ssl"].iloc[train_idx].reset_index(drop=True)], axis=1)
        X_val = pd.concat([manual_val, other_train["ssl"].iloc[val_idx].reset_index(drop=True)], axis=1)
        return X_tr.reindex(columns=columns_map["ssl"]), X_val.reindex(columns=columns_map["ssl"])

    raise ValueError(f"Unknown family: {family}")


def _build_test_family(
    family: str,
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    other_train: dict[str, pd.DataFrame],
    other_test: dict[str, pd.DataFrame],
    y: np.ndarray,
    columns_map: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_maps = fit_te_maps(train_base, y)
    manual_train = apply_te_maps(train_base, full_maps)
    manual_test = apply_te_maps(test_base, full_maps)

    if family == "manual":
        return manual_train.reindex(columns=columns_map["manual"]), manual_test.reindex(columns=columns_map["manual"])

    if family == "template":
        X_tr = pd.concat([manual_train, other_train["template"].reset_index(drop=True)], axis=1)
        X_te = pd.concat([manual_test, other_test["template"].reset_index(drop=True)], axis=1)
        return X_tr.reindex(columns=columns_map["template"]), X_te.reindex(columns=columns_map["template"])

    if family == "full":
        X_tr = pd.concat([manual_train, other_train["template"], other_train["ssl"]], axis=1)
        X_te = pd.concat([manual_test, other_test["template"], other_test["ssl"]], axis=1)
        return X_tr.reindex(columns=columns_map["full"]), X_te.reindex(columns=columns_map["full"])

    if family == "ssl":
        X_tr = pd.concat([manual_train, other_train["ssl"]], axis=1)
        X_te = pd.concat([manual_test, other_test["ssl"]], axis=1)
        return X_tr.reindex(columns=columns_map["ssl"]), X_te.reindex(columns=columns_map["ssl"])

    raise ValueError(f"Unknown family: {family}")


@dataclass
class ModelSpec:
    name: str
    family: str
    builder: Callable[[], Any]


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logit_selected",
            family="full",
            builder=lambda: make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=0.3, max_iter=4000),
            ),
        ),
        ModelSpec(
            name="lightgbm_full",
            family="full",
            builder=lambda: lgb.LGBMClassifier(
                n_estimators=900,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_lambda=3.0,
                reg_alpha=0.5,
                objective="binary",
                class_weight="balanced",
                random_state=SEED,
                n_jobs=-1,
                verbosity=-1,
            ),
        ),
        ModelSpec(
            name="catboost_manual",
            family="manual",
            builder=lambda: CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                depth=4,
                learning_rate=0.03,
                l2_leaf_reg=12.0,
                iterations=1600,
                random_strength=1.5,
                auto_class_weights="Balanced",
                verbose=False,
                allow_writing_files=False,
                random_seed=SEED,
            ),
        ),
        ModelSpec(
            name="catboost_template",
            family="template",
            builder=lambda: CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                depth=5,
                learning_rate=0.03,
                l2_leaf_reg=14.0,
                iterations=1800,
                random_strength=2.0,
                auto_class_weights="Balanced",
                verbose=False,
                allow_writing_files=False,
                random_seed=SEED + 1,
            ),
        ),
        ModelSpec(
            name="catboost_full",
            family="full",
            builder=lambda: CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="AUC",
                depth=5,
                learning_rate=0.025,
                l2_leaf_reg=16.0,
                iterations=2200,
                random_strength=2.5,
                auto_class_weights="Balanced",
                verbose=False,
                allow_writing_files=False,
                random_seed=SEED + 2,
            ),
        ),
    ]


def _fit_predict_model(
    model: Any,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> np.ndarray:
    if isinstance(model, CatBoostClassifier):
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True, verbose=False)
        return model.predict_proba(X_val)[:, 1]
    if isinstance(model, lgb.LGBMClassifier):
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return model.predict_proba(X_val)[:, 1]
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_val)[:, 1]


def _fit_predict_test(model: Any, X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame) -> np.ndarray:
    if isinstance(model, CatBoostClassifier):
        model.fit(X_tr, y_tr, verbose=False)
        return model.predict_proba(X_te)[:, 1]
    if isinstance(model, lgb.LGBMClassifier):
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1]
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def weighted_rank_average(preds: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    ranked = []
    total = 0.0
    for name, pred in preds.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        ranked.append(pd.Series(pred).rank(method="average", pct=True).to_numpy(dtype=float) * weight)
        total += weight
    if total <= 0:
        raise ValueError("Rank-average weights must sum to a positive value")
    return np.sum(ranked, axis=0) / total


def train_spec_ensemble(
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    train_template: pd.DataFrame,
    test_template: pd.DataFrame,
    train_ssl: pd.DataFrame,
    test_ssl: pd.DataFrame,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> dict[str, Any]:
    columns_map = _feature_family_columns(train_base, train_template, train_ssl, y)
    other_train = {"template": train_template, "ssl": train_ssl}
    other_test = {"template": test_template, "ssl": test_ssl}

    model_specs = build_model_specs()
    oof_preds: dict[str, np.ndarray] = {spec.name: np.zeros(len(train_base), dtype=np.float32) for spec in model_specs}
    test_preds: dict[str, np.ndarray] = {}
    model_scores: dict[str, dict[str, float]] = {}

    for spec in model_specs:
        for tr_idx, val_idx in cv.split(train_base, y):
            X_tr, X_val = _build_feature_family(
                family=spec.family,
                train_base=train_base,
                other_train=other_train,
                train_idx=tr_idx,
                val_idx=val_idx,
                y=y,
                columns_map=columns_map,
            )
            model = spec.builder()
            oof_preds[spec.name][val_idx] = _fit_predict_model(model, X_tr, y[tr_idx], X_val, y[val_idx])

        X_train_full, X_test_full = _build_test_family(
            family=spec.family,
            train_base=train_base,
            test_base=test_base,
            other_train=other_train,
            other_test=other_test,
            y=y,
            columns_map=columns_map,
        )
        final_model = spec.builder()
        test_preds[spec.name] = _fit_predict_test(final_model, X_train_full, y, X_test_full)
        model_scores[spec.name] = {
            "pauc_01": partial_auc_score(y, oof_preds[spec.name], max_fpr=0.1),
            "pauc_0035": partial_auc_score(y, oof_preds[spec.name], max_fpr=0.035),
            "logloss": float(log_loss(y, np.clip(oof_preds[spec.name], 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y, np.clip(oof_preds[spec.name], 1e-6, 1 - 1e-6))),
        }

    weight_raw = {name: max(scores["pauc_01"], 1e-6) for name, scores in model_scores.items()}
    weight_sum = sum(weight_raw.values())
    weights = {name: value / weight_sum for name, value in weight_raw.items()}

    ensemble_oof = weighted_rank_average(oof_preds, weights)
    ensemble_test = weighted_rank_average(test_preds, weights)
    return {
        "oof_predictions": oof_preds,
        "test_predictions": test_preds,
        "weights": weights,
        "model_scores": model_scores,
        "ensemble_oof": ensemble_oof,
        "ensemble_test": ensemble_test,
        "ensemble_scores": {
            "pauc_01": partial_auc_score(y, ensemble_oof, max_fpr=0.1),
            "pauc_0035": partial_auc_score(y, ensemble_oof, max_fpr=0.035),
            "logloss": float(log_loss(y, np.clip(ensemble_oof, 1e-6, 1 - 1e-6))),
            "brier": float(brier_score_loss(y, np.clip(ensemble_oof, 1e-6, 1 - 1e-6))),
        },
    }


def load_split_base(cache: SplitCache) -> pd.DataFrame:
    return pd.read_parquet(cache.base_path)


def load_split_hash(cache: SplitCache) -> pd.DataFrame:
    return pd.read_parquet(cache.hash_path)


def build_template_and_ssl_features(
    train_cache: SplitCache,
    unlabeled_cache: SplitCache,
    test_cache: SplitCache,
    y: np.ndarray,
    cache_dir: Path,
    device: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_base = load_split_base(train_cache)
    unlabeled_base = load_split_base(unlabeled_cache)
    test_base = load_split_base(test_cache)
    train_hash = load_split_hash(train_cache)
    unlabeled_hash = load_split_hash(unlabeled_cache)
    test_hash = load_split_hash(test_cache)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    use_cuda = str(device).startswith("cuda")
    ssl_batch_size = 1024 if use_cuda else 256
    infer_batch_size = 2048 if use_cuda else 512

    train_hash_feats, unlabeled_hash_feats, test_hash_feats = build_hash_feature_frames(
        train_hash=train_hash,
        unlabeled_hash=unlabeled_hash,
        test_hash=test_hash,
        y=y,
        cv=cv,
    )

    train_text = load_texts(train_cache.text_path)
    unlabeled_text = load_texts(unlabeled_cache.text_path)
    test_text = load_texts(test_cache.text_path)
    _, _, train_text_red, unlabeled_text_red, test_text_red = fit_text_reducer(
        train_text=train_text,
        unlabeled_text=unlabeled_text,
        test_text=test_text,
    )

    train_traj = np.hstack(
        [
            load_memmap(train_cache.traj_view_path, np.float32, (train_cache.n_rows, TRAJ_POINTS * 2)),
            load_memmap(train_cache.traj_geo_path, np.float32, (train_cache.n_rows, TRAJ_POINTS * 2)),
            load_memmap(train_cache.traj_arc_path, np.float32, (train_cache.n_rows, TRAJ_POINTS * 2)),
        ]
    )
    unlabeled_traj = np.hstack(
        [
            load_memmap(unlabeled_cache.traj_view_path, np.float32, (unlabeled_cache.n_rows, TRAJ_POINTS * 2)),
            load_memmap(unlabeled_cache.traj_geo_path, np.float32, (unlabeled_cache.n_rows, TRAJ_POINTS * 2)),
            load_memmap(unlabeled_cache.traj_arc_path, np.float32, (unlabeled_cache.n_rows, TRAJ_POINTS * 2)),
        ]
    )
    test_traj = np.hstack(
        [
            load_memmap(test_cache.traj_view_path, np.float32, (test_cache.n_rows, TRAJ_POINTS * 2)),
            load_memmap(test_cache.traj_geo_path, np.float32, (test_cache.n_rows, TRAJ_POINTS * 2)),
            load_memmap(test_cache.traj_arc_path, np.float32, (test_cache.n_rows, TRAJ_POINTS * 2)),
        ]
    )
    _, _, _, train_traj_red, unlabeled_traj_red, test_traj_red = fit_dense_reducer(
        train_traj,
        unlabeled_traj,
        test_traj,
        n_components=16,
    )
    _, _, _, train_manual_red, unlabeled_manual_red, test_manual_red = fit_dense_reducer(
        train_base.to_numpy(dtype=np.float32),
        unlabeled_base.to_numpy(dtype=np.float32),
        test_base.to_numpy(dtype=np.float32),
        n_components=16,
    )

    train_seq = load_memmap(train_cache.seq_path, np.float16, (train_cache.n_rows, SEQ_LEN, SEQ_CHANNELS))
    unlabeled_seq = load_memmap(unlabeled_cache.seq_path, np.float16, (unlabeled_cache.n_rows, SEQ_LEN, SEQ_CHANNELS))
    test_seq = load_memmap(test_cache.seq_path, np.float16, (test_cache.n_rows, SEQ_LEN, SEQ_CHANNELS))
    ssl_model = pretrain_ssl_encoder(
        train_seq=train_seq,
        unlabeled_seq=unlabeled_seq,
        model_path=cache_dir / "ssl_encoder.pt",
        device=device,
        batch_size=ssl_batch_size,
    )
    train_emb, train_mask_loss, train_next_loss = infer_ssl_embeddings(
        ssl_model,
        train_seq,
        batch_size=infer_batch_size,
        device=device,
    )
    unlabeled_emb, _, _ = infer_ssl_embeddings(
        ssl_model,
        unlabeled_seq,
        batch_size=infer_batch_size,
        device=device,
    )
    test_emb, test_mask_loss, test_next_loss = infer_ssl_embeddings(
        ssl_model,
        test_seq,
        batch_size=infer_batch_size,
        device=device,
    )

    _, _, _, train_ssl_red, unlabeled_ssl_red, test_ssl_red = fit_dense_reducer(
        train_emb,
        unlabeled_emb,
        test_emb,
        n_components=16,
    )

    duplicate_space_train = np.hstack([train_manual_red, train_traj_red, train_text_red, train_ssl_red]).astype(np.float32)
    duplicate_space_unlabeled = np.hstack(
        [unlabeled_manual_red, unlabeled_traj_red, unlabeled_text_red, unlabeled_ssl_red]
    ).astype(np.float32)
    duplicate_space_test = np.hstack([test_manual_red, test_traj_red, test_text_red, test_ssl_red]).astype(np.float32)

    train_knn, unlabeled_knn, test_knn = build_knn_features(
        train_space=duplicate_space_train,
        unlabeled_space=duplicate_space_unlabeled,
        test_space=duplicate_space_test,
        y=y,
        cv=cv,
    )
    train_cluster, unlabeled_cluster, test_cluster = build_cluster_features(
        train_space=duplicate_space_train,
        unlabeled_space=duplicate_space_unlabeled,
        test_space=duplicate_space_test,
        y=y,
        cv=cv,
        n_clusters_list=[50, 100],
    )

    train_template = pd.concat(
        [
            train_hash_feats.reset_index(drop=True),
            pd.DataFrame(train_text_red, columns=[f"text_svd_{i}" for i in range(train_text_red.shape[1])]),
            pd.DataFrame(train_traj_red, columns=[f"traj_pca_{i}" for i in range(train_traj_red.shape[1])]),
            train_knn.reset_index(drop=True),
            train_cluster.reset_index(drop=True),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)

    test_template = pd.concat(
        [
            test_hash_feats.reset_index(drop=True),
            pd.DataFrame(test_text_red, columns=[f"text_svd_{i}" for i in range(test_text_red.shape[1])]),
            pd.DataFrame(test_traj_red, columns=[f"traj_pca_{i}" for i in range(test_traj_red.shape[1])]),
            test_knn.reset_index(drop=True),
            test_cluster.reset_index(drop=True),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)

    unlabeled_template = pd.concat(
        [
            unlabeled_hash_feats.reset_index(drop=True),
            pd.DataFrame(unlabeled_text_red, columns=[f"text_svd_{i}" for i in range(unlabeled_text_red.shape[1])]),
            pd.DataFrame(unlabeled_traj_red, columns=[f"traj_pca_{i}" for i in range(unlabeled_traj_red.shape[1])]),
            unlabeled_knn.reset_index(drop=True),
            unlabeled_cluster.reset_index(drop=True),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)

    train_ssl = pd.concat(
        [
            pd.DataFrame(train_emb, columns=[f"ssl_embedding_{i}" for i in range(train_emb.shape[1])]),
            pd.DataFrame(train_ssl_red, columns=[f"ssl_pca_{i}" for i in range(train_ssl_red.shape[1])]),
            pd.DataFrame(
                {
                    "masked_reconstruction_loss": train_mask_loss,
                    "next_state_prediction_loss": train_next_loss,
                }
            ),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)

    test_ssl = pd.concat(
        [
            pd.DataFrame(test_emb, columns=[f"ssl_embedding_{i}" for i in range(test_emb.shape[1])]),
            pd.DataFrame(test_ssl_red, columns=[f"ssl_pca_{i}" for i in range(test_ssl_red.shape[1])]),
            pd.DataFrame(
                {
                    "masked_reconstruction_loss": test_mask_loss,
                    "next_state_prediction_loss": test_next_loss,
                }
            ),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)

    unlabeled_ssl = pd.concat(
        [
            pd.DataFrame(unlabeled_emb, columns=[f"ssl_embedding_{i}" for i in range(unlabeled_emb.shape[1])]),
            pd.DataFrame(unlabeled_ssl_red, columns=[f"ssl_pca_{i}" for i in range(unlabeled_ssl_red.shape[1])]),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan)

    return train_template, test_template, unlabeled_template, train_ssl, test_ssl, unlabeled_ssl


def save_submission(pred: np.ndarray, out_path: Path) -> None:
    pd.DataFrame({"id": np.arange(len(pred)), "probability": pred}).to_csv(out_path, index=False)


def run_full_complex_pipeline(
    train_path: str | Path = "data/train.parquet",
    test_path: str | Path = "data/test.parquet",
    unlabeled_path: str | Path = "data/unlabelled.parquet",
    cache_dir: str | Path = "outputs/complex_cache",
    submission_path: str | Path = "outputs/submit_complex.csv",
    metrics_path: str | Path = "outputs/complex_metrics.json",
    device: str = "cpu",
) -> dict[str, Any]:
    set_seed(SEED)
    train_path = Path(train_path)
    test_path = Path(test_path)
    unlabeled_path = Path(unlabeled_path)
    cache_dir = Path(cache_dir)
    submission_path = Path(submission_path)
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    train_cache = prepare_split_cache("train", train_path, cache_dir)
    test_cache = prepare_split_cache("test", test_path, cache_dir)
    unlabeled_cache = prepare_split_cache("unlabeled", unlabeled_path, cache_dir)

    train_df = pd.read_parquet(train_path, columns=["target"])
    y = train_df["target"].astype(int).to_numpy()
    train_base = load_split_base(train_cache)
    test_base = load_split_base(test_cache)

    train_template, test_template, _, train_ssl, test_ssl, _ = build_template_and_ssl_features(
        train_cache=train_cache,
        unlabeled_cache=unlabeled_cache,
        test_cache=test_cache,
        y=y,
        cache_dir=cache_dir,
        device=device,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = train_spec_ensemble(
        train_base=train_base,
        test_base=test_base,
        train_template=train_template,
        test_template=test_template,
        train_ssl=train_ssl,
        test_ssl=test_ssl,
        y=y,
        cv=cv,
    )
    save_submission(results["ensemble_test"], submission_path)
    _dump_json(metrics_path, {"model_scores": results["model_scores"], "weights": results["weights"], "ensemble": results["ensemble_scores"]})
    return {
        "submission_path": str(submission_path),
        "metrics_path": str(metrics_path),
        "ensemble_scores": results["ensemble_scores"],
        "weights": results["weights"],
        "model_scores": results["model_scores"],
    }
