# noqa: D100
"""
Stage 2 — Directional return classification (3 kelas).

Kelas 0 (FLAT): |ln(close_{t+H}/close_t)| <= flat_return_threshold
Kelas 1 (UP):   ln(close_{t+H}/close_t) > threshold
Kelas 2 (DOWN): ln(close_{t+H}/close_t) < -threshold

`horizon_bars` = jumlah bar ke depan pada timeframe proyek (D1: H=1 = 1 hari trading).
Opsional `daily_anchor`: hanya bar pembuka jam tertentu (WIB) yang dipakai training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from utils.logging_config import stage_logger
from utils.paths import ensure_dir
from zoneinfo import ZoneInfo

WIB = ZoneInfo("Asia/Jakarta")

LABEL_DROP = -1
LABEL_FLAT, LABEL_UP, LABEL_DOWN = 0, 1, 2


def _rolling_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    prev_c = np.roll(close, 1)
    prev_c[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    s = pd.Series(tr)
    atr = s.rolling(window, min_periods=max(2, window // 2)).mean().to_numpy(dtype=float)
    return np.where(np.isfinite(atr), atr, np.nan)


def _first_hit_index(highs: np.ndarray, lows: np.ndarray, tp_price: float, sl_price: float) -> Tuple[int, int]:
    tp_idx = -1
    sl_idx = -1
    tp_hits = np.flatnonzero(highs >= tp_price)
    sl_hits = np.flatnonzero(lows <= sl_price)
    if len(tp_hits):
        tp_idx = int(tp_hits[0])
    if len(sl_hits):
        sl_idx = int(sl_hits[0])
    return tp_idx, sl_idx


def run_stage_2(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = ensure_dir(run_dir / "stage_2")
    log = stage_logger("stage_2_labeling", out)
    s2 = cfg.get("stage_2", {})
    proj = cfg["project"]
    risk = cfg["risk"]

    horizon = int(proj["horizon_bars"])
    thresh = float(risk["flat_return_threshold"])

    c = df["close"].to_numpy(dtype=float)
    sp = df["spread"].to_numpy(dtype=float)
    n = len(df)

    labels = np.full(n, LABEL_DROP, dtype=np.int8)
    log_rets_fwd = np.full(n, np.nan, dtype=float)
    keep = np.ones(n, dtype=bool)

    spread_med = float(np.nanmedian(sp))
    spread_std = float(np.nanstd(sp) + 1e-9)
    max_z = float(s2.get("max_spread_zscore", 4.0))
    wick_ratio_max = float(s2.get("wick_anomaly_ratio", 6.0))

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    atr_w = int(s2.get("atr_window", cfg.get("stage_3", {}).get("atr_window", 14)))
    atr = _rolling_atr(h, l, c, atr_w)
    label_cfg = s2.get("risk_labeling", {}) if isinstance(s2.get("risk_labeling", {}), dict) else {}
    sl_mult = float(label_cfg.get("sl_atr_multiplier", 1.6))
    rr = max(1.5, float(label_cfg.get("tp_rr", 1.8)))

    tp_sl_outcome = np.zeros(n, dtype=np.int8)  # 1=TP first, -1=SL first, 0=none/flat
    mfe_long = np.full(n, np.nan, dtype=float)
    mae_long = np.full(n, np.nan, dtype=float)
    mfe_short = np.full(n, np.nan, dtype=float)
    mae_short = np.full(n, np.nan, dtype=float)

    last = n - horizon - 1
    for i in range(max(0, last + 1)):
        j = i + horizon
        if j >= n:
            keep[i] = False
            continue

        body = abs(c[i] - o[i])
        hl = h[i] - l[i] + 1e-12
        wick_ratio = (hl - body) / (body + 1e-9)
        zsp = (sp[i] - spread_med) / spread_std
        if abs(zsp) > max_z or wick_ratio > wick_ratio_max:
            keep[i] = False
            labels[i] = LABEL_DROP
            continue

        lr = float(np.log(c[j] / c[i]))
        log_rets_fwd[i] = lr
        if abs(lr) <= thresh:
            labels[i] = LABEL_FLAT
        elif lr > thresh:
            labels[i] = LABEL_UP
        else:
            labels[i] = LABEL_DOWN

        if np.isfinite(atr[i]):
            atr_i = float(atr[i])
        else:
            hist = atr[: i + 1]
            hist = hist[np.isfinite(hist)]
            atr_i = float(np.median(hist)) if len(hist) else float("nan")
        if not np.isfinite(atr_i) or atr_i <= 0:
            atr_i = max(float(h[i] - l[i]), 1e-9)
        sl_dist = max(sl_mult * atr_i, 1e-9)
        tp_dist = rr * sl_dist

        f_high = h[i + 1 : j + 1]
        f_low = l[i + 1 : j + 1]
        if len(f_high):
            mfe_long[i] = float(np.max(f_high - c[i]))
            mae_long[i] = float(np.max(c[i] - f_low))
            mfe_short[i] = float(np.max(c[i] - f_low))
            mae_short[i] = float(np.max(f_high - c[i]))

        # Outcome hanya dievaluasi untuk arah label dominan agar sinkron dengan target utama.
        if labels[i] == LABEL_UP:
            tp_idx, sl_idx = _first_hit_index(
                f_high, f_low, tp_price=c[i] + tp_dist, sl_price=c[i] - sl_dist
            )
            if tp_idx >= 0 and (sl_idx < 0 or tp_idx <= sl_idx):
                tp_sl_outcome[i] = 1
            elif sl_idx >= 0 and (tp_idx < 0 or sl_idx < tp_idx):
                tp_sl_outcome[i] = -1
        elif labels[i] == LABEL_DOWN:
            # Untuk short: TP tercapai saat low <= entry - tp_dist, SL saat high >= entry + sl_dist.
            tp_hits = np.flatnonzero(f_low <= (c[i] - tp_dist))
            sl_hits = np.flatnonzero(f_high >= (c[i] + sl_dist))
            tp_idx = int(tp_hits[0]) if len(tp_hits) else -1
            sl_idx = int(sl_hits[0]) if len(sl_hits) else -1
            if tp_idx >= 0 and (sl_idx < 0 or tp_idx <= sl_idx):
                tp_sl_outcome[i] = 1
            elif sl_idx >= 0 and (tp_idx < 0 or sl_idx < tp_idx):
                tp_sl_outcome[i] = -1

    if last + 1 < n:
        keep[last + 1 :] = False
        labels[last + 1 :] = LABEL_DROP

    df = df.copy()
    df["target"] = labels
    df["forward_log_return"] = log_rets_fwd
    df["keep_for_training"] = (
        keep & np.isin(labels, [LABEL_FLAT, LABEL_UP, LABEL_DOWN])
    ).astype(np.int8)
    df["flat_return_threshold"] = thresh
    df["horizon_bars"] = horizon
    df["tp_sl_outcome"] = tp_sl_outcome
    df["mfe_long"] = mfe_long
    df["mae_long"] = mae_long
    df["mfe_short"] = mfe_short
    df["mae_short"] = mae_short
    df["label_atr_window"] = atr_w
    df["label_sl_atr_multiplier"] = sl_mult
    df["label_tp_rr"] = rr

    anchor_meta: Dict[str, Any] = {"enabled": False}
    anchor_cfg = proj.get("daily_anchor") or {}
    if bool(anchor_cfg.get("enabled", False)):
        wib_h = int(anchor_cfg.get("wib_open_hour", 5))
        times_wib = pd.to_datetime(df["time"], utc=True).dt.tz_convert(WIB)
        is_anchor = times_wib.dt.hour == wib_h
        df.loc[~is_anchor, "keep_for_training"] = 0
        anchor_meta = {
            "enabled": True,
            "wib_open_hour": wib_h,
            "anchor_rows": int(is_anchor.sum()),
            "kept_after_anchor": int((df["keep_for_training"] == 1).sum()),
        }

    stride_meta: Dict[str, Any] = {"enabled": False}
    stride = int(proj.get("training_label_stride", 1))
    if stride > 1:
        kept_idx = np.flatnonzero(df["keep_for_training"].to_numpy() == 1)
        if len(kept_idx):
            keep_every = np.zeros(len(kept_idx), dtype=bool)
            keep_every[np.arange(0, len(kept_idx), stride)] = True
            df.loc[kept_idx[~keep_every], "keep_for_training"] = 0
        stride_meta = {
            "enabled": True,
            "stride": stride,
            "kept_after_stride": int((df["keep_for_training"] == 1).sum()),
        }

    k = df["keep_for_training"].to_numpy() == 1
    kept_labels = labels[k]
    meta = {
        "labeling": "directional_return_3class",
        "horizon_bars": horizon,
        "flat_return_threshold": thresh,
        "class_counts_all_rows": {
            "DROP_FILTER": int((labels == LABEL_DROP).sum()),
            "FLAT": int((labels == LABEL_FLAT).sum()),
            "UP": int((labels == LABEL_UP).sum()),
            "DOWN": int((labels == LABEL_DOWN).sum()),
        },
        "class_counts_kept_only": {
            "FLAT": int((kept_labels == LABEL_FLAT).sum()),
            "UP": int((kept_labels == LABEL_UP).sum()),
            "DOWN": int((kept_labels == LABEL_DOWN).sum()),
        },
        "risk_outcome_counts_kept_only": {
            "TP_FIRST": int(((df["keep_for_training"] == 1) & (df["tp_sl_outcome"] == 1)).sum()),
            "SL_FIRST": int(((df["keep_for_training"] == 1) & (df["tp_sl_outcome"] == -1)).sum()),
            "NO_HIT_OR_FLAT": int(((df["keep_for_training"] == 1) & (df["tp_sl_outcome"] == 0)).sum()),
        },
        "risk_labeling": {
            "atr_window": atr_w,
            "sl_atr_multiplier": sl_mult,
            "tp_rr": rr,
        },
        "kept_fraction": float(k.mean()),
        "daily_anchor": anchor_meta,
        "training_label_stride": stride_meta,
    }
    flat_kept = int((kept_labels == LABEL_FLAT).sum())
    flat_ratio = (flat_kept / max(int(len(kept_labels)), 1))
    log.info(
        "Stage2 distribusi target (kept) | FLAT=%d UP=%d DOWN=%d | flat_ratio=%.4f",
        flat_kept,
        int((kept_labels == LABEL_UP).sum()),
        int((kept_labels == LABEL_DOWN).sum()),
        float(flat_ratio),
    )

    kept_mask = df["keep_for_training"].to_numpy() == 1
    df["target_up_binary"] = (df["target"] == LABEL_UP).astype(np.int8)
    df["target_down_binary"] = (df["target"] == LABEL_DOWN).astype(np.int8)
    df.loc[df["keep_for_training"] != 1, "target_up_binary"] = 0
    df.loc[df["keep_for_training"] != 1, "target_down_binary"] = 0

    meta["binary_label_counts"] = {
        "up_positive": int((df.loc[kept_mask, "target_up_binary"] == 1).sum()),
        "up_negative": int((df.loc[kept_mask, "target_up_binary"] == 0).sum()),
        "down_positive": int((df.loc[kept_mask, "target_down_binary"] == 1).sum()),
        "down_negative": int((df.loc[kept_mask, "target_down_binary"] == 0).sum()),
    }

    pq = out / "stage_2_labeled.parquet"
    df.to_parquet(pq, index=False)
    with (out / "stage_2_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Stage2 selesai | %s | parquet=%s", meta["class_counts_kept_only"], pq)
    return df, meta
