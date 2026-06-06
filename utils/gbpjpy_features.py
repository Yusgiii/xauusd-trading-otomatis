"""Fitur stasioner FX/metal — dipakai Stage 3 dan inferensi live Stage 9."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.rolling_safe import roll_shift1, roll_std_shift1
from utils.sessions import day_of_week, hour_of_day_utc, tokyo_london_ny_flags


def prepare_inference_bars(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Kolom kalender + vol (sama seperti Stage 1) agar selaras dengan model terlatih."""
    d = df.copy()
    if "time" in d.columns:
        d["time"] = pd.to_datetime(d["time"], utc=True, errors="coerce")
        if "hour_of_day" not in d.columns:
            d["hour_of_day"] = hour_of_day_utc(d["time"])
        if "day_of_week" not in d.columns:
            d["day_of_week"] = day_of_week(d["time"])
    if "realized_vol_shift1" not in d.columns and "close" in d.columns:
        vol_w = int(cfg.get("stage_3", {}).get("vol_window", 60))
        rets = np.log(d["close"]).diff()
        d["realized_vol_shift1"] = roll_std_shift1(rets, vol_w)
    return d


def _rsi_zscore(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window // 2).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window // 2).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    mu = rsi.rolling(window, min_periods=window // 2).mean()
    sd = rsi.rolling(window, min_periods=window // 2).std()
    z = (rsi - mu) / (sd + 1e-9)
    return z.shift(1)


def _atr_zscore(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_c).abs(), (low - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window, min_periods=window // 2).mean()
    mu = atr.rolling(window, min_periods=window // 2).mean()
    sd = atr.rolling(window, min_periods=window // 2).std()
    z = (atr - mu) / (sd + 1e-9)
    return z.shift(1)


def _hurst_proxy(rs_window: int, rets: pd.Series) -> pd.Series:
    r = rets.abs()
    rng = rets.rolling(rs_window).max() - rets.rolling(rs_window).min()
    h = (r / (rng + 1e-9)).rolling(rs_window).mean()
    return h.shift(1)


def _asian_range_breakout(d: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    t = pd.to_datetime(d["time"], utc=True, errors="coerce")
    day = t.dt.floor("D")
    is_asian = d["hour_of_day"].between(2, 9)
    asian_high = d["high"].where(is_asian).groupby(day).transform("max")
    asian_low = d["low"].where(is_asian).groupby(day).transform("min")
    up = (d["close"] > asian_high).astype(float).shift(1)
    down = (d["close"] < asian_low).astype(float).shift(1)
    return up, down


def _daily_range_position(d: pd.DataFrame) -> pd.Series:
    t = pd.to_datetime(d["time"], utc=True, errors="coerce")
    day = t.dt.floor("D")
    daily_high = d["high"].groupby(day).transform("max")
    daily_low = d["low"].groupby(day).transform("min")
    pos = (d["close"] - daily_low) / (daily_high - daily_low + 1e-9)
    return pos.clip(0.0, 1.0).shift(1)


def build_gbpjpy_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """Bangun matriks fitur; semua rolling memakai shift(1) (tanpa look-ahead)."""
    s3 = cfg.get("stage_3", {})
    ema_fast = int(s3.get("ema_fast", 20))
    ema_slow = int(s3.get("ema_slow", 50))
    zw = int(s3.get("zscore_window", 60))
    atr_w = int(s3.get("atr_window", 14))
    rsi_w = int(s3.get("rsi_window", 14))
    hw = int(s3.get("hurst_window", 32))

    d = df.copy()
    if "hour_of_day" not in d.columns and "time" in d.columns:
        d["hour_of_day"] = hour_of_day_utc(d["time"])
    if "day_of_week" not in d.columns and "time" in d.columns:
        d["day_of_week"] = day_of_week(d["time"])

    c = d["close"]
    h, l = d["high"], d["low"]
    logret = np.log(c).diff()

    d["log_return"] = logret.shift(1)
    ema20 = roll_shift1(c, ema_fast, "mean")
    ema50 = roll_shift1(c, ema_slow, "mean")
    d["distance_to_ema20"] = (c - ema20) / (roll_std_shift1(c, ema_fast) + 1e-9)
    d["distance_to_ema50"] = (c - ema50) / (roll_std_shift1(c, ema_slow) + 1e-9)
    d["rsi_zscore"] = _rsi_zscore(c, rsi_w)
    d["atr_zscore"] = _atr_zscore(h, l, c, atr_w)
    d["hurst_proxy"] = _hurst_proxy(hw, logret)
    atr_abs = pd.concat(
        [(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1).rolling(atr_w, min_periods=max(2, atr_w // 2)).mean()
    atr_abs_shift1 = atr_abs.shift(1)
    sp_roll_mu = d["spread"].rolling(zw, min_periods=max(5, zw // 3)).mean().shift(1)
    sp_roll_sd = d["spread"].rolling(zw, min_periods=max(5, zw // 3)).std().shift(1)
    d["spread_shock"] = ((d["spread"] - sp_roll_mu) / (sp_roll_sd + 1e-9)).replace(
        [np.inf, -np.inf], np.nan
    )
    d["bar_momentum"] = ((c - d["open"]) / (atr_abs_shift1 + 1e-9)).replace(
        [np.inf, -np.inf], np.nan
    )
    arb_up, arb_down = _asian_range_breakout(d)
    d["asian_range_breakout_up"] = arb_up
    d["asian_range_breakout_down"] = arb_down
    d["daily_range_pos"] = _daily_range_position(d)

    tokyo, london, ny = tokyo_london_ny_flags(d["hour_of_day"])
    d["session_tokyo"] = tokyo
    d["session_london"] = london
    d["session_ny"] = ny
    d["london_open_proxy"] = (d["hour_of_day"] == 7).astype(float)
    # Posisi progres sesi [0,1] sebagai fitur timing intraday.
    d["session_progress"] = ((d["hour_of_day"] % 8) / 8.0).astype(float)

    meta_exclude = {
        "time",
        "target",
        "tp_sl_outcome",
        "mfe_long",
        "mae_long",
        "mfe_short",
        "mae_short",
        "label_quality_score",
        "ambiguity_score",
        "keep_for_training",
        "gap_before",
        "spread_spike",
        "session",
        "vol_regime",
        "realized_vol_shift1",
        "flat_return_threshold",
        "horizon_bars",
        "label_atr_window",
        "label_sl_atr_multiplier",
        "label_tp_rr",
    }
    price_cols = {"open", "high", "low", "close", "spread"}

    feat_cols = [
        col
        for col in d.columns
        if col not in meta_exclude and col not in price_cols
    ]

    d[feat_cols] = d[feat_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return d, feat_cols
