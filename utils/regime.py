"""Deteksi market regime untuk training/inference."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


REGIME_TREND_UP = "TREND_UP"
REGIME_TREND_DOWN = "TREND_DOWN"
REGIME_RANGE = "RANGE"
REGIME_HIGH_VOL = "HIGH_VOL"


def _adx_like(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    up = h.diff()
    down = -l.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=max(2, window // 2)).mean()
    plus_di = 100.0 * (pd.Series(plus_dm).rolling(window, min_periods=max(2, window // 2)).mean() / (atr + 1e-9))
    minus_di = 100.0 * (pd.Series(minus_dm).rolling(window, min_periods=max(2, window // 2)).mean() / (atr + 1e-9))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)) * 100.0
    return dx.rolling(window, min_periods=max(2, window // 2)).mean()


def detect_regime(df: pd.DataFrame, cfg: Dict[str, float] | None = None) -> pd.Series:
    """Klasifikasikan setiap bar ke TREND_UP/TREND_DOWN/RANGE/HIGH_VOL.

    Args:
        df: DataFrame wajib punya kolom `close`, `high`, `low`.
        cfg: Opsional override threshold.
    """
    cfg = cfg or {}
    adx_thr = float(cfg.get("adx_threshold", 22.0))
    vol_pct_thr = float(cfg.get("atr_percentile_high", 0.80))
    slope_thr = float(cfg.get("ema_slope_threshold", 0.0))
    ema_w = int(cfg.get("ema_window", 50))
    atr_w = int(cfg.get("atr_window", 14))

    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    ema = c.ewm(span=ema_w, adjust=False).mean()
    ema_slope = ema.diff()
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_w, min_periods=max(2, atr_w // 2)).mean()
    atr_pct = atr.rank(pct=True)
    adx = _adx_like(df, window=atr_w)

    out = pd.Series(REGIME_RANGE, index=df.index, dtype=object)
    out.loc[atr_pct >= vol_pct_thr] = REGIME_HIGH_VOL
    trend_mask = (adx >= adx_thr) & (atr_pct < vol_pct_thr)
    out.loc[trend_mask & (ema_slope > slope_thr)] = REGIME_TREND_UP
    out.loc[trend_mask & (ema_slope < -slope_thr)] = REGIME_TREND_DOWN
    return out

