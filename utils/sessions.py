from __future__ import annotations

import numpy as np
import pandas as pd

# Hour ranges in UTC (FX gold liquidity proxy; adjust for your broker server time)
ASIA = (0, 8)
LONDON = (7, 16)
NY = (13, 22)
OVERLAP_LONDON_NY = (13, 16)


def hour_of_day_utc(ts: pd.Series) -> pd.Series:
    return ts.dt.hour.astype(np.int16)


def day_of_week(ts: pd.Series) -> pd.Series:
    return ts.dt.dayofweek.astype(np.int16)


def tokyo_london_ny_flags(hour: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """One-hot sesi FX (UTC): Tokyo 00–08, London 07–16, NY 13–22."""
    h = hour.astype(int)
    tokyo = ((h >= 0) & (h < 8)).astype(float)
    london = ((h >= 7) & (h < 16)).astype(float)
    ny = ((h >= 13) & (h < 22)).astype(float)
    return tokyo, london, ny


def session_bucket(hour: pd.Series) -> pd.Series:
    """Discrete session label: asia | london | ny | overlap | off."""

    def _one(h: int) -> str:
        if OVERLAP_LONDON_NY[0] <= h < OVERLAP_LONDON_NY[1]:
            return "overlap"
        if LONDON[0] <= h < LONDON[1] and not (OVERLAP_LONDON_NY[0] <= h < OVERLAP_LONDON_NY[1]):
            return "london"
        if NY[0] <= h < NY[1] and h >= OVERLAP_LONDON_NY[1]:
            return "ny"
        if ASIA[0] <= h < ASIA[1]:
            return "asia"
        return "off"

    return hour.map(_one)
