"""Kalender pasar FX — hari libur akhir pekan (Sabtu & Minggu, UTC)."""

from __future__ import annotations

from typing import Tuple

import pandas as pd

# pandas dayofweek: Senin=0 … Jumat=4, Sabtu=5, Minggu=6
WEEKEND_DAYOFWEEK = (5, 6)


def weekend_mask(time: pd.Series) -> pd.Series:
    """True untuk bar dengan timestamp di Sabtu atau Minggu (UTC)."""
    t = pd.to_datetime(time, utc=True, errors="coerce")
    return t.dt.dayofweek.isin(WEEKEND_DAYOFWEEK)


def drop_weekend_bars(df: pd.DataFrame, *, time_col: str = "time") -> Tuple[pd.DataFrame, int]:
    """
    Buang bar Sabtu/Minggu. Return (dataframe, jumlah bar dihapus).
    """
    if df.empty or time_col not in df.columns:
        return df, 0
    mask = weekend_mask(df[time_col])
    n = int(mask.sum())
    if n == 0:
        return df, 0
    out = df.loc[~mask].sort_values(time_col).reset_index(drop=True)
    return out, n
