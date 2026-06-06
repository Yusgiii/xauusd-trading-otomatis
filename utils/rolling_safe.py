from __future__ import annotations

import pandas as pd


def roll_shift1(series: pd.Series, window: int, fn: str = "mean") -> pd.Series:
    """
    Rolling statistic with mandatory causal shift(1): value at t uses [t-window, t-1].
    """
    rolled = getattr(series.rolling(window, min_periods=max(2, window // 3)), fn)()
    return rolled.shift(1)


def roll_std_shift1(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=max(2, window // 3)).std().shift(1)
