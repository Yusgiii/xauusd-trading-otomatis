"""Wrapper fitur XAUUSD agar naming tidak misleading."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from utils.gbpjpy_features import (
    build_gbpjpy_features as build_base_features,
    prepare_inference_bars as prepare_base_inference_bars,
)


def prepare_inference_bars(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Alias inference bars untuk pipeline XAUUSD."""
    return prepare_base_inference_bars(df, cfg)


def build_xauusd_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    """Alias builder fitur utama XAUUSD (tetap gunakan base implementation)."""
    return build_base_features(df, cfg)

