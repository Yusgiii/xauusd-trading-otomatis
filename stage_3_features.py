# noqa: D100
"""Stage 3 — Fitur stasioner (rolling wajib shift(1) via utils/gbpjpy_features)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from utils.xauusd_features import build_xauusd_features
from utils.logging_config import stage_logger
from utils.paths import ensure_dir


def run_stage_3(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = ensure_dir(run_dir / "stage_3")
    log = stage_logger("stage_3_features", out)

    d, feat_cols = build_xauusd_features(df, cfg)

    meta = {"n_features": len(feat_cols), "feature_columns": feat_cols}
    pq = out / "stage_3_featured.parquet"
    d.to_parquet(pq, index=False)
    with (out / "stage_3_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Stage3 selesai | n_features=%s | %s", len(feat_cols), pq)
    return d, meta
