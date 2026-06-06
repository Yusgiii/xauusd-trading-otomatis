# noqa: D100
"""Stage 4 — Purged walk-forward + expanding window fold indices (strict time order)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.logging_config import stage_logger
from utils.paths import ensure_dir


def _regime_tag(vol: np.ndarray, i: int, window: int) -> str:
    if i < window:
        return "unknown"
    w = vol[i - window : i]
    q1, q2 = np.nanquantile(w, 0.33), np.nanquantile(w, 0.66)
    v = vol[i]
    if v < q1:
        return "low_vol"
    if v < q2:
        return "mid_vol"
    return "high_vol"


def run_stage_4(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = ensure_dir(run_dir / "stage_4")
    log = stage_logger("stage_4_validation", out)
    s4 = cfg["stage_4"]
    proj = cfg["project"]
    horizon = int(proj["horizon_bars"])
    n_splits = int(s4["n_splits"])
    embargo = int(s4["embargo_bars"])
    if embargo < horizon:
        embargo = horizon
    rw = int(s4["regime_vol_window"])

    n = len(df)
    end_ix = n - horizon - 2
    if end_ix < 80:
        horizon = max(1, n // 20)
        end_ix = n - horizon - 2
    cold = min(max(5000, int(0.1 * n)), max(40, end_ix // 2))
    if n < 4000:
        cold = min(max(30, int(0.06 * n)), max(20, end_ix // 3))
    testable = np.arange(cold, end_ix, dtype=int)
    if len(testable) < 80:
        cold = max(10, int(0.05 * n))
        testable = np.arange(cold, end_ix, dtype=int)

    chunks = np.array_split(testable, n_splits)

    vol = df["realized_vol_shift1"].to_numpy(dtype=float)

    train_min = int(s4.get("min_train_rows", 1000))
    test_min = int(s4.get("min_test_rows", 50))
    if n < 5000:
        train_min = max(60, int(0.12 * n))
        test_min = max(12, int(0.02 * n))

    folds: List[Dict[str, Any]] = []
    has_outcome = "tp_sl_outcome" in df.columns
    tp_sl_arr = df["tp_sl_outcome"].to_numpy(dtype=np.int8) if has_outcome else None
    for fold_id, test_idx in enumerate(chunks):
        if len(test_idx) < test_min:
            continue
        test_start = int(test_idx.min())
        train_end = max(cold, test_start - embargo)
        train_idx = np.arange(0, train_end, dtype=int)
        if len(train_idx) < train_min:
            continue
        reg = _regime_tag(vol, int(test_idx[len(test_idx) // 2]), rw)
        fold_item: Dict[str, Any] = {
            "fold": fold_id,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "regime_probe": reg,
            "embargo_bars": embargo,
        }
        if has_outcome and tp_sl_arr is not None:
            y_out = tp_sl_arr[test_idx]
            fold_item["tp_sl_outcome_test_counts"] = {
                "tp_first": int((y_out == 1).sum()),
                "sl_first": int((y_out == -1).sum()),
                "none_or_flat": int((y_out == 0).sum()),
            }
        folds.append(fold_item)

    meta = {
        "n_folds": len(folds),
        "embargo_bars": embargo,
        "horizon_bars": horizon,
        "has_tp_sl_outcome": bool(has_outcome),
    }
    with (out / "stage_4_fold_indices.json").open("w", encoding="utf-8") as f:
        json.dump(folds, f)

    with (out / "stage_4_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Stage4 selesai | folds=%s | meta=%s", len(folds), meta)
    return df, meta
