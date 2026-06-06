# noqa: D100
"""Stage 1 — OHLCV bar prep (M5 / H1 / …); anti-leakage: no future-derived fields."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from utils.logging_config import stage_logger
from utils.mt5_export import ensure_input_csv
from utils.paths import ensure_dir, project_root
from utils.sessions import day_of_week, hour_of_day_utc, session_bucket
from utils.rolling_safe import roll_std_shift1
from utils.trading_calendar import drop_weekend_bars


REQUIRED = ("time", "open", "high", "low", "close", "spread")

_TF_BAR_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def _bar_minutes(cfg: Dict[str, Any]) -> int:
    s1 = cfg.get("stage_1", {})
    if s1.get("bar_minutes") is not None:
        return int(s1["bar_minutes"])
    tf = str(cfg.get("project", {}).get("timeframe", "M5")).upper()
    if tf not in _TF_BAR_MINUTES:
        raise ValueError(
            f"timeframe tidak dikenal: {tf}. Tambahkan stage_1.bar_minutes atau gunakan salah satu {sorted(_TF_BAR_MINUTES)}"
        )
    return _TF_BAR_MINUTES[tf]


def run_stage_1(
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    s1 = cfg["stage_1"]
    proj = cfg["project"]
    out = ensure_dir(run_dir / "stage_1")
    log = stage_logger("stage_1_data", out)

    path = Path(s1["input_csv"])
    if not path.is_absolute():
        path = project_root() / path

    if not path.is_file():
        if bool(s1.get("auto_fetch_mt5", True)):
            log.info(
                "Stage1 | CSV tidak ada — mengunduh %s %s dari MT5 ke %s",
                proj.get("timeframe", "D1"),
                proj["symbol"],
                path,
            )
            n_bars = int(s1.get("mt5_fetch_bars", 100_000))
            try:
                ensure_input_csv(
                    path,
                    str(proj["symbol"]),
                    timeframe=str(proj.get("timeframe", "H1")),
                    n_bars=n_bars,
                )
                log.info("Stage1 | unduhan MT5 selesai | %s bar", n_bars)
            except Exception as exc:
                raise FileNotFoundError(
                    f"Data tidak ditemukan: {path} dan unduhan MT5 gagal: {exc}\n"
                    "Pastikan MetaTrader 5 terbuka & login, lalu jalankan:\n"
                    f"  python scripts/fetch_ohlcv_from_mt5.py --config configs/pipeline.yaml"
                ) from exc
        else:
            raise FileNotFoundError(
                f"Data tidak ditemukan: {path}. Set stage_1.input_csv atau "
                "aktifkan stage_1.auto_fetch_mt5, atau jalankan scripts/fetch_ohlcv_from_mt5.py"
            )

    df = pd.read_csv(path)
    miss = [c for c in REQUIRED if c not in df.columns]
    if miss:
        raise ValueError(f"Kolom wajib hilang: {miss}")

    df = df.copy()
    if bool(s1.get("drop_extra_csv_columns", False)):
        keep = list(REQUIRED)
        df = df[keep].copy()
    tz = s1.get("timezone", "UTC")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    dup = int(df["time"].duplicated().sum())
    df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

    for col in ("open", "high", "low", "close", "spread"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

    weekend_removed = 0
    if bool(s1.get("exclude_weekend_bars", True)):
        df, weekend_removed = drop_weekend_bars(df)
        if weekend_removed:
            log.info("Stage1 | bar Sabtu/Minggu dihapus: %s", weekend_removed)

    bar_m = _bar_minutes(cfg)
    dt = df["time"].diff()
    max_gap = int(s1.get("max_gap_minutes", 30))
    gap_mask = dt > pd.Timedelta(minutes=max_gap)
    df["gap_before"] = gap_mask.astype(np.int8)

    # Spread sanity
    sp = df["spread"].astype(float)
    cap_q = float(cfg.get("risk", {}).get("spread_cap_quantile", 0.99))
    sp_cap = float(sp.quantile(cap_q))
    df["spread_spike"] = (sp > sp_cap).astype(np.int8)

    # Session + calendar (causal calendar flags — no leakage)
    df["hour_of_day"] = hour_of_day_utc(df["time"])
    df["day_of_week"] = day_of_week(df["time"])
    df["session"] = session_bucket(df["hour_of_day"])

    # Volatility regime (past-only after shift inside helper)
    rets = np.log(df["close"]).diff()
    vol_w = int(cfg.get("stage_3", {}).get("vol_window", 60))
    rv = roll_std_shift1(rets, vol_w)
    df["realized_vol_shift1"] = rv
    q1, q2 = rv.quantile(0.33), rv.quantile(0.66)

    def _regime(x: float) -> str:
        if np.isnan(x):
            return "unknown"
        if x < q1:
            return "low"
        if x < q2:
            return "mid"
        return "high"

    df["vol_regime"] = rv.map(_regime)

    # CSV sering punya kolom tanggal/string tambahan (mis. duplikat date) → buang agar Stage 5 tidak error to_numpy.
    keep_object = {"session", "vol_regime"}
    for c in list(df.columns):
        if c in keep_object or c == "time":
            continue
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            log.warning("Stage1 | menghapus kolom non-numerik / string: %s", c)
            df = df.drop(columns=[c])

    meta: Dict[str, Any] = {
        "rows": int(len(df)),
        "duplicates_removed": dup,
        "time_start": df["time"].iloc[0].isoformat() if len(df) else None,
        "time_end": df["time"].iloc[-1].isoformat() if len(df) else None,
        "spread_median": float(df["spread"].median()),
        "spread_cap_used": sp_cap,
        "timeframe": proj["timeframe"],
        "bar_minutes": bar_m,
        "gap_rows": int(gap_mask.sum()),
        "weekend_bars_removed": weekend_removed,
        "exclude_weekend_bars": bool(s1.get("exclude_weekend_bars", True)),
    }

    pq = out / "stage_1_clean.parquet"
    df.to_parquet(pq, index=False)
    with (out / "stage_1_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log.info("Stage1 selesai | rows=%s | parquet=%s", len(df), pq)
    return df, meta
