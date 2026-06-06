"""Helper horizon prediksi (bar timeframe -> durasi kalender)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict

_TF_BAR_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def horizon_bars(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("project", {}).get("horizon_bars", 1))


def horizon_timedelta(cfg: Dict[str, Any]) -> timedelta:
    """Durasi horizon dari jumlah bar × menit per bar timeframe."""
    bars = horizon_bars(cfg)
    tf = str(cfg.get("project", {}).get("timeframe", "H1")).upper()
    minutes = _TF_BAR_MINUTES.get(tf, 60)
    return timedelta(minutes=bars * minutes)


def horizon_label_id(cfg: Dict[str, Any]) -> str:
    """Teks singkat untuk laporan Telegram / skripsi."""
    bars = horizon_bars(cfg)
    td = horizon_timedelta(cfg)
    hours = td.total_seconds() / 3600
    tf = str(cfg.get("project", {}).get("timeframe", "H1")).upper()
    if tf == "H1" and bars == 1:
        return "1 jam ke depan (1 bar H1)"
    if tf == "D1" and bars == 1:
        return "1 hari ke depan (1 bar D1)"
    if hours >= 23.5:
        return f"{int(round(hours))} jam ke depan (~{bars} bar {tf})"
    if hours >= 1:
        return f"{hours:.0f} jam ke depan (~{bars} bar {tf})"
    return f"{bars} bar {tf} ke depan"
