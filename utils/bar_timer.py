"""Utilitas timing bar untuk M15 dan timeframe lain."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def timeframe_to_bar_minutes(timeframe: str) -> int:
    """Derive bar length in minutes from timeframe string (e.g. M15 → 15)."""
    tf = str(timeframe).strip().upper()
    if tf.startswith("M") and tf[1:].isdigit():
        return max(1, int(tf[1:]))
    if tf.startswith("H") and tf[1:].isdigit():
        return max(1, int(tf[1:])) * 60
    if tf == "D1":
        return 1440
    return 15


def bar_open_time(dt: datetime, bar_minutes: int = 15) -> datetime:
    """
    Hitung waktu open bar saat ini berdasarkan timeframe.

    Contoh bar_minutes=15:
    - 11:47 → bar open 11:45
    - 11:59 → bar open 11:45
    - 12:00 → bar open 12:00
    """
    dt_utc = dt.astimezone(timezone.utc)
    total_minutes = dt_utc.hour * 60 + dt_utc.minute
    bar_start_minutes = (total_minutes // bar_minutes) * bar_minutes
    bar_open = dt_utc.replace(
        hour=bar_start_minutes // 60,
        minute=bar_start_minutes % 60,
        second=0,
        microsecond=0,
    )
    return bar_open


def next_bar_time(dt: datetime, bar_minutes: int = 15) -> datetime:
    """Hitung waktu open bar BERIKUTNYA."""
    current_bar = bar_open_time(dt, bar_minutes)
    return current_bar + timedelta(minutes=bar_minutes)


def seconds_until_next_bar(dt: datetime, bar_minutes: int = 15, delay_seconds: int = 120) -> float:
    """
    Hitung berapa detik lagi sampai bar baru terbentuk + delay.

    delay_seconds=120 = tunggu 2 menit setelah bar baru sebelum analisa.
    """
    next_bar = next_bar_time(dt, bar_minutes)
    target = next_bar + timedelta(seconds=delay_seconds)
    now_utc = dt.astimezone(timezone.utc)
    diff = (target - now_utc).total_seconds()
    return max(0.0, diff)


def is_new_bar(last_bar_open: datetime | None, dt: datetime, bar_minutes: int = 15) -> bool:
    """Return True jika bar saat ini berbeda dari bar terakhir yang dianalisa."""
    current_bar = bar_open_time(dt, bar_minutes)
    if last_bar_open is None:
        return True
    return current_bar > last_bar_open
