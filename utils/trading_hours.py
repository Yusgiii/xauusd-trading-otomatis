"""Filter jam trading XAUUSD."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from typing import Optional, Tuple

MARKET_CLOSE_UTC = time(22, 0)  # Jumat 22:00 UTC = Sabtu 05:00 WIB
MARKET_OPEN_UTC = time(23, 0)  # Minggu 23:00 UTC = Senin 06:00 WIB


def is_market_open(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    if dt is None:
        dt = datetime.now(timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    weekday = dt_utc.weekday()
    t = dt_utc.time()
    if weekday == 5:
        return False, "Sabtu — pasar tutup"
    if weekday == 4 and t >= MARKET_CLOSE_UTC:
        return False, f"Jumat {t.strftime('%H:%M')} UTC — pasar tutup"
    if weekday == 6 and t < MARKET_OPEN_UTC:
        return False, f"Minggu {t.strftime('%H:%M')} UTC — belum buka"
    return True, "Pasar buka"


def seconds_until_market_open(dt: Optional[datetime] = None) -> float:
    if dt is None:
        dt = datetime.now(timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    if is_market_open(dt_utc)[0]:
        return 0.0
    weekday = dt_utc.weekday()
    days_ahead = {4: 2, 5: 1, 6: 0}.get(weekday, 0)
    target = dt_utc.replace(hour=23, minute=0, second=0, microsecond=0) + timedelta(days=days_ahead)
    if target <= dt_utc:
        target += timedelta(days=7)
    return max(0.0, (target - dt_utc).total_seconds())


def get_session(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    if not is_market_open(dt_utc)[0]:
        return "CLOSED"
    h = dt_utc.hour
    s = []
    if 0 <= h < 9:
        s.append("Tokyo")
    if 7 <= h < 16:
        s.append("London")
    if 12 <= h < 21:
        s.append("NewYork")
    return "+".join(s) if s else "Off-hours"
