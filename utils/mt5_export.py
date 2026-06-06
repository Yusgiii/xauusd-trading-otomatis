"""Ekspor riwayat OHLCV+spread dari MetaTrader 5 ke CSV pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

from utils.mt5_connection import initialize_mt5, shutdown_mt5
from utils.trading_calendar import drop_weekend_bars


_TF_MAP = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
}


def mt5_timeframe_const(mt5, timeframe: str) -> int:
    """Konstanta MT5 untuk string timeframe (H1, D1, …)."""
    tf = timeframe.upper()
    if tf not in _TF_MAP:
        raise ValueError(f"timeframe tidak didukung: {tf} (didukung: {sorted(_TF_MAP)})")
    attr = f"TIMEFRAME_{tf}"
    if not hasattr(mt5, attr):
        raise ValueError(f"MetaTrader5 tidak punya {attr}")
    return int(getattr(mt5, attr))


def _maybe_mt5():
    try:
        import MetaTrader5 as mt5

        return mt5
    except ImportError:
        return None


def resolve_symbol(mt5, symbol: str) -> str:
    if mt5.symbol_select(symbol, True):
        return symbol
    candidates: List[str] = []
    for s in mt5.symbols_get() or []:
        name = s.name
        if name.upper().startswith(symbol.upper()):
            candidates.append(name)
    if not candidates:
        raise RuntimeError(
            f"Simbol '{symbol}' tidak ditemukan di MT5. Buka Market Watch dan aktifkan simbol tersebut."
        )
    for c in sorted(candidates, key=lambda x: (x.upper() != symbol.upper(), len(x))):
        if mt5.symbol_select(c, True):
            return c
    raise RuntimeError(f"Tidak bisa memilih simbol untuk {symbol}. Kandidat: {candidates[:8]}")


def _fetch_rates_max_history(
    mt5,
    resolved: str,
    tf_const: int,
    *,
    n_bars: int,
):
    """Ambil riwayat sepanjang mungkin: range dari tahun lama + copy_rates_from_pos."""
    end = datetime.now(timezone.utc)
    best = None
    best_n = 0

    for start_year in (1990, 1995, 2000, 2005, 2010, 2015, 2018, 2020):
        start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
        chunk = mt5.copy_rates_range(resolved, tf_const, start, end)
        if chunk is not None and len(chunk) > best_n:
            best = chunk
            best_n = len(chunk)

    for count in (int(n_bars), 200_000, 100_000, 50_000, 20_000, 10_000, 5_000):
        if count <= best_n:
            continue
        chunk = mt5.copy_rates_from_pos(resolved, tf_const, 0, count)
        if chunk is not None and len(chunk) > best_n:
            best = chunk
            best_n = len(chunk)

    return best


def export_ohlcv_csv(
    symbol: str,
    out_path: Path,
    *,
    timeframe: str = "H1",
    n_bars: int = 100_000,
) -> int:
    """
    Tarik riwayat OHLCV sepanjang mungkin dari MT5 → CSV (time, open, high, low, close, spread).
    Mengembalikan jumlah bar yang ditulis.
    """
    tf = timeframe.upper()
    if tf not in _TF_MAP:
        raise ValueError(f"timeframe tidak didukung: {tf}")

    ok, mt5 = initialize_mt5()
    if not ok:
        raise RuntimeError(f"MT5 initialize gagal: {mt5.last_error()}")

    try:
        resolved = resolve_symbol(mt5, symbol)
        tf_const = mt5_timeframe_const(mt5, tf)

        rates = _fetch_rates_max_history(mt5, resolved, tf_const, n_bars=n_bars)

        if rates is None or len(rates) == 0:
            raise RuntimeError(
                f"Tidak ada data untuk {resolved} {tf}: {mt5.last_error()}. "
                f"Pastikan MT5 login, simbol aktif di Market Watch, dan buka chart {tf} {symbol} sekali."
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if "spread" not in df.columns:
            df["spread"] = 0

        out = (
            df[["time", "open", "high", "low", "close", "spread"]]
            .drop_duplicates(subset=["time"], keep="last")
            .sort_values("time")
        )
        out, _ = drop_weekend_bars(out)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        return len(out)
    finally:
        shutdown_mt5(mt5)


def ensure_input_csv(
    csv_path: Path,
    symbol: str,
    *,
    timeframe: str = "H1",
    n_bars: int = 100_000,
) -> Optional[int]:
    """Unduh dari MT5 jika `csv_path` belum ada. Return jumlah bar jika unduh, else None."""
    if csv_path.is_file():
        return None
    n = export_ohlcv_csv(symbol, csv_path, timeframe=timeframe, n_bars=n_bars)
    return n
