# noqa: D100
"""Test koneksi MT5 tanpa kirim order nyata."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    from utils.mt5_connection import get_mt5_terminal_path, initialize_mt5, shutdown_mt5

    path = get_mt5_terminal_path()
    print(f"Inisialisasi MT5 ({path or 'default'})...")
    try:
        ok, mt5 = initialize_mt5()
    except RuntimeError as exc:
        print(f"GAGAL: {exc}")
        sys.exit(1)
    if not ok:
        print(f"GAGAL init: {mt5.last_error()}")
        sys.exit(1)

    info = mt5.account_info()
    if info is None:
        print(f"GAGAL: account_info None — {mt5.last_error()}")
        shutdown_mt5(mt5)
        sys.exit(1)

    print(f"Account: {info.login} | Balance: {info.balance:.2f}")
    print(f"Server: {info.server}")
    print(f"Trade allowed: {info.trade_allowed}")

    from utils.mt5_export import resolve_symbol

    sym = resolve_symbol(mt5, "XAUUSD")
    si = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    print(f"Symbol: {sym} | filling_mode={getattr(si, 'filling_mode', '?')}")
    if tick:
        print(f"Bid: {tick.bid} | Ask: {tick.ask}")

    print("MT5 terkoneksi dengan benar (tidak ada order dikirim).")
    shutdown_mt5(mt5)


if __name__ == "__main__":
    main()
