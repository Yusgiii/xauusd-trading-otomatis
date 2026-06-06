"""Koneksi MetaTrader 5 — path terminal terpusat."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Instalasi MT5 yang dipakai proyek ini
DEFAULT_MT5_TERMINAL = Path(r"C:\Program Files\mt1\terminal64.exe")

_terminal_override: Optional[str] = None


def set_mt5_terminal_path(path: str | Path | None) -> None:
    """Override path terminal (mis. dari config saat startup service)."""
    global _terminal_override
    _terminal_override = str(path).strip() if path else None


def get_mt5_terminal_path(cfg: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Path ke terminal64.exe.
    Prioritas: override runtime → config → env MT5_TERMINAL_PATH → default mt1.
    """
    if _terminal_override:
        p = Path(_terminal_override)
        return str(p) if p.is_file() else _terminal_override

    if cfg:
        s9 = cfg.get("stage_9", {}) if isinstance(cfg.get("stage_9"), dict) else {}
        for candidate in (
            s9.get("mt5_terminal_path"),
            (s9.get("mt5") or {}).get("terminal_path") if isinstance(s9.get("mt5"), dict) else None,
            cfg.get("project", {}).get("mt5_terminal_path") if isinstance(cfg.get("project"), dict) else None,
        ):
            if candidate:
                c = str(candidate).strip()
                if c:
                    return c

    env = os.environ.get("MT5_TERMINAL_PATH", "").strip()
    if env:
        return env

    if DEFAULT_MT5_TERMINAL.is_file():
        return str(DEFAULT_MT5_TERMINAL)

    return None


def initialize_mt5(cfg: Optional[Dict[str, Any]] = None) -> Tuple[bool, Any]:
    """
    Inisialisasi MT5 dengan path terminal yang benar.
    Return (ok, mt5_module).
    """
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise RuntimeError(f"Paket MetaTrader5 tidak terpasang: {exc}") from exc

    path = get_mt5_terminal_path(cfg)
    if path:
        ok = mt5.initialize(path=path)
    else:
        ok = mt5.initialize()
    return ok, mt5


def shutdown_mt5(mt5: Any) -> None:
    try:
        mt5.shutdown()
    except Exception:
        pass
