"""Track live trade outcomes untuk evaluasi performance ongoing."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.mt5_connection import initialize_mt5, shutdown_mt5
from utils.paths import project_root

LIVE_LOG_COLUMNS = [
    "timestamp_signal",
    "timestamp_entry",
    "symbol",
    "side",
    "lot_size",
    "confidence_tier",
    "confidence_score",
    "meta_score",
    "p_up_binary",
    "p_down_binary",
    "entry_price",
    "sl_price",
    "tp_price",
    "rr_planned",
    "drift_level",
    "confirmed_by_user",
    "outcome",
    "exit_price",
    "profit_r",
    "run_id",
]

_CLOSED_OUTCOMES = {"TP", "SL", "MANUAL_CLOSE"}
_OUTCOME_ALIASES = {
    "TP_TOUCH": "TP",
    "SL_TOUCH": "SL",
    "TP_HIT": "TP",
    "SL_HIT": "SL",
}


def live_log_path(cfg: Optional[Dict[str, Any]] = None) -> Path:
    rel = "logs/live_trade_log.csv"
    if cfg:
        rel = str(cfg.get("stage_9", {}).get("live_trade_log_path", rel))
    p = Path(rel)
    return p if p.is_absolute() else project_root() / p


def log_signal(log_path: Path, trade_data: Dict[str, Any]) -> str:
    """Append satu baris ke live trade log. Return timestamp_signal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = str(trade_data.get("timestamp_signal") or datetime.now(timezone.utc).isoformat())
    row = {col: trade_data.get(col, "") for col in LIVE_LOG_COLUMNS}
    row["timestamp_signal"] = ts
    write_header = not log_path.is_file() or log_path.stat().st_size == 0
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LIVE_LOG_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return ts


def update_outcome(
    log_path: Path,
    timestamp_signal: str,
    outcome: str,
    exit_price: float,
    profit_r: float,
) -> None:
    """Update outcome untuk trade yang sudah close."""
    if not log_path.is_file():
        return
    df = pd.read_csv(log_path, engine="python", on_bad_lines="skip")
    for col in LIVE_LOG_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    mask = df["timestamp_signal"].astype(str) == str(timestamp_signal)
    if mask.sum() == 0:
        return
    df.loc[mask, "outcome"] = outcome
    df.loc[mask, "exit_price"] = exit_price
    df.loc[mask, "profit_r"] = profit_r
    df.to_csv(log_path, index=False)


def _normalize_outcomes(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.upper().str.strip()
    return s.replace(_OUTCOME_ALIASES)


def _deal_matches_symbol(deal: Any, resolved_sym: str, base_symbol: str) -> bool:
    sym = str(getattr(deal, "symbol", "")).upper()
    base = base_symbol.upper()
    resolved = resolved_sym.upper()
    return sym == resolved or sym == base or sym.startswith(base)


def _deal_to_profit_r(deal: Any, log_df: Optional[pd.DataFrame]) -> float:
    profit = float(getattr(deal, "profit", 0.0))
    if profit <= 0:
        return -1.0
    if log_df is not None and not log_df.empty and "timestamp_entry" in log_df.columns:
        try:
            deal_ts = pd.Timestamp.fromtimestamp(int(getattr(deal, "time", 0)), tz="UTC")
            entries = pd.to_datetime(log_df["timestamp_entry"], utc=True, errors="coerce")
            diffs = (entries - deal_ts).abs()
            if diffs.notna().any():
                idx = diffs.idxmin()
                if pd.notna(diffs.loc[idx]) and diffs.loc[idx] <= pd.Timedelta(hours=12):
                    rr = pd.to_numeric(log_df.loc[idx, "rr_planned"], errors="coerce")
                    if pd.notna(rr) and float(rr) > 0:
                        return float(rr)
        except Exception:
            pass
    return 1.0


def fetch_mt5_trade_stats(
    symbol: str = "XAUUSD",
    *,
    days: int = 7,
    log_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Ringkasan trade + akun dari MT5 (backup jika CSV belum ter-update)."""
    out: Dict[str, Any] = {"ok": False, "symbol": symbol, "days": days}
    try:
        ok, mt5 = initialize_mt5()
    except RuntimeError as exc:
        out["error"] = str(exc)
        return out
    if not ok:
        out["error"] = str(mt5.last_error())
        return out

    try:
        from utils.mt5_export import resolve_symbol

        resolved = resolve_symbol(mt5, symbol)
        account = mt5.account_info()
        balance = float(account.balance) if account is not None else 0.0
        equity = float(account.equity) if account is not None else 0.0

        positions = mt5.positions_get(symbol=resolved)
        if not positions:
            positions = mt5.positions_get(symbol=symbol)
        open_n = len(positions) if positions else 0

        from_date = datetime.now(timezone.utc) - timedelta(days=days)
        to_date = datetime.now(timezone.utc)
        deals = mt5.history_deals_get(from_date, to_date)
        closed_deals: List[Any] = []
        if deals:
            for d in deals:
                if int(getattr(d, "entry", -1)) != 1:
                    continue
                if not _deal_matches_symbol(d, resolved, symbol):
                    continue
                closed_deals.append(d)

        tp_n = sum(1 for d in closed_deals if float(getattr(d, "profit", 0.0)) > 0)
        sl_n = sum(1 for d in closed_deals if float(getattr(d, "profit", 0.0)) <= 0)
        profits_r = [_deal_to_profit_r(d, log_df) for d in closed_deals]

        out.update(
            ok=True,
            balance=balance,
            equity=equity,
            open=open_n,
            tp=tp_n,
            sl=sl_n,
            total_closed=tp_n + sl_n,
            profits_r=profits_r,
            source="mt5",
        )
        return out
    except Exception as exc:
        out["error"] = str(exc)
        return out
    finally:
        shutdown_mt5(mt5)


def summarize_live_trades(log_path: Path, *, days: int = 7) -> Dict[str, Any]:
    """Ringkasan live trades dari CSV (7 hari terakhir)."""
    out: Dict[str, Any] = {"days": days, "log_path": str(log_path), "source": "csv"}
    if not log_path.is_file():
        out["error"] = "no log file"
        return out

    df = pd.read_csv(log_path, engine="python", on_bad_lines="skip")
    if df.empty:
        out["error"] = "empty log"
        return out

    df["timestamp_signal"] = pd.to_datetime(df["timestamp_signal"], utc=True, errors="coerce")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    df = df[df["timestamp_signal"] >= cutoff].copy()
    if df.empty:
        out["error"] = "no rows in window"
        return out

    outcome = _normalize_outcomes(df["outcome"])
    tp_n = int((outcome == "TP").sum())
    sl_n = int((outcome == "SL").sum())
    open_n = int((outcome == "OPEN").sum())

    closed = df[outcome.isin(_CLOSED_OUTCOMES)].copy()
    pr = pd.to_numeric(closed["profit_r"], errors="coerce").dropna()
    profits_r = pr.tolist()

    out["tp"] = tp_n
    out["sl"] = sl_n
    out["open"] = open_n
    out["total"] = tp_n + sl_n
    out["profits_r"] = profits_r
    out["avg_profit_r"] = float(pr.mean()) if len(pr) else 0.0
    out["log_df"] = df
    return out


def build_trades_report(cfg: Dict[str, Any], *, days: int = 7) -> str:
    """Pesan plain text untuk /trades — CSV + MT5 backup, tanpa Markdown."""
    symbol = str(cfg.get("project", {}).get("symbol", "XAUUSD"))
    log_path = live_log_path(cfg)
    csv_sum = summarize_live_trades(log_path, days=days)
    log_df = csv_sum.get("log_df") if isinstance(csv_sum.get("log_df"), pd.DataFrame) else None

    mt5_sum = fetch_mt5_trade_stats(symbol, days=days, log_df=log_df)

    tp = int(csv_sum.get("tp", 0)) if not csv_sum.get("error") else 0
    sl = int(csv_sum.get("sl", 0)) if not csv_sum.get("error") else 0
    profits: List[float] = list(csv_sum.get("profits_r") or []) if not csv_sum.get("error") else []
    closed_csv = tp + sl

    if mt5_sum.get("ok"):
        if closed_csv == 0 and int(mt5_sum.get("total_closed", 0)) > 0:
            tp = int(mt5_sum.get("tp", 0))
            sl = int(mt5_sum.get("sl", 0))
            profits = list(mt5_sum.get("profits_r") or [])
        elif not profits and mt5_sum.get("profits_r"):
            profits = list(mt5_sum["profits_r"])
            if closed_csv == 0:
                tp = int(mt5_sum.get("tp", 0))
                sl = int(mt5_sum.get("sl", 0))

        open_count = int(mt5_sum.get("open", 0))
        balance = float(mt5_sum.get("balance", 0.0))
        equity = float(mt5_sum.get("equity", 0.0))
    else:
        open_count = int(csv_sum.get("open", 0)) if not csv_sum.get("error") else 0
        balance = 0.0
        equity = 0.0

    total = tp + sl
    tp_rate = tp / total if total > 0 else 0.0
    sl_rate = sl / total if total > 0 else 0.0
    avg_profit = sum(profits) / len(profits) if profits else 0.0
    profit_emoji = "📈" if avg_profit >= 0 else "📉"

    if total == 0 and open_count == 0 and csv_sum.get("error") and not mt5_sum.get("ok"):
        err = csv_sum.get("error", "no data")
        mt5_err = mt5_sum.get("error", "")
        hint = "Trade tercatat setelah konfirmasi ya + order MT5."
        if mt5_err:
            hint += f"\nMT5: {mt5_err}"
        return (
            f"📊 Live Trades ({days} hari terakhir)\n\n"
            f"Belum ada data ({err}).\n"
            f"{hint}"
        )

    return (
        f"📊 Live Trades ({days} hari terakhir)\n"
        f"\n"
        f"Total  : {total} trades\n"
        f"TP     : {tp} ({tp_rate:.0%})\n"
        f"SL     : {sl} ({sl_rate:.0%})\n"
        f"Open   : {open_count}\n"
        f"{profit_emoji} Avg P/L: {avg_profit:+.2f}R\n"
        f"\n"
        f"Balance: ${balance:,.2f}\n"
        f"Equity : ${equity:,.2f}"
    )


def format_trades_message(summary: Dict[str, Any], days: int = 7) -> str:
    """Deprecated — gunakan build_trades_report. Tetap ada untuk kompatibilitas."""
    if summary.get("error"):
        return (
            f"📊 Live Trades ({days} hari terakhir)\n\n"
            f"Belum ada data ({summary.get('error')})."
        )
    total = int(summary.get("total", 0))
    tp = int(summary.get("tp", 0))
    sl = int(summary.get("sl", 0))
    open_n = int(summary.get("open", 0))
    tp_rate = tp / total if total > 0 else 0.0
    sl_rate = sl / total if total > 0 else 0.0
    avg_r = float(summary.get("avg_profit_r", 0.0))
    profit_emoji = "📈" if avg_r >= 0 else "📉"
    return (
        f"📊 Live Trades ({days} hari terakhir)\n"
        f"\n"
        f"Total  : {total} trades\n"
        f"TP     : {tp} ({tp_rate:.0%})\n"
        f"SL     : {sl} ({sl_rate:.0%})\n"
        f"Open   : {open_n}\n"
        f"{profit_emoji} Avg P/L: {avg_r:+.2f}R"
    )
