"""Monitor posisi MT5 nyata sampai close (TP/SL/manual) — bukan simulasi level harga."""

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Optional, Set

from utils.live_tracker import live_log_path, update_outcome
from utils.mt5_connection import initialize_mt5, shutdown_mt5
from utils.paths import project_root
from utils.telegram_notify import send_telegram_message

log = logging.getLogger("position_monitor")

_ALERT_STATE_PATH = project_root() / "logs" / "moment_alert_state.json"
_ACTIVE_MONITORS: Set[int] = set()
_MONITOR_LOCK = threading.Lock()

DEAL_ENTRY_OUT = 1


def _setup_logging() -> None:
    if log.handlers:
        return
    log_dir = project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "stage9_service.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(fh)
    log.setLevel(logging.INFO)


def _resolve_mt5_symbol(mt5: Any, symbol: str) -> str:
    from utils.mt5_export import resolve_symbol

    return resolve_symbol(mt5, symbol)


def _deal_matches_symbol(deal: Any, resolved: str, base_symbol: str) -> bool:
    sym = str(getattr(deal, "symbol", "")).upper()
    return sym == resolved.upper() or sym == base_symbol.upper() or sym.startswith(base_symbol.upper())


def _position_matches(position: Any, ticket: int) -> bool:
    t = int(ticket)
    for attr in ("ticket", "identifier"):
        val = int(getattr(position, attr, 0) or 0)
        if val == t:
            return True
    return False


def is_position_still_open(
    ticket: int,
    symbol: str,
    *,
    mt5: Any | None = None,
) -> bool:
    """
    Cek apakah posisi masih terbuka.
    Mencoba ticket langsung, scan semua posisi symbol, dan optional magic.
    """
    own_mt5 = False
    if mt5 is None:
        try:
            ok, mt5 = initialize_mt5()
        except RuntimeError:
            return False
        if not ok:
            return False
        own_mt5 = True

    try:
        resolved = _resolve_mt5_symbol(mt5, symbol)
        t = int(ticket)

        positions = mt5.positions_get(ticket=t)
        if positions and len(positions) > 0:
            return True

        all_positions = mt5.positions_get(symbol=resolved)
        if not all_positions:
            all_positions = mt5.positions_get(symbol=symbol)

        if all_positions:
            for p in all_positions:
                if _position_matches(p, t):
                    return True

        return False
    finally:
        if own_mt5:
            shutdown_mt5(mt5)


def find_close_deal(
    ticket: int,
    symbol: str,
    *,
    mt5: Any | None = None,
    lookback_minutes: int = 30,
    allow_symbol_fallback: bool = True,
) -> Any | None:
    """
    Cari deal close untuk ticket/position tertentu.
    Return deal object atau None.
    """
    own_mt5 = False
    if mt5 is None:
        try:
            ok, mt5 = initialize_mt5()
        except RuntimeError:
            return None
        if not ok:
            return None
        own_mt5 = True

    try:
        resolved = _resolve_mt5_symbol(mt5, symbol)
        from_time = datetime.now(timezone.utc) - timedelta(minutes=lookback_minutes)
        to_time = datetime.now(timezone.utc)

        deals = mt5.history_deals_get(from_time, to_time)
        if not deals:
            return None

        close_deals = [
            d
            for d in deals
            if int(getattr(d, "entry", -1)) == DEAL_ENTRY_OUT
            and _deal_matches_symbol(d, resolved, symbol)
        ]
        if not close_deals:
            return None

        t = int(ticket)
        for deal in close_deals:
            if int(getattr(deal, "order", 0) or 0) == t:
                return deal
            if int(getattr(deal, "position_id", 0) or 0) == t:
                return deal

        if allow_symbol_fallback and len(close_deals) == 1:
            return close_deals[0]

        if allow_symbol_fallback and close_deals:
            log.debug(
                "find_close_deal: fallback deal terbaru symbol=%s ticket=%s",
                symbol,
                ticket,
            )
            return max(close_deals, key=lambda d: float(getattr(d, "time", 0)))

        return None
    finally:
        if own_mt5:
            shutdown_mt5(mt5)


def set_moment_alert_active(
    *,
    ticket: int,
    symbol: str,
    timeframe: str,
    side: str,
    entry: float,
    sl: float,
    tp: float,
) -> None:
    """Tandai cooldown aktif (ada posisi nyata) — blok moment alert baru."""
    state: Dict[str, Any] = {}
    if _ALERT_STATE_PATH.is_file():
        try:
            state = json.loads(_ALERT_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    state["active"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ticket": int(ticket),
        "symbol": symbol,
        "timeframe": timeframe,
        "side": side,
        "entry": entry,
        "sl": sl,
        "tp": tp,
    }
    _ALERT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ALERT_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def clear_moment_alert_active() -> None:
    if not _ALERT_STATE_PATH.is_file():
        return
    try:
        state = json.loads(_ALERT_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return
    state["active"] = {}
    _ALERT_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def run_cooldown_monitor(
    trade_plan: dict,
    telegram_send_fn: Optional[Callable[[str], None]],
    cfg: dict,
    *,
    poll_seconds: int = 30,
    max_wait_hours: int = 4,
    timestamp_signal: Optional[str] = None,
) -> str:
    """
    Monitor posisi MT5 sampai close.
    Return: TP, SL, MANUAL_CLOSE, TIMEOUT, NO_TICKET.
    """
    _setup_logging()

    ticket = int(trade_plan.get("ticket") or 0)
    symbol = str(trade_plan.get("symbol", cfg.get("project", {}).get("symbol", "XAUUSD")))

    if ticket <= 0:
        log.warning("Monitor: tidak ada ticket — skip")
        return "NO_TICKET"

    max_seconds = max_wait_hours * 3600
    elapsed = 0
    check_count = 0
    poll_seconds = max(15, int(poll_seconds))

    log.info("Monitor start | ticket=%s symbol=%s poll=%ds", ticket, symbol, poll_seconds)

    while elapsed < max_seconds:
        time.sleep(poll_seconds)
        elapsed += poll_seconds
        check_count += 1

        ok, _mt5 = initialize_mt5(cfg)
        if not ok:
            log.warning("Monitor: MT5 tidak terkoneksi (check #%d)", check_count)
            continue
        mt5 = _mt5

        try:
            still_open = is_position_still_open(
                ticket,
                symbol,
                mt5=mt5,
            )

            if still_open:
                if check_count % 10 == 0:
                    resolved = _resolve_mt5_symbol(mt5, symbol)
                    positions = mt5.positions_get(symbol=resolved) or mt5.positions_get(symbol=symbol)
                    profit = 0.0
                    if positions:
                        for p in positions:
                            if _position_matches(p, ticket):
                                profit = float(getattr(p, "profit", 0.0))
                                break
                    log.info(
                        "Monitor: posisi #%s masih terbuka | profit=%.2f | elapsed=%.0f menit",
                        ticket,
                        profit,
                        elapsed / 60,
                    )
                continue

            log.info("Monitor: posisi #%s tidak ditemukan — cari deal close", ticket)

            close_deal = find_close_deal(
                ticket,
                symbol,
                mt5=mt5,
                lookback_minutes=30,
                allow_symbol_fallback=True,
            )

            account = mt5.account_info()
            balance = float(account.balance) if account is not None else 0.0

            if close_deal:
                profit = float(getattr(close_deal, "profit", 0.0))
                close_price = float(getattr(close_deal, "price", 0.0))
                close_type = "TP" if profit > 0 else "SL"
                emoji = "✅" if close_type == "TP" else "❌"
                profit_emoji = "📈" if profit > 0 else "📉"
                profit_r = float(trade_plan.get("rr_actual") or trade_plan.get("risk_reward") or 1.0)
                if profit <= 0:
                    profit_r = -1.0

                msg = (
                    f"{emoji} Trade Close — {close_type}\n"
                    f"Symbol: {symbol}\n"
                    f"Close price: {close_price:.2f}\n"
                    f"{profit_emoji} P/L: ${profit:+.2f}\n"
                    f"Balance: ${balance:,.2f}\n"
                    f"Ticket: #{ticket}"
                )
                log.info(
                    "Monitor: close detected | type=%s | profit=%.2f | balance=%.2f",
                    close_type,
                    profit,
                    balance,
                )
                if telegram_send_fn:
                    telegram_send_fn(msg)
                if timestamp_signal:
                    try:
                        update_outcome(
                            live_log_path(cfg),
                            timestamp_signal,
                            close_type,
                            close_price,
                            profit_r,
                        )
                    except Exception as exc:
                        log.error("update_outcome gagal: %s", exc)
                clear_moment_alert_active()
                return close_type

            log.warning("Monitor: posisi close tapi tidak ada deal | ticket=%s", ticket)
            if telegram_send_fn:
                telegram_send_fn(
                    f"⚠️ Posisi #{ticket} sudah close\n"
                    f"Balance: ${balance:,.2f}\n"
                    "Cek History MT5 untuk detail."
                )
            clear_moment_alert_active()
            return "MANUAL_CLOSE"
        finally:
            shutdown_mt5(mt5)

    log.warning("Monitor: timeout %d jam | ticket=%s", max_wait_hours, ticket)
    if telegram_send_fn:
        telegram_send_fn(
            f"⏰ Posisi #{ticket} belum close setelah {max_wait_hours} jam\n"
            "Pertimbangkan tutup manual."
        )
    return "TIMEOUT"


def start_position_monitor_thread(
    cfg: dict,
    trade_plan: dict,
    *,
    token: str,
    chat_id: str,
    timestamp_signal: Optional[str] = None,
) -> None:
    """Jalankan monitor di thread daemon — tidak block polling bot."""
    ticket = int(trade_plan.get("ticket") or 0)
    if ticket <= 0:
        log.warning("start_position_monitor_thread: ticket invalid — skip")
        return

    with _MONITOR_LOCK:
        if ticket in _ACTIVE_MONITORS:
            log.info("Monitor ticket=%s sudah berjalan — skip duplikat", ticket)
            return
        _ACTIVE_MONITORS.add(ticket)

    moment_cfg = cfg.get("stage_9", {}).get("moment_alert", {})
    if isinstance(moment_cfg, dict):
        poll_seconds = int(moment_cfg.get("monitor_poll_seconds", 30))
    else:
        poll_seconds = 30

    def _telegram_send(msg: str) -> None:
        send_telegram_message(token, chat_id, msg, parse_mode=None)

    def _run() -> None:
        result = "NO_TICKET"
        try:
            result = run_cooldown_monitor(
                trade_plan,
                _telegram_send,
                cfg,
                poll_seconds=poll_seconds,
                timestamp_signal=timestamp_signal,
            )
            if result in {"TP", "SL", "MANUAL_CLOSE"}:
                _telegram_send(
                    f"🔔 Cooldown selesai ({result})\n"
                    f"• {trade_plan.get('symbol', 'XAUUSD')} ticket #{ticket}\n"
                    "Moment alert siap kirim sinyal berikutnya."
                )
        except Exception as exc:
            log.error("Monitor thread error ticket=%s: %s", ticket, exc, exc_info=True)
        finally:
            with _MONITOR_LOCK:
                _ACTIVE_MONITORS.discard(ticket)

    threading.Thread(
        target=_run,
        name=f"mt5-monitor-{ticket}",
        daemon=True,
    ).start()
    log.info("Monitor thread started | ticket=%s poll=%ds", ticket, poll_seconds)
