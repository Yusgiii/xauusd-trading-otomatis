"""Bot Telegram — analisa hanya saat user memanggil command."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from utils.paths import project_root, resolve_pipeline_run_dir
from utils.mt5_connection import initialize_mt5, shutdown_mt5
from utils.mt5_execution import place_market_order_from_plan
from utils.position_monitor import set_moment_alert_active, start_position_monitor_thread
from utils.live_tracker import build_trades_report, live_log_path, log_signal
from utils.mt5_export import mt5_timeframe_const, resolve_symbol
from utils.telegram_notify import is_placeholder, send_telegram_message, telegram_get_updates

WIB = ZoneInfo("Asia/Jakarta")

CMD_ANALISA = {"/analisa"}
CMD_START = {"/start"}
CMD_STATUS = {"/status"}
CMD_TRADES = {"/trades"}
CMD_AKUN = {"/akun"}
VALID_TIMEFRAMES = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
YES_WORDS = {"ya", "y", "yes", "ok", "gas", "masuk"}
NO_WORDS = {"tidak", "ga", "gak", "no", "n", "batal", "skip"}

log = logging.getLogger(__name__)

# Shared antrian konfirmasi — dipakai thread polling & moment_alert
_PENDING_EXEC: Dict[str, Dict[str, Any]] = {}


def _setup_telegram_logging() -> None:
    if log.handlers:
        return
    log_dir = project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_dir / "stage9_bot_errors.log", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)
    log.setLevel(logging.INFO)


def _safe_telegram_send(
    token: str,
    chat_id: str,
    text: str,
    *,
    parse_mode: Optional[str] = None,
) -> None:
    try:
        send_telegram_message(token, chat_id, text, parse_mode=parse_mode)
    except Exception as exc:
        log.error("Telegram send gagal: %s", exc, exc_info=True)


def register_pending_execution(chat_id: str, payload: Dict[str, Any]) -> None:
    """Daftarkan trade plan menunggu konfirmasi user (harus sebelum user balas ya)."""
    _PENDING_EXEC[str(chat_id)] = payload
    tp = payload.get("trade_plan", {}) if isinstance(payload.get("trade_plan"), dict) else {}
    log.info(
        "Pending exec diset | chat=%s | side=%s | expires=%s",
        chat_id,
        tp.get("side"),
        payload.get("expires_at"),
    )


def get_pending_execution(chat_id: str) -> Optional[Dict[str, Any]]:
    return _PENDING_EXEC.get(str(chat_id))


def clear_pending_execution(chat_id: str) -> None:
    _PENDING_EXEC.pop(str(chat_id), None)


def _text_is_confirmation_yes(text_norm: str) -> bool:
    if not text_norm:
        return False
    if text_norm in YES_WORDS:
        return True
    first = text_norm.split()[0]
    return first in YES_WORDS


def _text_is_confirmation_no(text_norm: str) -> bool:
    if not text_norm:
        return False
    if text_norm in NO_WORDS:
        return True
    first = text_norm.split()[0]
    return first in NO_WORDS


def _handle_confirm_yes(
    cfg: Dict[str, Any],
    token: str,
    chat_id: str,
    pend: Dict[str, Any],
    pending_exec: Dict[str, Dict[str, Any]],
) -> None:
    """Eksekusi MT5 setelah user balas ya — semua error di-catch, tidak re-raise."""
    try:
        log.info("=== KONFIRMASI YA DITERIMA === chat_id=%s", chat_id)
        print("[bot] === KONFIRMASI YA DITERIMA ===", flush=True)
        log.info("Pending exec payload: %s", pend)

        exec_cfg = cfg.get("stage_9", {}).get("execution", {})
        if not bool(exec_cfg.get("enabled", True)):
            log.warning("Eksekusi MT5 disabled di config")
            _safe_telegram_send(
                token,
                chat_id,
                "⛔ Eksekusi MT5 dinonaktifkan di config (`stage_9.execution.enabled=false`).",
            )
            return

        symbol_exec = str(pend.get("symbol", cfg.get("project", {}).get("symbol", "XAUUSD")))
        tp = pend.get("trade_plan", {}) if isinstance(pend.get("trade_plan", {}), dict) else {}
        log.info("Trade plan: side=%s sl=%s tp=%s lot=%s", tp.get("side"), tp.get("sl"), tp.get("tp"), tp.get("lot_size"))

        if not tp or tp.get("side") not in {"BUY", "SELL"}:
            log.error("Trade plan tidak valid untuk order: %s", tp)
            _safe_telegram_send(token, chat_id, "⚠️ Trade plan tidak valid. Jalankan /analisa lagi.")
            return

        if tp.get("sl") is None or tp.get("tp") is None:
            log.error("SL/TP kosong pada pending trade_plan")
            _safe_telegram_send(token, chat_id, "⚠️ SL/TP kosong — jalankan /analisa lagi.")
            return

        if not bool(pend.get("meta_allow", True)):
            log.warning("meta_allow=False — order ditolak")
            _safe_telegram_send(
                token,
                chat_id,
                "⛔ Order ditolak: meta-filter tidak lolos pada sinyal ini.",
            )
            return

        _safe_telegram_send(
            token,
            chat_id,
            f"📤 Menempatkan order {tp.get('side', 'NONE')} {symbol_exec} ke MT5…",
        )

        log.info("Memanggil place_market_order_from_plan symbol=%s side=%s", symbol_exec, tp.get("side"))
        print("[bot] Memanggil place_market_order_from_plan...", flush=True)
        rs = place_market_order_from_plan(
            cfg,
            symbol=symbol_exec,
            trade_plan=tp,
            comment=f"stage9_{chat_id}",
        )
        log.info("Hasil MT5: %s", rs)
        print(f"[bot] Order result: ok={rs.get('ok')} error={rs.get('error', '')}", flush=True)

        if rs.get("ok"):
            try:
                ts_sig = str(pend.get("timestamp_signal") or datetime.now(timezone.utc).isoformat())
                log_signal(
                    live_log_path(cfg),
                    {
                        "timestamp_signal": ts_sig,
                        "timestamp_entry": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol_exec,
                        "side": tp.get("side"),
                        "lot_size": rs.get("volume", tp.get("lot_size")),
                        "confidence_tier": tp.get("confidence_tier", ""),
                        "confidence_score": tp.get("confidence", 0.0),
                        "meta_score": tp.get("meta_score", ""),
                        "p_up_binary": tp.get("p_up", 0.0),
                        "p_down_binary": tp.get("p_down", 0.0),
                        "entry_price": rs.get("price", tp.get("entry")),
                        "sl_price": tp.get("sl"),
                        "tp_price": tp.get("tp"),
                        "rr_planned": tp.get("rr_actual", tp.get("risk_reward")),
                        "drift_level": tp.get("drift_level", "NORMAL"),
                        "confirmed_by_user": "yes",
                        "outcome": "OPEN",
                        "run_id": str(pend.get("run_id", "")),
                    },
                )
            except Exception as log_exc:
                log.error("live_trade_log error: %s", log_exc, exc_info=True)
                print(f"[bot] live_trade_log error: {log_exc}", flush=True)

            price = float(rs.get("price", 0.0))
            vol = rs.get("volume", tp.get("lot_size"))
            ticket = int(rs.get("ticket") or rs.get("order") or 0)
            tp["ticket"] = ticket
            _safe_telegram_send(
                token,
                chat_id,
                (
                    f"✅ Order masuk: {rs.get('side')} {vol} {rs.get('symbol')} @ {price:.5f}\n"
                    f"Ticket: #{ticket}"
                ),
                parse_mode=None,
            )
            if ticket > 0:
                tf = str(
                    pend.get("timeframe", cfg.get("project", {}).get("timeframe", "M15"))
                ).upper()
                set_moment_alert_active(
                    ticket=ticket,
                    symbol=symbol_exec,
                    timeframe=tf,
                    side=str(tp.get("side", "")),
                    entry=float(rs.get("price", tp.get("entry", 0.0))),
                    sl=float(tp.get("sl", 0.0)),
                    tp=float(tp.get("tp", 0.0)),
                )
                monitor_plan = {
                    **tp,
                    "symbol": symbol_exec,
                    "ticket": ticket,
                }
                start_position_monitor_thread(
                    cfg,
                    monitor_plan,
                    token=token,
                    chat_id=chat_id,
                    timestamp_signal=ts_sig,
                )
            else:
                log.warning("Order ok tetapi ticket=0 — monitor posisi tidak dijalankan")
        else:
            _safe_telegram_send(
                token,
                chat_id,
                f"❌ Order gagal: {rs.get('error', rs)}",
            )
    except Exception as exc:
        log.error("Error eksekusi order: %s", exc, exc_info=True)
        print(f"[bot] Error eksekusi order: {exc}", flush=True)
        _safe_telegram_send(token, chat_id, f"❌ Error eksekusi order: {str(exc)[:200]}")
    finally:
        clear_pending_execution(chat_id)
        pending_exec.pop(chat_id, None)


def parse_command(text: Optional[str]) -> Tuple[Optional[str], List[str]]:
    if not text:
        return None, []
    t = text.strip()
    if not t.startswith("/"):
        return None, []
    parts = t.split()
    cmd = parts[0].split("@")[0].lower()
    return cmd, parts[1:]


def allowed_chat_ids(cfg: Dict[str, Any]) -> Set[str]:
    risk = cfg.get("risk", {})
    bot = cfg.get("stage_9", {}).get("bot", {})
    ids: Set[str] = set()
    primary = str(risk.get("telegram_chat_id", "")).strip()
    if primary and not is_placeholder(primary):
        ids.add(primary)
    for cid in bot.get("allowed_chat_ids") or []:
        s = str(cid).strip()
        if s and not is_placeholder(s):
            ids.add(s)
    return ids


def _suggest_symbols(mt5, symbol: str, max_items: int = 5) -> List[str]:
    s = symbol.upper()
    names = [str(x.name) for x in (mt5.symbols_get() or []) if getattr(x, "name", None)]
    starts = [n for n in names if n.upper().startswith(s)]
    contains = [n for n in names if s in n.upper() and n not in starts]
    return (starts + contains)[:max_items]


def validate_mt5_inputs(symbol: str, timeframe: str) -> Tuple[bool, str]:
    """Validasi simbol + timeframe di MT5 sebelum inferensi."""
    try:
        ok, mt5 = initialize_mt5()
    except RuntimeError:
        return True, ""

    if not ok:
        return False, f"MT5 belum siap/login: {mt5.last_error()}"
    try:
        resolved = resolve_symbol(mt5, symbol)
        try:
            mt5_timeframe_const(mt5, timeframe)
        except Exception as exc:
            return False, f"Timeframe `{timeframe}` tidak didukung MT5: {exc}"
        return True, resolved
    except Exception as exc:
        sugg = _suggest_symbols(mt5, symbol)
        if sugg:
            return False, f"{exc}\nSaran simbol: {', '.join(sugg)}"
        return False, str(exc)
    finally:
        shutdown_mt5(mt5)


def _mt5_account_summary(symbol: str = "XAUUSD") -> Tuple[bool, str]:
    """Ringkasan akun MT5 (balance/equity + P/L harian)."""
    try:
        ok, mt5 = initialize_mt5()
    except RuntimeError as exc:
        return False, f"❌ MetaTrader5 tidak terpasang: {exc}"

    if not ok:
        return False, f"❌ MT5 tidak terkoneksi: {mt5.last_error()}"

    try:
        account = mt5.account_info()
        if account is None:
            return False, "❌ Tidak bisa ambil info akun MT5."

        positions = mt5.positions_get(symbol=symbol)
        n_open = len(positions) if positions else 0
        open_profit = float(sum(float(getattr(p, "profit", 0.0)) for p in positions) if positions else 0.0)

        now_utc = datetime.now(timezone.utc)
        day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(day_start, now_utc)
        today_profit = 0.0
        today_trades = 0
        today_win = 0
        today_loss = 0
        if deals:
            closed = [d for d in deals if int(getattr(d, "entry", -1)) == 1]
            today_profit = float(sum(float(getattr(d, "profit", 0.0)) for d in closed))
            today_trades = len(closed)
            today_win = sum(1 for d in closed if float(getattr(d, "profit", 0.0)) > 0)
            today_loss = sum(1 for d in closed if float(getattr(d, "profit", 0.0)) < 0)

        profit_emoji = "📈" if today_profit >= 0 else "📉"
        open_emoji = "🟡" if n_open > 0 else "⚪"
        lines = [
            "💰 *Info Akun MT5*",
            f"• Balance: *${float(account.balance):,.2f}*",
            f"• Equity: *${float(account.equity):,.2f}*",
            f"• Free Margin: ${float(account.margin_free):,.2f}",
            "",
            f"{open_emoji} *Posisi Terbuka: {n_open}*",
        ]
        if n_open > 0:
            lines.append(f"• Floating P/L: ${open_profit:+.2f}")
            for p in (positions or [])[:5]:
                side = "BUY" if int(getattr(p, "type", 1)) == 0 else "SELL"
                lines.append(
                    f"  - {side} {float(getattr(p, 'volume', 0.0)):.2f} @ "
                    f"{float(getattr(p, 'price_open', 0.0)):.2f} → P/L: ${float(getattr(p, 'profit', 0.0)):+.2f}"
                )

        lines.extend(
            [
                "",
                f"{profit_emoji} *Hari Ini ({today_trades} trade)*",
                f"• P/L: *${today_profit:+.2f}*",
                f"• Win: {today_win} | Loss: {today_loss}",
            ]
        )
        if today_trades > 0:
            lines.append(f"• Winrate: {today_win / max(1, today_trades) * 100:.0f}%")

        return True, "\n".join(lines)
    except Exception as exc:
        log.error("cmd_akun error: %s", exc, exc_info=True)
        return False, f"❌ Error akun MT5: {str(exc)[:200]}"
    finally:
        shutdown_mt5(mt5)


def run_bot_polling(
    cfg: Dict[str, Any],
    run_dir: Path,
    *,
    run_analysis: Callable[..., Dict[str, Any]],
    skip_mt5: bool = False,
) -> None:
    """
    Loop getUpdates: tanggapi perintah user.
    `run_analysis(cfg, run_dir, skip_mt5=..., target_chat_id=..., send_telegram=...)`
    """
    risk = cfg.get("risk", {})
    token = str(risk.get("telegram_token", ""))
    if is_placeholder(token):
        raise RuntimeError("telegram_token belum diisi di pipeline.secrets.yaml")

    _setup_telegram_logging()

    bot_cfg = cfg.get("stage_9", {}).get("bot", {})
    poll_timeout = int(bot_cfg.get("poll_timeout_sec", 30))
    cooldown = int(bot_cfg.get("command_cooldown_sec", 45))
    allowed = allowed_chat_ids(cfg)

    offset: Optional[int] = None
    last_run: Dict[str, float] = {}
    pending_exec = _PENDING_EXEC

    output_root = str(cfg.get("experiment", {}).get("output_root", "artifacts"))

    run_holder: List[Path] = [run_dir]

    def _active_run_dir() -> Path:
        resolved = resolve_pipeline_run_dir(run_holder[0], output_root=output_root)
        run_holder[0] = resolved
        return resolved

    try:
        run_dir = _active_run_dir()
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc

    print(
        f"Bot polling aktif | run_dir={run_dir} | allowed_chats={allowed or 'semua (tidak disarankan)'}",
        flush=True,
    )
    print(
        "Perintah: /analisa /status /trades /akun — Ctrl+C berhenti",
        flush=True,
    )

    while True:
        try:
            data = telegram_get_updates(token, offset=offset, timeout=poll_timeout)
        except Exception as exc:
            print(f"[bot] getUpdates error: {exc}")
            time.sleep(5)
            continue

        if not data.get("ok"):
            print(f"[bot] API error: {data}")
            time.sleep(5)
            continue

        for upd in data.get("result", []):
            offset = int(upd["update_id"]) + 1
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue

            chat = msg.get("chat") or {}
            chat_id = str(chat.get("id", ""))
            text = msg.get("text") or ""
            text_norm = str(text).strip().lower()
            now = time.time()

            try:
                # Tangani jawaban konfirmasi eksekusi (ya/tidak) sebelum parse command.
                pend = pending_exec.get(chat_id)
                if pend:
                    expires_at = float(pend.get("expires_at", 0.0))
                    if now > expires_at:
                        clear_pending_execution(chat_id)
                        pending_exec.pop(chat_id, None)
                        _safe_telegram_send(
                            token,
                            chat_id,
                            "⌛ Konfirmasi entry kedaluwarsa. Jalankan /analisa lagi bila ingin entry baru.",
                        )
                    elif _text_is_confirmation_yes(text_norm):
                        _handle_confirm_yes(cfg, token, chat_id, pend, pending_exec)
                        continue
                    elif _text_is_confirmation_no(text_norm):
                        try:
                            ts_sig = str(pend.get("timestamp_signal", ""))
                            if ts_sig:
                                log_signal(
                                    live_log_path(cfg),
                                    {
                                        "timestamp_signal": ts_sig,
                                        "symbol": str(pend.get("symbol", "")),
                                        "side": pend.get("trade_plan", {}).get("side", ""),
                                        "confirmed_by_user": "no",
                                        "outcome": "CANCELLED",
                                        "run_id": str(pend.get("run_id", "")),
                                    },
                                )
                        except Exception as log_exc:
                            log.error("live_trade_log cancel error: %s", log_exc, exc_info=True)
                        pending_exec.pop(chat_id, None)
                        _safe_telegram_send(
                            token,
                            chat_id,
                            "✅ Oke, tidak ada order yang dikirim.",
                        )
                        continue

                if _text_is_confirmation_yes(text_norm) or _text_is_confirmation_no(text_norm):
                    log.warning(
                        "Konfirmasi '%s' tanpa pending exec (chat=%s) — abaikan atau minta /analisa",
                        text_norm,
                        chat_id,
                    )
                    _safe_telegram_send(
                        token,
                        chat_id,
                        "⚠️ Tidak ada sinyal pending untuk dieksekusi.\n"
                        "Jalankan /analisa dulu, tunggu pesan *Konfirmasi entry*, lalu balas `ya`.",
                    )
                    continue

                cmd, cmd_args = parse_command(text)

                if not cmd:
                    continue

                if allowed and chat_id not in allowed:
                    _safe_telegram_send(
                        token,
                        chat_id,
                        "⛔ Chat tidak diizinkan. Hubungi admin untuk mendaftarkan chat_id Anda.",
                    )
                    continue

                if cmd in CMD_ANALISA:
                    if now - last_run.get(chat_id, 0) < cooldown:
                        wait = int(cooldown - (now - last_run.get(chat_id, 0)))
                        _safe_telegram_send(
                            token,
                            chat_id,
                            f"⏳ Tunggu {wait} detik sebelum meminta analisis lagi.",
                        )
                        continue

                    last_run[chat_id] = now
                    symbol_override: Optional[str] = None
                    timeframe_override: Optional[str] = None
                    if cmd_args:
                        for raw in cmd_args[:2]:
                            a = str(raw).strip().upper()
                            if not a:
                                continue
                            if a in VALID_TIMEFRAMES:
                                timeframe_override = a
                            else:
                                symbol_override = a

                    sym_disp = symbol_override or str(cfg.get("project", {}).get("symbol", "XAUUSD"))
                    tf_disp = timeframe_override or str(cfg.get("project", {}).get("timeframe", "H1")).upper()
                    if not skip_mt5:
                        ok_sym, sym_note = validate_mt5_inputs(sym_disp, tf_disp)
                        if not ok_sym:
                            _safe_telegram_send(
                                token,
                                chat_id,
                                f"❌ Input tidak valid di MT5 untuk `{sym_disp} {tf_disp}`.\n{sym_note}",
                            )
                            continue
                    _safe_telegram_send(
                        token,
                        chat_id,
                        f"⏳ Menganalisis {sym_disp} {tf_disp}… (MT5 + model + berita)",
                    )
                    try:
                        active = _active_run_dir()
                        rep = run_analysis(
                            cfg,
                            run_dir=active,
                            skip_mt5=skip_mt5,
                            target_chat_id=chat_id,
                            send_telegram=False,
                            symbol_override=symbol_override,
                            timeframe_override=timeframe_override,
                        )
                        message = str(rep.get("message", ""))
                        if message:
                            tg_main = send_telegram_message(token, chat_id, message)
                            if not tg_main.get("ok"):
                                err = rep.get("telegram_error") or tg_main
                                _safe_telegram_send(
                                    token,
                                    chat_id,
                                    f"Analisis selesai tetapi gagal kirim format penuh: {err}",
                                )

                        tp = rep.get("trade_plan", {}) if isinstance(rep.get("trade_plan", {}), dict) else {}
                        meta_dec = rep.get("meta_decision", {}) if isinstance(rep.get("meta_decision"), dict) else {}
                        meta_allow = bool(meta_dec.get("allow_execute", True))
                        rec_u = str(rep.get("recommendation", "")).upper()
                        if (
                            tp
                            and not bool(tp.get("is_no_trade", True))
                            and tp.get("side") in {"BUY", "SELL"}
                            and meta_allow
                            and "KONFLIK" not in rec_u
                        ):
                            timeout_sec = int(
                                cfg.get("stage_9", {}).get("execution", {}).get("confirm_timeout_sec", 120)
                            )
                            payload = {
                                "symbol": sym_disp,
                                "timeframe": tf_disp,
                                "trade_plan": tp,
                                "run_id": active.name,
                                "timestamp_signal": datetime.now(timezone.utc).isoformat(),
                                "meta_allow": meta_allow,
                                "created_at": now,
                                "expires_at": now + max(30, timeout_sec),
                            }
                            register_pending_execution(chat_id, payload)
                            _safe_telegram_send(
                                token,
                                chat_id,
                                (
                                    "⚠️ Konfirmasi entry:\n"
                                    f"Masuk posisi *{tp.get('side')}* untuk {sym_disp} {tf_disp}?\n"
                                    f"• Entry: {float(tp.get('entry', 0.0)):.5f}\n"
                                    f"• SL: {float(tp.get('sl', 0.0)):.5f}\n"
                                    f"• TP: {float(tp.get('tp', 0.0)):.5f}\n"
                                    f"• RR: 1:{float(tp.get('risk_reward', 2.0)):.2f}\n\n"
                                    f"Balas `ya` untuk eksekusi, `tidak` untuk batal "
                                    f"(timeout {max(30, timeout_sec)} detik)."
                                ),
                            )
                        else:
                            log.info(
                                "Tidak set pending exec | is_no_trade=%s side=%s meta_allow=%s rec=%s",
                                tp.get("is_no_trade"),
                                tp.get("side"),
                                meta_allow,
                                rec_u,
                            )
                    except Exception as exc:
                        log.error("Analisis gagal: %s", exc, exc_info=True)
                        _safe_telegram_send(
                            token,
                            chat_id,
                            f"❌ Analisis gagal: {exc}",
                        )
                    continue

                if cmd in CMD_START:
                    _safe_telegram_send(
                        token,
                        chat_id,
                        (
                            "Halo! Bot analisis XAUUSD aktif.\n\n"
                            "• /analisa — sinyal terbaru\n"
                            "• /status — status model & drift\n"
                            "• /trades — ringkasan live trades\n"
                            "• /akun — info saldo/equity MT5"
                        ),
                    )
                    continue

                if cmd in CMD_STATUS:
                    from stage_9_live_demo import format_system_status

                    active = _active_run_dir()
                    _safe_telegram_send(
                        token,
                        chat_id,
                        format_system_status(cfg, active, skip_mt5=skip_mt5),
                    )
                    continue

                if cmd in CMD_TRADES:
                    days = 7
                    if cmd_args and str(cmd_args[0]).isdigit():
                        days = max(1, min(int(cmd_args[0]), 90))
                    msg = build_trades_report(cfg, days=days)
                    _safe_telegram_send(token, chat_id, msg, parse_mode=None)
                    continue

                if cmd in CMD_AKUN:
                    ok, account_msg = _mt5_account_summary(
                        symbol=str(cfg.get("project", {}).get("symbol", "XAUUSD"))
                    )
                    _safe_telegram_send(token, chat_id, account_msg)
                    continue

            except Exception as exc:
                log.error("Handler error: %s", exc, exc_info=True)
                print(f"[bot] Handler error: {exc}", flush=True)
                _safe_telegram_send(token, chat_id, f"❌ Error: {str(exc)[:100]}")
