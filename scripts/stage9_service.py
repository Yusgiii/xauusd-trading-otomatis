# noqa: D100
"""
Layanan Stage 9 — bot Telegram on-demand (/analisa).

Jalankan sekali, biarkan terminal terbuka:
  python scripts/stage9_service.py --latest-run
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo
import atexit

WIB = ZoneInfo("Asia/Jakarta")
_LAST_NO_TRADE_NOTIF: Optional[datetime] = None
_LAST_POS_OPEN_LOG: dict[int, float] = {}

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from stage_9_live_demo import _maybe_mt5, run_daily_report
from utils.bar_timer import (
    bar_open_time,
    is_new_bar,
    seconds_until_next_bar,
    timeframe_to_bar_minutes,
)
from utils.config_loader import load_pipeline_config
from utils.paths import resolve_pipeline_run_dir, project_root
from utils.retrain_scheduler import (
    is_retrain_allowed_now,
    load_last_retrain_time,
    load_runtime_risk,
    run_retrain,
    save_last_retrain_time,
    should_retrain,
)
from utils.trading_hours import is_market_open, seconds_until_market_open
from utils.telegram_bot import register_pending_execution, run_bot_polling
from utils.mt5_connection import initialize_mt5, set_mt5_terminal_path, get_mt5_terminal_path, shutdown_mt5
from utils.position_monitor import clear_moment_alert_active, is_position_still_open
from utils.telegram_notify import is_placeholder, send_telegram_message, telegram_set_commands

_LOCK_FH = None
_ALERT_STATE_PATH = ROOT / "logs" / "moment_alert_state.json"
_MARKET_CLOSED_NOTIF_FLAG = project_root() / "logs" / "market_closed_notif.flag"


def _acquire_single_instance_lock(project_dir: Path) -> None:
    """Cegah dua instance stage9_service berjalan bersamaan (Windows lock file)."""
    global _LOCK_FH
    lock_path = project_dir / "logs" / "stage9_service.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = lock_path.open("a+", encoding="utf-8")
    fh.seek(0)
    if fh.tell() == 0:
        fh.write("0")
        fh.flush()
    fh.seek(0)
    try:
        import msvcrt  # Windows-only

        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        fh.close()
        raise RuntimeError("stage9_service sudah berjalan. Hentikan instance lama terlebih dulu.")
    _LOCK_FH = fh

    def _release_lock() -> None:
        try:
            if _LOCK_FH is None:
                return
            _LOCK_FH.seek(0)
            msvcrt.locking(_LOCK_FH.fileno(), msvcrt.LK_UNLCK, 1)
            _LOCK_FH.close()
        except Exception:
            pass

    atexit.register(_release_lock)


def _release_single_instance_lock() -> None:
    """Lepas lock file (dipanggil sebelum spawn proses service baru)."""
    global _LOCK_FH
    if _LOCK_FH is None:
        return
    try:
        import msvcrt

        _LOCK_FH.seek(0)
        msvcrt.locking(_LOCK_FH.fileno(), msvcrt.LK_UNLCK, 1)
        _LOCK_FH.close()
    except Exception:
        pass
    _LOCK_FH = None


def _spawn_fresh_service_instance() -> None:
    """Spawn stage9_service baru lalu exit — hindari hot-reload model di memory."""
    flag = ROOT / "logs" / "needs_restart.flag"
    if not flag.is_file():
        return

    try:
        flag.unlink()
    except OSError:
        pass

    script = ROOT / "scripts" / "stage9_service.py"
    argv = [sys.executable, "-u", str(script), *sys.argv[1:]]
    print(f"[auto_retrain] Spawn service baru: {' '.join(argv)}", flush=True)

    stdout_log = ROOT / "logs" / "stage9_service_stdout.log"
    stderr_log = ROOT / "logs" / "stage9_service_stderr.log"
    with stdout_log.open("a", encoding="utf-8") as out_f, stderr_log.open("a", encoding="utf-8") as err_f:
        subprocess.Popen(
            argv,
            cwd=str(ROOT),
            stdout=out_f,
            stderr=err_f,
        )

    _release_single_instance_lock()
    time.sleep(1.0)
    sys.exit(0)


def _load_alert_state() -> dict:
    if not _ALERT_STATE_PATH.is_file():
        return {}
    try:
        return json.loads(_ALERT_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_alert_state(state: dict) -> None:
    _ALERT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ALERT_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _bar_minutes_from_cfg(cfg: dict) -> int:
    proj = cfg.get("project", {})
    if "bar_minutes" in proj:
        return max(1, int(proj["bar_minutes"]))
    tf = str(proj.get("timeframe", "M15")).upper()
    return timeframe_to_bar_minutes(tf)


def should_send_no_trade_notif(cfg: dict) -> bool:
    """Cek apakah sudah waktunya kirim notif NO TRADE (interval 0 = setiap bar)."""
    global _LAST_NO_TRADE_NOTIF
    interval = int(
        cfg.get("stage_9", {})
        .get("moment_alert", {})
        .get("no_trade_min_interval_minutes", 0)
    )
    if interval == 0:
        return True
    now = datetime.now(timezone.utc)
    if _LAST_NO_TRADE_NOTIF is None:
        return True
    elapsed = (now - _LAST_NO_TRADE_NOTIF).total_seconds() / 60
    return elapsed >= interval


def mark_no_trade_notif_sent() -> None:
    global _LAST_NO_TRADE_NOTIF
    _LAST_NO_TRADE_NOTIF = datetime.now(timezone.utc)


def format_no_trade_summary(rep: dict) -> str:
    """Ringkasan singkat NO TRADE untuk Telegram (max ~5 baris)."""
    trade_plan = rep.get("trade_plan", {}) if isinstance(rep.get("trade_plan", {}), dict) else {}
    probs = rep.get("probs", {}) if isinstance(rep.get("probs", {}), dict) else {}
    p_up = float(probs.get("up", 0.0))
    p_down = float(probs.get("down", 0.0))
    pred = str(rep.get("pred_class", "?"))
    rec = str(rep.get("recommendation", "NO TRADE"))
    meta = float(trade_plan.get("meta_score") or rep.get("meta_score") or 0.0)
    meta_thr = float(trade_plan.get("meta_threshold") or rep.get("meta_threshold") or 0.0)
    reasons = trade_plan.get("no_trade_reasons") or []
    if isinstance(reasons, list) and reasons:
        reason = ", ".join(str(r) for r in reasons[:2])
    else:
        reason = "filter aktif"
    reason = reason[:120]

    now_wib = datetime.now(timezone.utc).astimezone(WIB).strftime("%H:%M")
    meta_ok = meta >= meta_thr if meta_thr > 0 else False

    return (
        f"📊 *Update {now_wib} WIB — NO TRADE*\n"
        f"• Prediksi: {pred} (UP {p_up:.0%} / DOWN {p_down:.0%})\n"
        f"• Rekomendasi: {rec}\n"
        f"• Meta: {meta:.3f} / {meta_thr:.3f} {'✅' if meta_ok else '❌'}\n"
        f"• Alasan: {reason}"
    )


def run_moment_alert(
    cfg: dict,
    run_dir: Path,
    *,
    skip_mt5: bool,
    chat_id: str,
    token: str,
    min_conf: float,
    rec_allowed: set[str],
) -> dict:
    """Jalankan analisa scheduled + kirim moment alert jika memenuhi syarat."""
    rep = run_daily_report(
        cfg,
        run_dir=run_dir,
        skip_mt5=skip_mt5,
        send_telegram=False,
        target_chat_id=chat_id,
        dedupe_bar=True,
    )
    if rep.get("skipped_duplicate_bar"):
        print(f"[moment_alert] Bar {rep.get('bar_time')} sudah dianalisa — skip duplikat")
        return rep

    trade_plan = rep.get("trade_plan", {}) if isinstance(rep.get("trade_plan", {}), dict) else {}
    rec = str(rep.get("recommendation", "")).upper()
    conf = float(trade_plan.get("confidence", 0.0))
    if (
        rec in rec_allowed
        and trade_plan
        and trade_plan.get("side") in {"BUY", "SELL"}
        and not bool(trade_plan.get("is_no_trade", True))
        and conf >= min_conf
    ):
        timeout_sec = int(cfg.get("stage_9", {}).get("execution", {}).get("confirm_timeout_sec", 120))
        now_ts = time.time()
        register_pending_execution(
            chat_id,
            {
                "symbol": str(rep.get("symbol", cfg.get("project", {}).get("symbol", "XAUUSD"))),
                "timeframe": str(cfg.get("project", {}).get("timeframe", "M15")).upper(),
                "trade_plan": trade_plan,
                "run_id": run_dir.name,
                "timestamp_signal": datetime.now(timezone.utc).isoformat(),
                "meta_allow": True,
                "created_at": now_ts,
                "expires_at": now_ts + max(30, timeout_sec),
            },
        )
        msg = (
            "🚨 *MOMENT ALERT*\n\n"
            + str(rep.get("message", ""))
            + "\n\nBalas `ya` untuk eksekusi order ke MT5, `tidak` untuk batal, "
            "atau /analisa untuk analisa ulang."
        )
        send_telegram_message(token, chat_id, msg, parse_mode=None)
    else:
        moment_cfg = cfg.get("stage_9", {}).get("moment_alert", {})
        send_no_trade = bool(moment_cfg.get("send_no_trade", True))
        is_no_trade = bool(trade_plan.get("is_no_trade", True)) or str(trade_plan.get("action", "")).upper() == "NO_TRADE"
        if send_no_trade and is_no_trade and should_send_no_trade_notif(cfg):
            msg = format_no_trade_summary(rep)
            send_telegram_message(token, chat_id, msg, parse_mode=None)
            mark_no_trade_notif_sent()
            print(f"[moment_alert] NO TRADE notif dikirim | {msg.splitlines()[-1]}", flush=True)
    return rep


def _maybe_auto_retrain(
    cfg: dict,
    run_holder: List[Path],
    *,
    config_path: Path,
    token: str,
    chat_id: str,
    is_retraining: List[bool],
    last_retrain_time: List[Optional[datetime]],
) -> bool:
    """
    Cek drift dan jalankan retrain jika perlu.
    Return True jika retrain dijalankan (skip analisa bar ini).
    """
    retrain_cfg = cfg.get("stage_9", {}).get("auto_retrain", {})
    if not isinstance(retrain_cfg, dict) or not bool(retrain_cfg.get("enabled", True)):
        return False
    if is_retraining[0]:
        return False

    logs_dir = project_root() / "logs"
    risk = load_runtime_risk(logs_dir)
    last_ts = last_retrain_time[0] or load_last_retrain_time(logs_dir)
    needs_retrain, retrain_reason = should_retrain(
        risk,
        last_ts,
        float(retrain_cfg.get("min_interval_hours", 6.0)),
        trigger_on_high_drift=bool(retrain_cfg.get("trigger_on_high_drift", True)),
        trigger_on_critical_drift=bool(retrain_cfg.get("trigger_on_critical_drift", True)),
    )
    if not needs_retrain:
        return False

    allowed, allow_reason = is_retrain_allowed_now()
    if not allowed:
        print(f"[auto_retrain] Retrain ditunda: {allow_reason}", flush=True)
        return False

    print(f"[auto_retrain] Trigger: {retrain_reason}", flush=True)
    is_retraining[0] = True

    notify = None
    if bool(retrain_cfg.get("notify_telegram", True)) and not is_placeholder(token) and chat_id:
        def notify(msg: str) -> None:
            send_telegram_message(token, chat_id, msg, parse_mode=None)

        notify = notify

    output_root = str(cfg.get("experiment", {}).get("output_root", "artifacts"))

    try:
        success, result = run_retrain(
            config_path=config_path,
            project_root=ROOT,
            python_executable=sys.executable,
            output_root=output_root,
            telegram_notify_fn=notify,
            timeout_seconds=int(retrain_cfg.get("timeout_seconds", 3600)),
        )
    finally:
        is_retraining[0] = False

    if success:
        now = datetime.now(timezone.utc)
        last_retrain_time[0] = now
        save_last_retrain_time(logs_dir, now)
        print("[auto_retrain] Retrain sukses — restart service (tanpa hot-reload)", flush=True)
        _spawn_fresh_service_instance()
    else:
        print(f"[auto_retrain] Gagal — tetap pakai model lama: {result[:200]}", flush=True)

    return True


def _run_moment_alert_loop(
    cfg: dict,
    run_holder: List[Path],
    skip_mt5: bool,
    *,
    config_path: Path,
) -> None:
    s9 = cfg.get("stage_9", {})
    a = s9.get("moment_alert", {}) if isinstance(s9.get("moment_alert", {}), dict) else {}
    if not bool(a.get("enabled", False)):
        return

    token = str(cfg.get("risk", {}).get("telegram_token", ""))
    chat_id = str(cfg.get("risk", {}).get("telegram_chat_id", ""))
    if is_placeholder(token) or is_placeholder(chat_id):
        print("[moment_alert] token/chat_id belum siap; loop tidak dijalankan")
        return

    poll_sec = max(15, int(a.get("poll_seconds", 60)))
    min_conf = float(a.get("min_confidence", 0.60))
    rec_allowed = {str(x).upper() for x in (a.get("recommendations") or ["STRONG BUY", "STRONG SELL", "BUY", "SELL"])}
    bar_minutes = _bar_minutes_from_cfg(cfg)
    analysis_delay_seconds = int(s9.get("bar_analysis_delay_seconds", 120))
    last_analyzed_bar: datetime | None = None
    is_retraining: List[bool] = [False]
    last_retrain_time: List[Optional[datetime]] = [load_last_retrain_time(project_root() / "logs")]

    retrain_on = bool(s9.get("auto_retrain", {}).get("enabled", True))
    print(
        f"[moment_alert] aktif | bar={bar_minutes}min | delay={analysis_delay_seconds}s "
        f"| min_conf={min_conf} | rec={sorted(rec_allowed)}"
        f" | auto_retrain={'on' if retrain_on else 'off'}"
        f" | trading_hours=Sen 06:00–Sab 05:00 WIB"
    )

    def _notify_market(msg: str) -> None:
        if not is_placeholder(token) and chat_id:
            send_telegram_message(token, chat_id, msg, parse_mode=None)

    while True:
        try:
            market_open, market_reason = is_market_open()
            if not market_open:
                wait_secs = seconds_until_market_open()
                print(
                    f"[moment_alert] Pasar tutup: {market_reason} — sleep {wait_secs / 3600:.1f} jam",
                    flush=True,
                )
                if not _MARKET_CLOSED_NOTIF_FLAG.exists():
                    _notify_market(
                        f"🔴 *Pasar XAUUSD Tutup*\n"
                        f"Alasan: {market_reason}\n"
                        f"Buka lagi: {wait_secs / 3600:.1f} jam\n"
                        "Bot sleep — analisa lanjut saat pasar buka."
                    )
                    _MARKET_CLOSED_NOTIF_FLAG.parent.mkdir(parents=True, exist_ok=True)
                    _MARKET_CLOSED_NOTIF_FLAG.write_text(
                        datetime.now(timezone.utc).isoformat(),
                        encoding="utf-8",
                    )
                    print(
                        "[moment_alert] Notifikasi pasar tutup dikirim — flag dibuat",
                        flush=True,
                    )
                else:
                    print(
                        "[moment_alert] Pasar tutup — notifikasi sudah dikirim, skip",
                        flush=True,
                    )
                time.sleep(min(wait_secs, 3600.0))
                continue

            if _MARKET_CLOSED_NOTIF_FLAG.exists():
                _MARKET_CLOSED_NOTIF_FLAG.unlink()
                print("[moment_alert] Pasar buka kembali — flag dihapus", flush=True)
                _notify_market("✅ *Pasar XAUUSD Buka*\nAnalisa otomatis dimulai kembali.")

            state = _load_alert_state()
            active = state.get("active")
            if isinstance(active, dict) and active:
                ticket = int(active.get("ticket") or 0)
                if ticket <= 0:
                    state["active"] = {}
                    _save_alert_state(state)
                else:
                    symbol_active = str(
                        active.get("symbol", cfg.get("project", {}).get("symbol", "XAUUSD"))
                    )
                    still_open = False
                    if not skip_mt5:
                        try:
                            ok_mt5, mt5_inst = initialize_mt5(cfg)
                        except RuntimeError:
                            ok_mt5 = False
                            mt5_inst = None
                        if ok_mt5 and mt5_inst is not None:
                            try:
                                still_open = is_position_still_open(
                                    ticket,
                                    symbol_active,
                                    mt5=mt5_inst,
                                )
                            finally:
                                shutdown_mt5(mt5_inst)
                    if still_open:
                        now_mono = time.monotonic()
                        if now_mono - _LAST_POS_OPEN_LOG.get(ticket, 0.0) >= 300:
                            print(
                                f"[moment_alert] Posisi #{ticket} masih terbuka — "
                                f"monitor MT5 aktif ({symbol_active})",
                                flush=True,
                            )
                            _LAST_POS_OPEN_LOG[ticket] = now_mono
                        time.sleep(poll_sec)
                        continue
                    clear_moment_alert_active()
                    print(
                        f"[moment_alert] Posisi #{ticket} sudah close — cooldown cleared",
                        flush=True,
                    )
                    _LAST_POS_OPEN_LOG.pop(ticket, None)

            now = datetime.now(timezone.utc)
            current_bar_open = bar_open_time(now, bar_minutes)

            if is_new_bar(last_analyzed_bar, now, bar_minutes):
                target_analysis_time = current_bar_open + timedelta(seconds=analysis_delay_seconds)
                wait_seconds = (target_analysis_time - now).total_seconds()

                if wait_seconds > 0:
                    print(
                        f"[moment_alert] Bar baru {current_bar_open.strftime('%H:%M UTC')} "
                        f"terdeteksi — tunggu {wait_seconds:.0f} detik sebelum analisa"
                    )
                    time.sleep(wait_seconds)

                active_run = resolve_pipeline_run_dir(
                    run_holder[0],
                    output_root=str(cfg.get("experiment", {}).get("output_root", "artifacts")),
                )
                run_holder[0] = active_run

                if _maybe_auto_retrain(
                    cfg,
                    run_holder,
                    config_path=config_path,
                    token=token,
                    chat_id=chat_id,
                    is_retraining=is_retraining,
                    last_retrain_time=last_retrain_time,
                ):
                    last_analyzed_bar = current_bar_open
                    print(
                        f"[moment_alert] Bar {current_bar_open.strftime('%H:%M UTC')} "
                        "— skip analisa (auto-retrain)"
                    )
                    continue

                print(f"[moment_alert] Menjalankan analisa bar {current_bar_open.strftime('%H:%M UTC')}")
                try:
                    run_moment_alert(
                        cfg,
                        run_holder[0],
                        skip_mt5=skip_mt5,
                        chat_id=chat_id,
                        token=token,
                        min_conf=min_conf,
                        rec_allowed=rec_allowed,
                    )
                    last_analyzed_bar = current_bar_open
                    print(
                        f"[moment_alert] Analisa selesai | bar={current_bar_open.strftime('%H:%M UTC')}"
                    )
                except Exception as exc:
                    print(f"[moment_alert] error: {exc}")
                    last_analyzed_bar = current_bar_open
            else:
                wait = seconds_until_next_bar(now, bar_minutes, delay_seconds=analysis_delay_seconds)
                sleep_duration = max(10.0, wait - 10.0)
                print(
                    f"[moment_alert] Bar {current_bar_open.strftime('%H:%M UTC')} sudah dianalisa "
                    f"— sleep {sleep_duration:.0f} detik"
                )
                time.sleep(sleep_duration)
        except Exception as exc:
            print(f"[moment_alert] error: {exc}")
            time.sleep(poll_sec)


def _setup_service_logging() -> logging.Logger:
    logs_dir = project_root() / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stage9_service")
    if logger.handlers:
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(logs_dir / "stage9_service.log", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)
    return logger


def run_service(args: argparse.Namespace) -> None:
    """Fungsi utama service — dipanggil oleh watchdog main()."""

    cfg = load_pipeline_config(args.config)
    mt5_path = get_mt5_terminal_path(cfg)
    if mt5_path:
        set_mt5_terminal_path(mt5_path)
        print(f"[stage9] MT5 terminal: {mt5_path}", flush=True)
    preferred = None if (args.latest_run or args.run_dir is None) else Path(args.run_dir)
    run_dir = resolve_pipeline_run_dir(
        preferred,
        output_root=str(cfg.get("experiment", {}).get("output_root", "artifacts")),
    )
    run_holder: List[Path] = [run_dir]

    logs_dir = project_root() / "logs"
    last_rt = load_last_retrain_time(logs_dir)
    if last_rt is not None:
        print(f"[auto_retrain] Last retrain: {last_rt.strftime('%Y-%m-%d %H:%M UTC')}", flush=True)

    risk = cfg.get("risk", {})
    token = str(risk.get("telegram_token", ""))
    chat_id = str(risk.get("telegram_chat_id", ""))
    if not is_placeholder(token):
        try:
            telegram_set_commands(
                token,
                [
                    {"command": "analisa", "description": "Analisa XAUUSD on-demand"},
                    {"command": "status", "description": "Status model & drift sistem"},
                    {"command": "trades", "description": "Ringkasan live trades"},
                    {"command": "akun", "description": "Info saldo/equity & P/L MT5"},
                ],
            )
        except Exception as exc:
            # Jangan hentikan service hanya karena setMyCommands gagal sementara (DNS/internet).
            print(f"[WARN] telegram_set_commands gagal: {exc}")

    if not is_placeholder(token) and not is_placeholder(chat_id):
        sym = str(cfg.get("project", {}).get("symbol", "XAUUSD"))
        tf = str(cfg.get("project", {}).get("timeframe", "M15")).upper()
        ma = cfg.get("stage_9", {}).get("moment_alert", {})
        mode_text = "on-demand (tidak ada jadwal otomatis)"
        ar = cfg.get("stage_9", {}).get("auto_retrain", {})
        if isinstance(ma, dict) and bool(ma.get("enabled", False)):
            delay = int(cfg.get("stage_9", {}).get("bar_analysis_delay_seconds", 120))
            mode_text = f"on-demand + moment alert (1x per bar M15 +{delay}s delay)"
            if isinstance(ar, dict) and bool(ar.get("enabled", True)):
                mode_text += " + auto-retrain on HIGH_DRIFT"
        send_telegram_message(
            token,
            chat_id,
            (
                f"*{sym} bot aktif ({tf})*\n\n"
                "Ketik /analisa untuk sinyal terbaru.\n"
                f"Mode: {mode_text}\n\n"
                "/status — status sistem | /trades — performance"
            ),
            parse_mode=None,
        )

    ma = cfg.get("stage_9", {}).get("moment_alert", {})
    if isinstance(ma, dict) and bool(ma.get("enabled", False)):
        th = threading.Thread(
            target=_run_moment_alert_loop,
            args=(cfg, run_holder, args.skip_mt5),
            kwargs={"config_path": args.config},
            daemon=True,
            name="moment_alert",
        )
        th.start()
        print("Moment alert: aktif (monitor posisi MT5 nyata, bukan simulasi harga)")

    print(f"Bot polling: aktif | run_dir={run_holder[0]}")
    print("Perintah Telegram: /analisa /status /trades /akun")
    run_bot_polling(cfg, run_holder[0], run_analysis=run_daily_report, skip_mt5=args.skip_mt5)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 9 service: bot on-demand (/analisa)")
    ap.add_argument("--config", type=Path, default=project_root() / "configs" / "pipeline.yaml")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--latest-run", action="store_true")
    ap.add_argument("--skip-mt5", action="store_true")
    args = ap.parse_args()

    svc_log = _setup_service_logging()

    lock_file = project_root() / "logs" / "stage9_service.lock"
    if lock_file.exists():
        lock_age_seconds = (
            datetime.now(timezone.utc).timestamp() - lock_file.stat().st_mtime
        )
        if lock_age_seconds > 300:
            lock_file.unlink()
            svc_log.info("Lock file stale dihapus (umur: %.0f detik)", lock_age_seconds)
        else:
            svc_log.warning(
                "Lock file baru ada — instance lain mungkin masih jalan (umur: %.0f detik)",
                lock_age_seconds,
            )
            sys.exit(1)

    _acquire_single_instance_lock(ROOT)

    max_restarts = 10
    restart_delay = 30
    restart_count = 0

    while restart_count < max_restarts:
        try:
            svc_log.info("Service start (attempt %d)", restart_count + 1)
            run_service(args)
            svc_log.info("Service exit normal")
            break
        except KeyboardInterrupt:
            svc_log.info("Service dihentikan manual (Ctrl+C)")
            break
        except Exception as exc:
            restart_count += 1
            svc_log.error(
                "Service crash (attempt %d/%d): %s",
                restart_count,
                max_restarts,
                exc,
                exc_info=True,
            )
            print(f"[stage9] Service crash: {exc}", flush=True)
            if restart_count < max_restarts:
                svc_log.info("Restart dalam %d detik...", restart_delay)
                print(f"[stage9] Restart dalam {restart_delay} detik...", flush=True)
                time.sleep(restart_delay)
            else:
                svc_log.critical("Max restart tercapai — service berhenti")
                raise


if __name__ == "__main__":
    main()
