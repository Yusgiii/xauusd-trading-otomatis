"""Eksekusi order market MT5 dari trade plan Stage 9."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from utils.mt5_connection import initialize_mt5, shutdown_mt5
from utils.mt5_export import resolve_symbol

log = logging.getLogger("mt5_execution")


def _setup_mt5_logging() -> None:
    if log.handlers:
        return
    from utils.paths import project_root

    log_dir = project_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "stage9_bot_errors.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    log.addHandler(fh)
    log.setLevel(logging.INFO)


def _normalize_volume(mt5, symbol_info, volume: float) -> float:
    vol_min = float(getattr(symbol_info, "volume_min", 0.01) or 0.01)
    vol_max = float(getattr(symbol_info, "volume_max", 100.0) or 100.0)
    vol_step = float(getattr(symbol_info, "volume_step", 0.01) or 0.01)
    volume = max(vol_min, min(vol_max, volume))
    if vol_step > 0:
        steps = round(volume / vol_step)
        volume = round(steps * vol_step, 8)
    return max(vol_min, volume)


def _resolve_position_ticket(
    mt5: Any,
    resolved: str,
    magic: int,
    order_id: int,
) -> int:
    """Ambil position ticket setelah order market (bukan hanya order ticket)."""
    for _ in range(8):
        positions = mt5.positions_get(symbol=resolved)
        if positions:
            magic_matches = [p for p in positions if int(getattr(p, "magic", 0)) == magic]
            pool = magic_matches if magic_matches else list(positions)
            newest = max(pool, key=lambda p: int(getattr(p, "time", 0)))
            return int(getattr(newest, "ticket", 0) or 0)
        time.sleep(0.25)
    return int(order_id) if order_id else 0


def _pick_filling_mode(mt5, symbol_info) -> int:
    """Pilih mode filling yang didukung broker untuk simbol."""
    mode = int(getattr(symbol_info, "filling_mode", 0))
    if mode & 1:
        return mt5.ORDER_FILLING_FOK
    if mode & 2:
        return mt5.ORDER_FILLING_IOC
    if mode & 4:
        return mt5.ORDER_FILLING_RETURN
    return mt5.ORDER_FILLING_RETURN


def place_market_order_from_plan(
    cfg: Dict[str, Any],
    *,
    symbol: str,
    trade_plan: Dict[str, Any],
    comment: str = "stage9_bot",
) -> Dict[str, Any]:
    _setup_mt5_logging()
    log.info("place_market_order_from_plan dipanggil | symbol=%s plan=%s", symbol, trade_plan)

    side = str(trade_plan.get("side", "NONE")).upper()
    sl = trade_plan.get("sl")
    tp = trade_plan.get("tp")
    if side not in {"BUY", "SELL"}:
        return {"ok": False, "error": f"Side tidak valid untuk order: {side}"}
    if sl is None or tp is None:
        return {"ok": False, "error": "SL/TP kosong pada trade_plan."}

    ex = cfg.get("stage_9", {}).get("execution", {})
    volume = float(trade_plan.get("lot_size") or ex.get("lot", 0.01))
    deviation = int(ex.get("deviation_points", 30))
    magic = int(ex.get("magic", 950001))

    try:
        ok, mt5 = initialize_mt5(cfg)
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    if not ok:
        err = mt5.last_error()
        log.error("MT5 init gagal: %s", err)
        return {"ok": False, "error": f"MT5 init gagal: {err}"}

    try:
        account_info = mt5.account_info()
        if account_info is None:
            log.error("MT5 tidak terkoneksi — account_info None | %s", mt5.last_error())
            return {"ok": False, "error": "MT5 tidak terkoneksi — buka terminal & login akun trading"}

        log.info(
            "MT5 connected | login=%s | balance=%.2f | server=%s",
            account_info.login,
            account_info.balance,
            account_info.server,
        )

        resolved = resolve_symbol(mt5, symbol)
        symbol_info = mt5.symbol_info(resolved)
        if symbol_info is None:
            return {"ok": False, "error": f"symbol_info tidak tersedia: {resolved}"}

        if not symbol_info.visible:
            mt5.symbol_select(resolved, True)

        volume = _normalize_volume(mt5, symbol_info, volume)
        tick = mt5.symbol_info_tick(resolved)
        if tick is None:
            return {"ok": False, "error": f"Tick tidak tersedia untuk {resolved}"}

        order_type = mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL
        price = float(tick.ask if side == "BUY" else tick.bid)
        filling = _pick_filling_mode(mt5, symbol_info)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": resolved,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": float(sl),
            "tp": float(tp),
            "deviation": deviation,
            "magic": magic,
            "comment": str(comment)[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": filling,
        }
        log.info("order_send request: %s", request)

        result = mt5.order_send(request)
        if result is None:
            err = mt5.last_error()
            log.error("order_send None: %s", err)
            return {"ok": False, "error": f"order_send kosong: {err}", "request": request}

        retcode = int(getattr(result, "retcode", -1))
        done_codes = {
            int(getattr(mt5, "TRADE_RETCODE_DONE", 10009)),
            10008,
            10009,
        }
        done = retcode in done_codes
        payload = {
            "ok": done,
            "retcode": retcode,
            "comment": getattr(result, "comment", ""),
            "order": getattr(result, "order", 0),
            "deal": getattr(result, "deal", 0),
            "price": price,
            "symbol": resolved,
            "side": side,
            "volume": volume,
            "request": request,
        }
        if not done:
            payload["error"] = f"Order gagal retcode={retcode} comment={payload['comment']}"
            log.error("Order gagal: %s", payload["error"])

            # Retry dengan filling mode alternatif jika unsupported filling (10030)
            if retcode == 10030:
                for alt_fill in (
                    mt5.ORDER_FILLING_RETURN,
                    mt5.ORDER_FILLING_FOK,
                    mt5.ORDER_FILLING_IOC,
                ):
                    if alt_fill == filling:
                        continue
                    request["type_filling"] = alt_fill
                    log.info("Retry order dengan filling=%s", alt_fill)
                    result2 = mt5.order_send(request)
                    if result2 is None:
                        continue
                    rc2 = int(getattr(result2, "retcode", -1))
                    if rc2 in done_codes:
                        payload.update(
                            {
                                "ok": True,
                                "retcode": rc2,
                                "comment": getattr(result2, "comment", ""),
                                "order": getattr(result2, "order", 0),
                                "deal": getattr(result2, "deal", 0),
                            }
                        )
                        payload.pop("error", None)
                        log.info("Order sukses pada retry filling=%s ticket=%s", alt_fill, payload["order"])
                        break
        else:
            position_ticket = _resolve_position_ticket(
                mt5,
                resolved,
                magic,
                int(payload.get("order", 0) or 0),
            )
            payload["ticket"] = position_ticket
            log.info(
                "Order sukses order=%s position_ticket=%s deal=%s",
                payload["order"],
                position_ticket,
                payload["deal"],
            )

        if payload.get("ok") and not payload.get("ticket"):
            payload["ticket"] = _resolve_position_ticket(
                mt5,
                resolved,
                magic,
                int(payload.get("order", 0) or 0),
            )

        return payload
    except Exception as exc:
        log.error("place_market_order_from_plan exception: %s", exc, exc_info=True)
        return {"ok": False, "error": str(exc)}
    finally:
        shutdown_mt5(mt5)
