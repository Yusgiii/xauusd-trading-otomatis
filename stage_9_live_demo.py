# noqa: D100
"""
Stage 9 — Live on-demand analysis via bot command + Telegram.

Jalankan sekali:
  python stage_9_live_demo.py --run-dir artifacts/run_XXXX
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from utils.config_loader import load_pipeline_config
from utils.live_tracker import live_log_path
from utils.news_fetch import fetch_all_headlines
from utils.prediction_log import append_prediction, resolve_pending_outcomes
from utils.xauusd_features import build_xauusd_features, prepare_inference_bars
from utils.horizon import horizon_label_id
from utils.mt5_export import mt5_timeframe_const, resolve_symbol
from utils.trading_calendar import drop_weekend_bars
from utils.paths import project_root, resolve_pipeline_run_dir, project_root
from utils.sessions import day_of_week, hour_of_day_utc
from utils.telegram_notify import is_placeholder, send_telegram_message, telegram_set_commands
from utils.regime import detect_regime

WIB = ZoneInfo("Asia/Jakarta")
CLASS_NAMES = ("FLAT", "UP", "DOWN")
_LOG = logging.getLogger("stage_9")

_LAST_SIGNAL_BAR_FILE = project_root() / "logs" / "last_signal_bar.txt"


def _bar_time_key(ts: Any) -> str:
    if hasattr(ts, "isoformat"):
        t = ts
        if getattr(t, "tzinfo", None) is None and hasattr(t, "replace"):
            t = t.replace(tzinfo=timezone.utc)
        return t.isoformat()
    return str(ts)


def _get_last_signal_bar() -> str:
    if _LAST_SIGNAL_BAR_FILE.is_file():
        return _LAST_SIGNAL_BAR_FILE.read_text(encoding="utf-8").strip()
    return ""


def _set_last_signal_bar(bar_time_str: str) -> None:
    _LAST_SIGNAL_BAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LAST_SIGNAL_BAR_FILE.write_text(bar_time_str, encoding="utf-8")


def _latest_atr(df_raw: pd.DataFrame, window: int) -> float:
    h = df_raw["high"].astype(float)
    l = df_raw["low"].astype(float)
    c = df_raw["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window, min_periods=max(2, window // 2)).mean()
    val = float(atr.iloc[-1]) if len(atr) else float("nan")
    if not np.isfinite(val) or val <= 0:
        val = float(tr.tail(max(3, window)).mean())
    return max(val, 1e-9)


def _default_sl_multiplier(symbol: str) -> float:
    s = symbol.upper()
    if "XAU" in s:
        return 1.5
    if "BTC" in s:
        return 2.2
    if s in {"EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"}:
        return 1.4
    return 1.5


def _trade_plan_cfg(cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    s9 = cfg.get("stage_9", {})
    base = s9.get("trade_plan", {}) if isinstance(s9.get("trade_plan", {}), dict) else {}
    merged = dict(base)
    per_symbol = base.get("per_symbol", {}) if isinstance(base.get("per_symbol", {}), dict) else {}
    sym_key = str(symbol).upper()
    sym_cfg = per_symbol.get(sym_key, {})
    if isinstance(sym_cfg, dict):
        merged.update(sym_cfg)
    return merged


def _threshold_cfg_for_side(run_dir: Path, side: str) -> Optional[float]:
    p = run_dir / "stage_5" / "threshold_config.json"
    if not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        best = payload.get("best", {}) if isinstance(payload, dict) else {}
        key = "conf_up" if str(side).upper() == "BUY" else "conf_down"
        if key in best:
            return float(best[key])
    except Exception:
        return None
    return None


def load_meta_bundle(run_dir: Path) -> Dict[str, Any]:
    meta_model_path = run_dir / "stage_5" / "meta_model.joblib"
    if not meta_model_path.is_file():
        raise FileNotFoundError(
            f"Meta-model tidak ditemukan: {meta_model_path}\n"
            "Pastikan Stage 5 sudah dijalankan dan meta_model.joblib tersedia."
        )
    return joblib.load(meta_model_path)


def load_runtime_risk() -> Dict[str, Any]:
    """Baca logs/runtime_risk.json untuk drift guard."""
    runtime_risk_path = project_root() / "logs" / "runtime_risk.json"
    default = {
        "drift_level": "NORMAL",
        "drift_status": "NORMAL",
        "risk_multiplier": 1.0,
        "drift_warning": "",
    }
    if not runtime_risk_path.is_file():
        return default
    try:
        risk_data = json.loads(runtime_risk_path.read_text(encoding="utf-8"))
    except Exception:
        return default
    drift_level = str(risk_data.get("drift_level") or risk_data.get("drift_status", "NORMAL")).upper()
    risk_multiplier = float(risk_data.get("risk_multiplier", 1.0))
    drift_warning = ""
    if drift_level == "HIGH_DRIFT":
        drift_warning = "⚠️ HIGH DRIFT — lot dikurangi 30%"
    elif drift_level == "CRITICAL":
        drift_warning = "🛑 CRITICAL DRIFT — eksekusi diblokir"
    return {
        "drift_level": drift_level,
        "drift_status": drift_level,
        "risk_multiplier": risk_multiplier,
        "drift_warning": drift_warning,
        "high_drift_features": risk_data.get("high_drift_features", []),
    }


def apply_meta_filter(
    *,
    run_dir: Path,
    probs: np.ndarray,
    df_raw: pd.DataFrame,
    df_feat: pd.DataFrame,
    trade_plan: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Meta-filter wajib — update trade_plan dan return meta_decision dict."""
    bundle = load_meta_bundle(run_dir)
    meta_model = bundle.get("model")
    meta_features = list(bundle.get("features", []))
    if meta_model is None or not meta_features:
        raise RuntimeError("meta_model.joblib tidak berisi model/features yang valid.")

    meta_threshold = float(bundle.get("threshold", 0.50))
    if cfg:
        s9 = cfg.get("stage_9", {})
        override = s9.get("meta_threshold") if isinstance(s9, dict) else None
        if override is not None:
            meta_threshold = float(override)
            _LOG.info("Meta threshold dari config override: %.3f", meta_threshold)
        else:
            _LOG.info("Meta threshold dari model artifact: %.3f", meta_threshold)
    else:
        _LOG.info("Meta threshold dari model artifact: %.3f", meta_threshold)
    x_meta = pd.DataFrame(
        [
            {
                "p_flat": float(probs[0]),
                "p_up": float(probs[1]),
                "p_down": float(probs[2]),
                "spread": float(df_raw["spread"].iloc[-1]) if "spread" in df_raw.columns else 0.0,
                "atr_zscore": float(df_feat.get("atr_zscore", pd.Series([0.0])).iloc[-1]),
                "session_london": float(df_feat.get("session_london", pd.Series([0.0])).iloc[-1]),
                "session_ny": float(df_feat.get("session_ny", pd.Series([0.0])).iloc[-1]),
                "london_open_proxy": float(df_feat.get("london_open_proxy", pd.Series([0.0])).iloc[-1]),
            }
        ]
    )
    meta_score = float(meta_model.predict_proba(x_meta[meta_features].fillna(0.0))[:, 1][0])
    allow = meta_score >= meta_threshold

    trade_plan["meta_score"] = round(meta_score, 4)
    trade_plan["meta_threshold"] = meta_threshold

    meta_decision = {
        "applied": True,
        "allow_execute": bool(allow),
        "score": meta_score,
        "threshold": meta_threshold,
    }

    if not allow:
        _LOG.info(
            "NO TRADE — meta-filter block (score=%.3f < threshold=%.3f)",
            meta_score,
            meta_threshold,
        )
        trade_plan["is_no_trade"] = True
        trade_plan["action"] = "NO_TRADE"
        trade_plan.setdefault("no_trade_reasons", []).append(
            f"meta_filter {meta_score:.3f} < {meta_threshold:.3f}"
        )
    return meta_decision


def apply_drift_risk(trade_plan: Dict[str, Any], risk_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Kurangi lot saat HIGH_DRIFT; flag block saat CRITICAL."""
    drift_level = str(risk_ctx.get("drift_level", "NORMAL")).upper()
    trade_plan["drift_level"] = drift_level
    trade_plan["drift_warning"] = str(risk_ctx.get("drift_warning", ""))

    if drift_level == "CRITICAL":
        trade_plan["is_no_trade"] = True
        trade_plan["action"] = "NO_TRADE"
        trade_plan.setdefault("no_trade_reasons", []).append("CRITICAL drift — eksekusi diblokir")
        return trade_plan

    if drift_level == "HIGH_DRIFT" and trade_plan.get("lot_size") is not None:
        mult = float(risk_ctx.get("risk_multiplier", 0.7))
        lot = round(float(trade_plan["lot_size"]) * mult, 2)
        lot = max(0.01, lot)
        _LOG.warning(
            "HIGH_DRIFT aktif — lot dikurangi ke %.2f (multiplier=%.2f)",
            lot,
            mult,
        )
        trade_plan["lot_size"] = lot
        trade_plan["drift_lot_multiplier"] = mult
    return trade_plan


def compute_lot_size(
    confidence: float,
    p_up: float,
    p_down: float,
    side: str,
    base_lot: float,
    cfg: Dict[str, Any],
    sentiment: int = 0,
) -> Tuple[float, str]:
    """
    Hitung lot size berdasarkan confidence score gabungan.

    Confidence gabungan = max(confidence_3class, p_directional_binary)
    """
    sizing_cfg = cfg.get("stage_9", {}).get("lot_sizing", {})
    high_thresh = float(sizing_cfg.get("high_confidence_threshold", 0.70))
    medium_thresh = float(sizing_cfg.get("medium_confidence_threshold", 0.55))
    low_thresh = float(sizing_cfg.get("low_confidence_threshold", 0.45))
    high_mult = float(sizing_cfg.get("high_lot_multiplier", 1.5))
    medium_mult = float(sizing_cfg.get("medium_lot_multiplier", 1.0))
    low_mult = float(sizing_cfg.get("low_lot_multiplier", 0.7))

    side_u = str(side).upper()
    if side_u not in {"BUY", "SELL"}:
        return 0.0, "BELOW_THRESHOLD"

    p_directional = float(p_up) if side_u == "BUY" else float(p_down)
    combined_conf = max(float(confidence), p_directional)

    if combined_conf >= high_thresh:
        multiplier = high_mult
        tier = "HIGH"
    elif combined_conf >= medium_thresh:
        multiplier = medium_mult
        tier = "MEDIUM"
    elif combined_conf >= low_thresh:
        multiplier = low_mult
        tier = "LOW"
    else:
        return 0.0, "BELOW_THRESHOLD"

    lot = round(float(base_lot) * multiplier, 2)

    s9 = cfg.get("stage_9", {})
    ls = s9.get("lot_sizing", {})
    if bool(ls.get("sentiment_adjustment_enabled", True)) and lot > 0:
        reduce_f = float(ls.get("sentiment_reduce_factor", 0.8))
        boost_f = float(ls.get("sentiment_boost_factor", 1.1))
        if side_u == "BUY" and sentiment <= -1:
            lot = round(lot * reduce_f, 2)
            tier = f"{tier}_REDUCED_SENTIMENT"
        elif side_u == "SELL" and sentiment >= 1:
            lot = round(lot * reduce_f, 2)
            tier = f"{tier}_REDUCED_SENTIMENT"
        elif (side_u == "BUY" and sentiment >= 2) or (side_u == "SELL" and sentiment <= -2):
            lot = round(lot * boost_f, 2)
            tier = f"{tier}_BOOSTED_SENTIMENT"

    max_lot = round(float(base_lot) * float(ls.get("max_lot_multiplier_cap", 1.5)), 2)
    lot = min(lot, max_lot)
    lot = max(0.01, lot)
    return lot, tier


def validate_trade_plan(trade_plan: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """Validasi trade plan sebelum dikirim. Return (is_valid, reason_if_invalid)."""
    s9 = cfg.get("stage_9", {})
    min_rr = float(s9.get("min_rr_to_send", 1.5))
    max_sl_pips = float(s9.get("max_sl_pips", 50.0))

    actual_rr = float(trade_plan.get("rr_actual", 0.0))
    if actual_rr < min_rr:
        return False, f"RR {actual_rr:.2f} < minimum {min_rr}"

    sl_pips = float(trade_plan.get("sl_distance_pips", 0.0))
    if sl_pips > max_sl_pips:
        return False, f"SL terlalu jauh: {sl_pips:.1f} pip (max {max_sl_pips})"

    return True, ""


def enrich_trade_plan(cfg: Dict[str, Any], trade_plan: Dict[str, Any]) -> Dict[str, Any]:
    """Tambahkan lot sizing, RR aktual, dan validasi ke trade plan."""
    entry = float(trade_plan.get("entry", 0.0))
    sl = trade_plan.get("sl")
    tp = trade_plan.get("tp")
    sl_dist = float(trade_plan.get("sl_distance", 0.0))
    tp_dist = float(trade_plan.get("tp_distance", 0.0))

    if sl is not None and tp is not None and entry and sl_dist > 0:
        trade_plan["rr_actual"] = float(tp_dist / sl_dist)
    else:
        trade_plan["rr_actual"] = float(trade_plan.get("risk_reward", 0.0))

    point_size = float(trade_plan.get("point_size", cfg.get("risk", {}).get("point_size", 0.01)))
    trade_plan["sl_distance_pips"] = float(sl_dist / 0.1) if sl_dist else 0.0
    trade_plan["tp_distance_pips"] = float(tp_dist / 0.1) if tp_dist else 0.0
    if point_size > 0:
        trade_plan["sl_points"] = sl_dist / point_size
        trade_plan["tp_points"] = tp_dist / point_size

    side = str(trade_plan.get("side", "NONE"))
    sizing_cfg = cfg.get("stage_9", {}).get("lot_sizing", {})
    if bool(sizing_cfg.get("enabled", True)) and side in {"BUY", "SELL"}:
        base_lot = float(sizing_cfg.get("base_lot", cfg.get("stage_9", {}).get("execution", {}).get("lot", 0.01)))
        lot, tier = compute_lot_size(
            confidence=float(trade_plan.get("confidence", 0.0)),
            p_up=float(trade_plan.get("p_up", 0.0)),
            p_down=float(trade_plan.get("p_down", 0.0)),
            side=side,
            base_lot=base_lot,
            cfg=cfg,
            sentiment=int(trade_plan.get("sentiment", 0)),
        )
        if lot == 0.0:
            trade_plan["is_no_trade"] = True
            trade_plan["action"] = "NO_TRADE"
            low_thresh = float(sizing_cfg.get("low_confidence_threshold", 0.45))
            trade_plan.setdefault("no_trade_reasons", []).append(
                f"confidence below lot sizing min threshold {low_thresh:.2f}"
            )
        else:
            trade_plan["lot_size"] = lot
            trade_plan["confidence_tier"] = tier
            trade_plan["action"] = side
            trade_plan["_base_lot"] = base_lot

    if not trade_plan.get("is_no_trade"):
        ok, reason = validate_trade_plan(trade_plan, cfg)
        if not ok:
            trade_plan["is_no_trade"] = True
            trade_plan["action"] = "NO_TRADE"
            trade_plan.setdefault("no_trade_reasons", []).append(reason)

    return trade_plan


def build_trade_plan(
    *,
    cfg: Dict[str, Any],
    symbol: str,
    recommendation: str,
    probs: np.ndarray,
    flat_dominant: bool,
    df_raw: pd.DataFrame,
    run_dir: Optional[Path] = None,
    p_up: Optional[float] = None,
    p_down: Optional[float] = None,
    model_source: str = "3class",
    sentiment: int = 0,
) -> Dict[str, Any]:
    tcfg = _trade_plan_cfg(cfg, symbol)
    point_size = float(cfg.get("risk", {}).get("point_size", 0.01))
    atr_w = int(cfg.get("stage_3", {}).get("atr_window", 14))
    atr_val = _latest_atr(df_raw, atr_w)
    entry = float(df_raw["close"].iloc[-1])
    spread = float(df_raw["spread"].iloc[-1]) if "spread" in df_raw.columns else 0.0

    sl_mult = float(tcfg.get("sl_atr_multiplier", _default_sl_multiplier(symbol)))
    rr_base = float(tcfg.get("tp_rr_base", 1.8))
    rr_strong = float(tcfg.get("tp_rr_strong", 2.0))
    rr_floor = float(tcfg.get("min_rr_floor", 1.5))
    rr = rr_strong if str(recommendation).startswith("STRONG") and not flat_dominant else rr_base
    rr = max(rr_floor, rr)

    p_up_bin = float(p_up) if p_up is not None else float(probs[1])
    p_down_bin = float(p_down) if p_down is not None else float(probs[2])
    p_up_side, p_down_side = float(probs[1]), float(probs[2])
    side = "NONE"
    if "KONFLIK" in str(recommendation).upper() or "NO TRADE" in str(recommendation).upper():
        side = "NONE"
    elif "BUY" in str(recommendation):
        side = "BUY"
    elif "SELL" in str(recommendation):
        side = "SELL"
    if flat_dominant:
        side = "NONE"

    sl_dist = sl_mult * atr_val
    tp_dist = rr * sl_dist

    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    if side == "BUY":
        sl_price = entry - sl_dist
        tp_price = entry + tp_dist
    elif side == "SELL":
        sl_price = entry + sl_dist
        tp_price = entry - tp_dist

    confidence = max(p_up_side, p_down_side, p_up_bin if side == "BUY" else 0.0, p_down_bin if side == "SELL" else 0.0)
    no_trade_reasons: List[str] = []
    if "KONFLIK" in str(recommendation).upper():
        no_trade_reasons.append("konflik ML vs sentiment")
    if side == "NONE":
        no_trade_reasons.append("bias tidak cukup kuat / FLAT dominan")
    min_conf = float(tcfg.get("min_confidence", 0.55))
    conf_threshold = min_conf
    if run_dir is not None and side in {"BUY", "SELL"}:
        th = _threshold_cfg_for_side(run_dir, side)
        if th is not None:
            conf_threshold = float(th)
    if confidence < conf_threshold:
        no_trade_reasons.append(f"confidence {confidence:.2f} < min {conf_threshold:.2f}")

    return {
        "side": side,
        "entry": entry,
        "sl": sl_price,
        "tp": tp_price,
        "risk_reward": rr,
        "atr_window": atr_w,
        "atr_value": atr_val,
        "sl_atr_multiplier": sl_mult,
        "sl_distance": sl_dist,
        "tp_distance": tp_dist,
        "sl_points": sl_dist / point_size if point_size > 0 else None,
        "tp_points": tp_dist / point_size if point_size > 0 else None,
        "point_size": point_size,
        "spread_last": spread,
        "confidence": confidence,
        "confidence_threshold_used": conf_threshold,
        "is_no_trade": len(no_trade_reasons) > 0,
        "no_trade_reasons": no_trade_reasons,
        "model_source": model_source,
        "p_up": p_up_bin,
        "p_down": p_down_bin,
        "sentiment": int(sentiment),
        "rr_actual": float(tp_dist / sl_dist) if sl_dist > 0 else rr,
        "sl_distance_pips": float(sl_dist / 0.1) if sl_dist else 0.0,
        "tp_distance_pips": float(tp_dist / 0.1) if tp_dist else 0.0,
    }
def _consensus_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    c = cfg.get("stage_9", {}).get("consensus", {})
    g = cfg.get("stage_9", {}).get("gemini", {})
    return {
        "weak_lean_min": float(c.get("weak_lean_min", 0.05)),
        "moderate_combined": float(c.get("moderate_combined", 0.12)),
        "strong_combined": float(c.get("strong_combined", 0.70)),
        "strong_class_prob": float(c.get("strong_class_prob", 0.38)),
        "sentiment_weight": float(c.get("sentiment_weight", 0.35)),
        "sentiment_scale": float(g.get("sentiment_scale", c.get("sentiment_scale", 2.0))),
    }


def _sentiment_label(score: int) -> str:
    if score <= -2:
        return "Sangat Bearish"
    if score == -1:
        return "Bearish"
    if score == 0:
        return "Netral"
    if score == 1:
        return "Bullish"
    return "Sangat Bullish"


def _format_sentiment_note_line(sentiment_note: str) -> str:
    """Format baris catatan sentimen untuk Telegram (cache / fallback / normal)."""
    note = str(sentiment_note)
    if "[cache" in note.lower():
        return f"{note} ⏱️"
    if "tidak tersedia" in note.lower() or "fallback keyword" in note.lower():
        return f"{note} ⚠️"
    return note


def _sentiment_emoji(score: int) -> str:
    if score <= -2:
        return "🔴"
    if score == -1:
        return "🟠"
    if score == 0:
        return "⚪"
    if score == 1:
        return "🟡"
    return "🟢"


def _maybe_mt5():
    try:
        import MetaTrader5 as mt5

        return mt5
    except ImportError:
        return None


def fetch_mt5_bars(
    symbol: str,
    n_bars: int,
    mt5,
    timeframe: str = "H1",
    *,
    cfg: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    from utils.mt5_connection import get_mt5_terminal_path

    path = get_mt5_terminal_path(cfg)
    if path:
        ok = mt5.initialize(path=path)
    else:
        ok = mt5.initialize()
    if not ok:
        raise RuntimeError(f"MT5 init gagal: {mt5.last_error()}")
    try:
        resolved = resolve_symbol(mt5, symbol)
        tf_const = mt5_timeframe_const(mt5, timeframe)
        rates = mt5.copy_rates_from_pos(resolved, tf_const, 0, int(n_bars))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"Tidak ada data MT5 untuk {resolved}: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        if "spread" not in df.columns:
            df["spread"] = 0.0
        out = df[["time", "open", "high", "low", "close", "spread"]].sort_values("time")
        out, _ = drop_weekend_bars(out)
        return out
    finally:
        mt5.shutdown()


def fetch_csv_tail(cfg: Dict[str, Any], n_bars: int) -> pd.DataFrame:
    """Cadangan jika MT5 tidak tersedia: bar terakhir dari CSV Stage 1."""
    csv_path = Path(cfg["stage_1"]["input_csv"])
    if not csv_path.is_absolute():
        csv_path = project_root() / csv_path
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time")
    if bool(cfg.get("stage_1", {}).get("exclude_weekend_bars", True)):
        df, _ = drop_weekend_bars(df)
    df = df.tail(int(n_bars))
    for col in ("open", "high", "low", "close", "spread"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


def load_binary_bundle(run_dir: Path) -> Dict[str, Any] | None:
    """Load xgb_binary_up/down.joblib jika ada."""
    up_path = run_dir / "stage_5" / "xgb_binary_up.joblib"
    down_path = run_dir / "stage_5" / "xgb_binary_down.joblib"
    if not up_path.is_file() or not down_path.is_file():
        return None
    return {
        "up": joblib.load(up_path),
        "down": joblib.load(down_path),
    }


def infer_binary_probs(df_feat: pd.DataFrame, run_dir: Path) -> Optional[tuple[float, float, float, float]]:
    """Return P_up, P_down, thr_up, thr_down dari binary models; None jika tidak ada."""
    bundles = load_binary_bundle(run_dir)
    if bundles is None:
        return None
    up_b = bundles["up"]
    down_b = bundles["down"]
    feats_up = up_b["features"]
    feats_down = down_b["features"]
    row_up = df_feat.iloc[[-1]][feats_up].astype(np.float32)
    row_down = df_feat.iloc[[-1]][feats_down].astype(np.float32)
    p_up = float(up_b["model"].predict_proba(row_up)[0, 1])
    p_down = float(down_b["model"].predict_proba(row_down)[0, 1])
    thr_up = float(up_b.get("threshold", 0.50))
    thr_down = float(down_b.get("threshold", 0.50))
    return p_up, p_down, thr_up, thr_down


def load_xgb_bundle(run_dir: Path, *, cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    output_root = "artifacts"
    if cfg:
        output_root = str(cfg.get("experiment", {}).get("output_root", "artifacts"))
    run_dir = resolve_pipeline_run_dir(run_dir, output_root=output_root)
    path = run_dir / "stage_5" / "xgb_model.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {path}. "
            "Jalankan: python run_pipeline.py lalu restart bot."
        )
    return joblib.load(path)


def infer_direction(df_feat: pd.DataFrame, bundle: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    feats = bundle["features"]
    row = df_feat.iloc[[-1]][feats].astype(np.float32)
    model = bundle["model"]
    regime_models = {}
    if bool(bundle.get("regime_enabled", False)):
        rp = Path(bundle.get("regime_models_path", ""))
        if rp and rp.is_file():
            try:
                regime_models = joblib.load(rp)
            except Exception:
                regime_models = {}
    if regime_models:
        reg = str(detect_regime(df_feat).iloc[-1])
        rb = regime_models.get(reg)
        if isinstance(rb, dict) and "model" in rb and "features" in rb:
            row = df_feat.iloc[[-1]][rb["features"]].astype(np.float32)
            model = rb["model"]
    probs = model.predict_proba(row)[0]
    pred = int(np.argmax(probs))
    return probs, pred


def resolve_sentiment(
    headlines: List[str],
    gemini_key: str,
    model_name: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[int, str, float]:
    """Gemini jika ada key; fallback keyword jika gagal / tanpa key. Return (score, note, confidence)."""
    from utils.news_fetch import simple_headline_sentiment

    if gemini_key and not gemini_key.startswith("YOUR_"):
        s9 = (cfg or {}).get("stage_9", {})
        gcfg = s9.get("gemini", {}) if isinstance(s9.get("gemini", {}), dict) else {}
        fb = list(s9.get("gemini_fallback_models") or [])
        symbol = str((cfg or {}).get("project", {}).get("symbol", "XAUUSD"))
        horizon_text = horizon_label_id(cfg or {})
        score, note, conf = gemini_sentiment(
            headlines,
            gemini_key,
            model_name,
            symbol=symbol,
            horizon_text=horizon_text,
            fallback_models=fb,
            cfg=cfg,
        )
        min_conf = float(gcfg.get("min_confidence_to_use", 0.3))
        if conf < min_conf:
            return 0, f"{note[:200]} (confidence {conf:.2f} < {min_conf}, netral)"[:240], conf
        return score, note[:240], conf
    kw_score, kw_note = simple_headline_sentiment(headlines)
    return kw_score, kw_note, 0.5


def gemini_sentiment(
    headlines: List[str],
    api_key: str,
    model_name: str,
    symbol: str = "XAUUSD",
    horizon_text: str = "1 jam ke depan (1 bar H1)",
    fallback_models: Optional[List[str]] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[int, str, float]:
    """Skor -2..+2 via utils.gemini_client (retry + cache fallback)."""
    from utils.gemini_client import gemini_news_sentiment_with_retry

    return gemini_news_sentiment_with_retry(
        headlines,
        api_key,
        symbol=symbol,
        horizon_text=horizon_text,
        model=model_name,
        fallback_models=fallback_models,
        cfg=cfg,
    )


def explain_signal(
    probs: np.ndarray,
    pred: int,
    sentiment: int,
    consensus: Dict[str, Any],
    *,
    flat_threshold: float = 0.0005,
    horizon_text: str = "24 jam ke depan",
) -> str:
    """Penjelasan rekomendasi untuk laporan Telegram."""
    p_flat, p_up, p_down = float(probs[0]), float(probs[1]), float(probs[2])
    lean_delta = float(consensus["lean_delta"])
    rec = str(consensus["recommendation"])
    lines: List[str] = []

    lines.insert(
        0,
        f"Horizon prediksi: *{horizon_text}* (dari close bar acuan).",
    )
    if pred == 0:
        lines.append(
            f"Kelas terbesar *FLAT* ({p_flat:.0%}): model mengharapkan range sempit "
            f"selama horizon (|return| ≤ {flat_threshold:.2%})."
        )
    elif pred == 1:
        lines.append(f"Kelas terbesar *UP* ({p_up:.0%}).")
    else:
        lines.append(f"Kelas terbesar *DOWN* ({p_down:.0%}).")

    lines.append(f"Bias UP−DOWN: *{lean_delta:+.1%}* | skor gabungan: *{consensus['combined']:+.2f}*")

    if rec.startswith("WEAK"):
        lines.append(
            "_Sinyal lemah: FLAT masih dominan, tetapi probabilitas UP/DOWN cond. "
            "condong ke satu arah — hati-hati, bukan entry agresif._"
        )
    elif "ABSTAIN" in rec:
        lines.append("_Tidak ada bias arah cukup kuat — tunggu setup berikutnya._")
    elif rec.startswith("STRONG"):
        lines.append("_Sinyal kuat: kelas arah + berita selaras._")

    if sentiment == 0:
        lines.append(f"Berita: netral ({_sentiment_label(sentiment)}).")
    elif sentiment > 0:
        lines.append(f"Berita: mendukung sisi bullish ({_sentiment_label(sentiment)}).")
    else:
        lines.append(f"Berita: mendukung sisi bearish ({_sentiment_label(sentiment)}).")
    if consensus.get("conflict"):
        lines.append(f"Konflik ML vs berita: {consensus.get('conflict_reason', '')}")
    return "\n".join(lines)


def consensus_matrix(
    probs: np.ndarray,
    pred: int,
    sentiment: int,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Gabungkan ML + sentimen → STRONG … WEAK … NEUTRAL.
    Saat FLAT dominan, tetap keluarkan WEAK BUY/SELL jika |P(UP)-P(DOWN)| cukup besar.
    """
    cc = _consensus_cfg(cfg)
    p_flat, p_up, p_down = float(probs[0]), float(probs[1]), float(probs[2])
    lean_delta = p_up - p_down

    sentiment_scale = max(float(cc.get("sentiment_scale", 2.0)), 1.0)
    sentiment_normalized = float(sentiment) / sentiment_scale
    combined = lean_delta + cc["sentiment_weight"] * sentiment_normalized

    wlean = cc["weak_lean_min"]
    mod = cc["moderate_combined"]
    strong = cc["strong_combined"]
    p_strong = cc["strong_class_prob"]

    rec = "NEUTRAL / ABSTAIN"
    flat_dominant = pred == 0 and p_flat >= 0.5
    conflict = False
    conflict_reason = ""

    ml_direction = 1 if lean_delta > 0.05 else (-1 if lean_delta < -0.05 else 0)
    if abs(lean_delta) < 0.25 and pred in (1, 2):
        ml_direction = 1 if pred == 1 else -1
    sentiment_direction = 1 if sentiment > 0 else (-1 if sentiment < 0 else 0)

    strong_conflict = (
        ml_direction != 0
        and sentiment_direction != 0
        and ml_direction != sentiment_direction
        and abs(sentiment) >= 2
        and abs(lean_delta) < 0.25
    )

    if strong_conflict:
        rec = "NO TRADE / KONFLIK"
        conflict = True
        conflict_reason = (
            f"ML={'UP' if ml_direction > 0 else 'DOWN'} vs Sentiment="
            f"{'Bearish' if sentiment < 0 else 'Bullish'} (score={sentiment})"
        )
        return {
            "recommendation": rec,
            "lean_delta": lean_delta,
            "combined": combined,
            "sentiment_normalized": sentiment_normalized,
            "pred_class": CLASS_NAMES[pred],
            "flat_dominant": flat_dominant,
            "conflict": conflict,
            "conflict_reason": conflict_reason,
        }

    if pred == 1 and sentiment >= 1 and p_up >= p_strong:
        rec = "STRONG BUY"
    elif pred == 2 and sentiment <= -1 and p_down >= p_strong:
        rec = "STRONG SELL"
    elif not flat_dominant and combined >= strong:
        rec = "BUY"
    elif not flat_dominant and combined <= -strong:
        rec = "SELL"
    elif not flat_dominant and combined >= mod:
        rec = "BUY"
    elif not flat_dominant and combined <= -mod:
        rec = "SELL"
    elif not flat_dominant and pred == 1 and p_up >= mod:
        rec = "BUY"
    elif not flat_dominant and pred == 2 and p_down >= mod:
        rec = "SELL"
    elif lean_delta >= wlean:
        rec = "WEAK BUY"
    elif lean_delta <= -wlean:
        rec = "WEAK SELL"

    return {
        "recommendation": rec,
        "lean_delta": lean_delta,
        "combined": combined,
        "sentiment_normalized": sentiment_normalized,
        "pred_class": CLASS_NAMES[pred],
        "flat_dominant": flat_dominant,
        "conflict": conflict,
        "conflict_reason": conflict_reason,
    }


def format_telegram_markdown(
    *,
    symbol: str,
    timeframe: str,
    probs: np.ndarray,
    pred: int,
    sentiment: int,
    sentiment_note: str,
    recommendation: str,
    bar_time: str,
    headlines: List[str],
    signal_note: str = "",
    flat_dominant: bool = False,
    horizon_text: str = "1 jam ke depan (1 bar H1)",
    trade_plan: Optional[Dict[str, Any]] = None,
    consensus: Optional[Dict[str, Any]] = None,
) -> str:
    p_up, p_down = float(probs[1]), float(probs[2])
    lean = "UP" if p_up > p_down + 0.02 else ("DOWN" if p_down > p_up + 0.02 else "seimbang")
    lines = [
        f"*{symbol} Sinyal {timeframe} — {bar_time}*",
        "",
        f"_Target: pergerakan *{horizon_text}*_",
        "",
        "*Model XGBoost (bar acuan terakhir)*",
        f"• Prediksi kelas: *{CLASS_NAMES[pred]}*",
        f"• P(FLAT)={probs[0]:.1%} | P(UP)={probs[1]:.1%} | P(DOWN)={probs[2]:.1%}",
        f"• Bias arah (UP−DOWN): *{p_up - p_down:+.1%}* → cond. *{lean}*",
        "",
        "*Sentimen Berita (Gemini AI)*",
        f"• Skor: *{sentiment:+d}/2* {_sentiment_emoji(sentiment)} ({_sentiment_label(sentiment)})",
        f"• {_format_sentiment_note_line(sentiment_note)}",
        "",
        f"*Rekomendasi:* *{recommendation}*",
        f"• Mode: *{'Conservative (FLAT-dominan)' if flat_dominant else 'Directional'}*",
    ]
    if consensus and consensus.get("conflict"):
        lines.append(f"⚠️ *KONFLIK*: {consensus.get('conflict_reason', '')} → NO TRADE")
    lines.extend(
        [
            "",
            "*Rencana Trading (ATR-based)*",
        ]
    )
    if trade_plan:
        if bool(trade_plan.get("is_no_trade")):
            lines.extend(
                [
                    "• *NO TRADE*",
                    f"• Entry referensi: `{trade_plan.get('entry', 0.0):.5f}`",
                    f"• Alasan: {', '.join(trade_plan.get('no_trade_reasons', []) or ['filter risk'])}",
                    f"• ATR({trade_plan.get('atr_window')}): `{trade_plan.get('atr_value', 0.0):.5f}`",
                    "",
                ]
            )
        else:
            tier = str(trade_plan.get("confidence_tier", ""))
            lot = trade_plan.get("lot_size")
            conf = float(trade_plan.get("confidence", 0.0))
            meta_sc = trade_plan.get("meta_score")
            meta_th = trade_plan.get("meta_threshold")
            meta_ok = (
                f" | Meta: {float(meta_sc):.2f} ✅"
                if meta_sc is not None and float(meta_sc) >= float(meta_th or 0)
                else (
                    f" | Meta: {float(meta_sc):.2f} ❌"
                    if meta_sc is not None
                    else ""
                )
            )
            drift_warn = str(trade_plan.get("drift_warning", "") or "")
            if drift_warn:
                lines.append(drift_warn)
            lines.append(f"• Side: *{trade_plan.get('side', 'NONE')}*")
            if tier:
                lines.append(f"• Confidence: *{conf:.2f}* [{tier}]{meta_ok}")
            else:
                lines.append(f"• Confidence: *{conf:.2f}*{meta_ok}")
            if lot is not None:
                base_lot = float(trade_plan.get("_base_lot", 0.01))
                mult = float(lot) / base_lot if base_lot > 0 else 1.0
                lines.append(f"• Lot Size: *{float(lot):.3f}* ({mult:.1f}x base)")
            lines.extend(
                [
                    f"• Entry: `{trade_plan.get('entry', 0.0):.5f}`",
                    f"• SL: `{trade_plan.get('sl', 0.0):.5f}` (-{trade_plan.get('sl_distance', 0.0):.2f} / -{trade_plan.get('sl_atr_multiplier', 1.5):.1f} ATR)",
                    f"• TP: `{trade_plan.get('tp', 0.0):.5f}` (+{trade_plan.get('tp_distance', 0.0):.2f} / RR {float(trade_plan.get('rr_actual', trade_plan.get('risk_reward', 1.8))):.2f})",
                    f"• ATR({trade_plan.get('atr_window')}): `{trade_plan.get('atr_value', 0.0):.5f}`",
                    "",
                ]
            )
    lines.extend(
        [
        "*Catatan:*",
        signal_note,
        "",
        "_Headlines:_",
        ]
    )
    for h in headlines[:5]:
        lines.append(f"• {h[:120]}")
    if len(headlines) > 5:
        lines.append(f"• … +{len(headlines) - 5} lainnya")
    return "\n".join(lines)


def format_system_status(
    cfg: Dict[str, Any],
    run_dir: Path,
    *,
    skip_mt5: bool = False,
    last_signal: Optional[Dict[str, Any]] = None,
) -> str:
    from utils.retrain_scheduler import load_last_retrain_time
    from utils.trading_hours import get_session, is_market_open

    risk_ctx = load_runtime_risk()
    drift = str(risk_ctx.get("drift_level", risk_ctx.get("drift_status", "NORMAL")))
    drift_icon = "✅" if drift == "NORMAL" else ("⚠️" if "HIGH" in drift.upper() else "🛑")

    market_open, _market_reason = is_market_open()
    session = get_session()
    market_line = f"✅ {session}" if market_open else "🔴 Tutup"

    min_retrain_h = float(cfg.get("stage_9", {}).get("auto_retrain", {}).get("min_interval_hours", 6.0))
    logs_dir = project_root() / "logs"
    last_rt = load_last_retrain_time(logs_dir)
    if last_rt is not None:
        last_retrain_str = last_rt.astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")
        next_rt = last_rt + timedelta(hours=min_retrain_h)
        if datetime.now(timezone.utc) >= next_rt:
            next_retrain_str = "boleh sekarang (jika drift tinggi)"
        else:
            next_retrain_str = next_rt.astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")
    else:
        last_retrain_str = "belum pernah"
        next_retrain_str = "— (jika drift HIGH/CRITICAL)"

    lines = [
        "*System Status*",
        f"• Run: `{run_dir.name}`",
        f"• Drift: {drift} {drift_icon}",
        f"• Pasar: {market_line}",
        f"• MT5: {'CSV fallback' if skip_mt5 else 'live'}",
        f"• Last retrain: {last_retrain_str}",
        f"• Next retrain allowed: {next_retrain_str}",
        f"• Waktu: {datetime.now(WIB).strftime('%Y-%m-%d %H:%M WIB')}",
    ]
    if last_signal:
        lines.append(
            f"• Last signal: {last_signal.get('side', '—')} "
            f"(meta={last_signal.get('meta_score', '—')}) @ {last_signal.get('time', '—')}"
        )
    log_p = live_log_path(cfg)
    if log_p.is_file():
        try:
            import pandas as pd

            df = pd.read_csv(log_p, engine="python", on_bad_lines="skip")
            today = datetime.now(timezone.utc).date()
            df["timestamp_signal"] = pd.to_datetime(df["timestamp_signal"], utc=True, errors="coerce")
            d0 = df[df["timestamp_signal"].dt.date == today]
            open_n = int((d0["outcome"].astype(str) == "OPEN").sum()) if len(d0) else 0
            closed = d0[d0["outcome"].astype(str).isin(["TP", "SL", "MANUAL_CLOSE"])] if len(d0) else d0
            tp_n = int((closed["outcome"].astype(str) == "TP").sum()) if len(closed) else 0
            lines.append(f"• Trades today: {len(d0)} ({open_n} open, {tp_n} closed TP)")
        except Exception:
            pass
    lines.extend(
        [
            "",
            "💡 *Cara pakai:*",
            "• /analisa — minta sinyal terbaru",
            "• Balas *ya* untuk eksekusi ke MT5",
            "• /trades — cek performance",
            "• Tutup posisi manual setelah 3 jam jika belum hit TP/SL",
        ]
    )
    return "\n".join(lines)


def run_daily_report(
    cfg: Dict[str, Any],
    *,
    run_dir: Path,
    dry_run: bool = False,
    skip_mt5: bool = False,
    send_telegram: bool = True,
    target_chat_id: Optional[str] = None,
    symbol_override: Optional[str] = None,
    timeframe_override: Optional[str] = None,
    verbose: bool = False,
    dedupe_bar: bool = False,
) -> Dict[str, Any]:
    proj = cfg["project"]
    risk = cfg["risk"]
    s9 = cfg.get("stage_9", {})
    symbol = str(symbol_override or proj["symbol"]).upper()
    tf = str(timeframe_override or proj.get("timeframe", "H1")).upper()
    n_bars = int(s9.get("mt5_bars", 120))

    output_root = str(cfg.get("experiment", {}).get("output_root", "artifacts"))
    run_dir = resolve_pipeline_run_dir(run_dir, output_root=output_root)

    out: Dict[str, Any] = {
        "dry_run": dry_run,
        "symbol": symbol,
        "run_dir": run_dir.name,
    }

    if dry_run:
        out["note"] = "dry_run aktif — tidak ada inferensi / Telegram."
        out["recommendation"] = "NEUTRAL / ABSTAIN"
        return out

    risk_ctx = load_runtime_risk()
    out["runtime_risk"] = risk_ctx
    if risk_ctx["drift_level"] == "CRITICAL":
        msg = (
            f"*{symbol} — TRADING DIBLOKIR*\n\n"
            f"{risk_ctx.get('drift_warning', 'CRITICAL drift')}\n"
            "Tidak ada trade plan yang dikirim. Periksa `logs/runtime_risk.json` dan retrain jika perlu."
        )
        if send_telegram:
            send_telegram_message(
                str(risk.get("telegram_token", "")),
                str(target_chat_id or risk.get("telegram_chat_id", "")),
                msg,
            )
        out.update(
            {
                "action": "NO_TRADE",
                "reason": "CRITICAL drift",
                "message": msg,
                "trade_plan": {"is_no_trade": True, "action": "NO_TRADE", "drift_level": "CRITICAL"},
            }
        )
        return out

    df_raw: pd.DataFrame
    if skip_mt5:
        df_raw = fetch_csv_tail(cfg, n_bars)
        out["data_source"] = "csv_tail"
    else:
        mt5 = _maybe_mt5()
        if mt5 is None:
            df_raw = fetch_csv_tail(cfg, n_bars)
            out["data_source"] = "csv_tail_fallback"
        else:
            df_raw = fetch_mt5_bars(symbol, n_bars, mt5, timeframe=tf, cfg=cfg)
            out["data_source"] = "mt5"

    latest_bar_ts = df_raw["time"].iloc[-1]
    current_bar_str = _bar_time_key(latest_bar_ts)
    if dedupe_bar:
        last_bar_str = _get_last_signal_bar()
        if current_bar_str == last_bar_str:
            _LOG.info("Bar %s sudah dianalisa sebelumnya — skip duplikat", current_bar_str)
            return {
                **out,
                "skipped_duplicate_bar": True,
                "bar_time": current_bar_str,
                "recommendation": "SKIPPED_DUPLICATE",
            }

    df_prep = prepare_inference_bars(df_raw, cfg)
    df_feat, _ = build_xauusd_features(df_prep, cfg)

    bundle = load_xgb_bundle(run_dir, cfg=cfg)
    model_source = "3class"
    probs, pred = infer_direction(df_feat, bundle)
    p_up_val: Optional[float] = float(probs[1])
    p_down_val: Optional[float] = float(probs[2])

    bin_probs = infer_binary_probs(df_feat, run_dir)
    if bin_probs is not None:
        p_up_val, p_down_val = float(bin_probs[0]), float(bin_probs[1])

    headlines, news_meta = fetch_all_headlines(cfg)
    from utils.news_fetch import filter_headlines_for_xauusd

    relevant_headlines, irrelevant_headlines = filter_headlines_for_xauusd(headlines)
    out["news"] = {**news_meta, "filtered_relevant": len(relevant_headlines), "filtered_irrelevant": len(irrelevant_headlines)}
    sentiment, sentiment_note, sentiment_confidence = resolve_sentiment(
        relevant_headlines,
        str(risk.get("gemini_api_key", "")),
        str(s9.get("gemini_model", "gemini-2.5-flash")),
        cfg=cfg,
    )
    if irrelevant_headlines:
        sentiment_note += f" ({len(irrelevant_headlines)} berita non-gold diabaikan)"
    out["sentiment_confidence"] = sentiment_confidence
    flat_thr = float(risk.get("flat_return_threshold", 0.0015))
    htext = horizon_label_id(cfg)
    consensus = consensus_matrix(probs, pred, sentiment, cfg)
    recommendation = str(consensus["recommendation"])
    trade_plan = build_trade_plan(
        cfg=cfg,
        symbol=symbol,
        recommendation=recommendation,
        probs=probs,
        flat_dominant=bool(consensus.get("flat_dominant", False)),
        df_raw=df_raw,
        run_dir=run_dir,
        p_up=p_up_val,
        p_down=p_down_val,
        model_source=model_source,
        sentiment=sentiment,
    )
    if consensus.get("conflict"):
        trade_plan["is_no_trade"] = True
        trade_plan["action"] = "NO_TRADE"
        trade_plan.setdefault("no_trade_reasons", []).append(consensus.get("conflict_reason", "konflik"))
    meta_decision = apply_meta_filter(
        run_dir=run_dir,
        probs=probs,
        df_raw=df_raw,
        df_feat=df_feat,
        trade_plan=trade_plan,
        cfg=cfg,
    )
    if not trade_plan.get("is_no_trade"):
        trade_plan = enrich_trade_plan(cfg, trade_plan)
        trade_plan = apply_drift_risk(trade_plan, risk_ctx)
    signal_note = explain_signal(
        probs, pred, sentiment, consensus, flat_threshold=flat_thr, horizon_text=htext
    )
    out["consensus"] = consensus
    out["meta_decision"] = meta_decision

    bar_time = df_raw["time"].iloc[-1].tz_convert(WIB).strftime("%Y-%m-%d %H:%M WIB")
    message = format_telegram_markdown(
        symbol=symbol,
        timeframe=tf,
        probs=probs,
        pred=pred,
        sentiment=sentiment,
        sentiment_note=sentiment_note,
        recommendation=recommendation,
        bar_time=bar_time,
        headlines=headlines,
        signal_note=signal_note,
        flat_dominant=bool(consensus.get("flat_dominant", False)),
        horizon_text=htext,
        trade_plan=trade_plan,
        consensus=consensus,
    )

    chat_id = str(target_chat_id or risk.get("telegram_chat_id", ""))
    tg: Dict[str, Any] = {"ok": False, "skipped": True}
    if send_telegram:
        tg = send_telegram_message(
            str(risk.get("telegram_token", "")),
            chat_id,
            message,
        )
        if not tg.get("ok"):
            out["telegram_error"] = tg

    report = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "bar_time_wib": bar_time,
        "probs": {"flat": float(probs[0]), "up": float(probs[1]), "down": float(probs[2])},
        "pred_class": CLASS_NAMES[pred],
        "sentiment": sentiment,
        "sentiment_note": sentiment_note,
        "recommendation": recommendation,
        "trade_plan": trade_plan,
        "meta_decision": meta_decision,
        "message": message,
        "telegram": tg,
    }
    out.update(
        {
            "pred_class": CLASS_NAMES[pred],
            "sentiment": sentiment,
            "sentiment_note": sentiment_note,
            "recommendation": recommendation,
            "model_source": model_source,
            "p_up": float(p_up_val) if p_up_val is not None else None,
            "p_down": float(p_down_val) if p_down_val is not None else None,
            "meta_score": trade_plan.get("meta_score"),
            "meta_threshold": trade_plan.get("meta_threshold"),
            "drift_level": risk_ctx.get("drift_level"),
        }
    )
    if verbose:
        out["verbose"] = {
            "model_source": model_source,
            "confidence_tier": trade_plan.get("confidence_tier"),
            "lot_size": trade_plan.get("lot_size"),
            "meta_score": trade_plan.get("meta_score"),
            "meta_threshold": trade_plan.get("meta_threshold"),
            "p_up": trade_plan.get("p_up"),
            "p_down": trade_plan.get("p_down"),
            "rr_actual": trade_plan.get("rr_actual"),
            "sentiment": sentiment,
            "sentiment_confidence": sentiment_confidence,
            "conflict": consensus.get("conflict"),
        }

    rd = ensure_report_dir(run_dir)
    (rd / "daily_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (rd / "daily_report.md").write_text(message, encoding="utf-8")

    bar_utc = df_raw["time"].iloc[-1]
    if hasattr(bar_utc, "to_pydatetime"):
        bar_utc = bar_utc.to_pydatetime()
    if bar_utc.tzinfo is None:
        bar_utc = bar_utc.replace(tzinfo=timezone.utc)

    log_id = append_prediction(
        cfg,
        bar_time_utc=bar_utc,
        symbol=symbol,
        close_at_signal=float(df_raw["close"].iloc[-1]),
        pred_class_idx=pred,
        probs=probs,
        sentiment=sentiment,
        recommendation=recommendation,
        consensus=consensus,
        data_source=str(out.get("data_source", "")),
        run_dir=run_dir,
        trade_plan=trade_plan,
    )
    if log_id:
        res_log = resolve_pending_outcomes(cfg, skip_mt5=skip_mt5)
        report["prediction_log"] = {"log_id": log_id, **res_log}

    if dedupe_bar:
        _set_last_signal_bar(current_bar_str)

    report["runtime_risk"] = risk_ctx
    report.update(out)
    return report


def ensure_report_dir(run_dir: Path) -> Path:
    rd = run_dir / "stage_9"
    rd.mkdir(parents=True, exist_ok=True)
    return rd


def run_stage_9(
    *,
    config_path: Path,
    run_dir: Path,
    dry_run: bool = False,
    skip_mt5: bool = False,
    send_telegram: bool = True,
) -> Dict[str, Any]:
    cfg = load_pipeline_config(config_path)
    return run_daily_report(
        cfg,
        run_dir=run_dir,
        dry_run=dry_run,
        skip_mt5=skip_mt5,
        send_telegram=send_telegram,
    )


def run_test_telegram(cfg: Dict[str, Any]) -> Dict[str, Any]:
    risk = cfg.get("risk", {})
    token = str(risk.get("telegram_token", ""))
    chat_id = str(risk.get("telegram_chat_id", ""))
    if is_placeholder(token) or is_placeholder(chat_id):
        return {
            "ok": False,
            "error": "Isi telegram_token & telegram_chat_id di configs/pipeline.secrets.yaml",
            "hint": "python scripts/setup_telegram.py --write-secrets",
        }
    symbol = str(cfg.get("project", {}).get("symbol", "XAUUSD"))
    tf = str(cfg.get("project", {}).get("timeframe", "H1")).upper()
    text = (
        f"*Tes koneksi — {symbol} Pipeline ({tf})*\n\n"
        "Telegram terhubung. Stage 9 siap.\n"
        f"Waktu: {datetime.now(WIB).strftime('%Y-%m-%d %H:%M WIB')}"
    )
    return send_telegram_message(token, chat_id, text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 9 — XAUUSD on-demand analysis + Telegram")
    ap.add_argument("--config", type=Path, default=project_root() / "configs" / "pipeline.yaml")
    ap.add_argument("--run-dir", type=Path, default=None, help="Folder run berisi stage_5/xgb_model.joblib")
    ap.add_argument("--latest-run", action="store_true", help="Pakai run pipeline terbaru di artifacts/")
    ap.add_argument("--dry-run", action="store_true", help="Tanpa inferensi / Telegram")
    ap.add_argument("--skip-mt5", action="store_true", help="Data dari CSV (bukan MT5 live)")
    ap.add_argument("--test-telegram", action="store_true", help="Hanya kirim pesan uji ke Telegram")
    ap.add_argument("--no-telegram", action="store_true", help="Inferensi tanpa kirim Telegram")
    ap.add_argument("--verbose", action="store_true", help="Tampilkan field diagnostik lengkap di output JSON")
    ap.add_argument(
        "--bot",
        action="store_true",
        help="Mode bot: dengarkan /analisa dari user (polling Telegram)",
    )
    args = ap.parse_args()

    cfg = load_pipeline_config(args.config)
    risk = cfg.get("risk", {})
    token = str(risk.get("telegram_token", ""))
    if not is_placeholder(token):
        try:
            telegram_set_commands(
                token,
                [
                    {"command": "analisa", "description": "Analisa XAUUSD on-demand"},
                    {"command": "status", "description": "Status model & drift sistem"},
                    {"command": "trades", "description": "Ringkasan live trades"},
                ],
            )
        except Exception as exc:
            # Tetap lanjutkan bot meski setMyCommands gagal sementara.
            print(f"[WARN] telegram_set_commands gagal: {exc}")

    if args.test_telegram:
        print(json.dumps(run_test_telegram(cfg), indent=2, default=str))
        return

    preferred = None if (args.latest_run or args.run_dir is None) else Path(args.run_dir)
    run_dir = resolve_pipeline_run_dir(
        preferred,
        output_root=str(cfg.get("experiment", {}).get("output_root", "artifacts")),
    )
    print(f"Menggunakan run_dir: {run_dir}")

    if args.bot:
        from utils.telegram_bot import run_bot_polling

        run_bot_polling(
            cfg,
            run_dir,
            run_analysis=run_daily_report,
            skip_mt5=args.skip_mt5,
        )
    else:
        rep = run_daily_report(
            cfg,
            run_dir=run_dir,
            dry_run=args.dry_run,
            skip_mt5=args.skip_mt5,
            send_telegram=not args.no_telegram,
            verbose=args.verbose,
        )
        print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
