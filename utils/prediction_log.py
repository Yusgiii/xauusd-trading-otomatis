"""Log prediksi live + evaluasi outcome (untuk ukur akurasi & profitabilitas sinyal)."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from utils.horizon import horizon_label_id, horizon_timedelta
from utils.paths import project_root

WIB = ZoneInfo("Asia/Jakarta")
CLASS_NAMES = ("FLAT", "UP", "DOWN")

FIELDNAMES = [
    "log_id",
    "ts_utc",
    "bar_time_utc",
    "symbol",
    "close_at_signal",
    "pred_class",
    "prob_flat",
    "prob_up",
    "prob_down",
    "sentiment",
    "recommendation",
    "lean_delta",
    "combined_score",
    "data_source",
    "run_dir",
    "lot_size",
    "confidence_tier",
    "p_up_binary",
    "p_down_binary",
    "tp_rr",
    "sl_pips",
    "tp_pips",
    "model_source",
    "outcome_resolved",
    "outcome_bar_time_utc",
    "close_at_outcome",
    "forward_log_return",
    "outcome_class",
    "ml_correct",
    "signal_correct",
    "signal_pnl_proxy",
]


def log_csv_path(cfg: Dict[str, Any]) -> Path:
    rel = cfg.get("stage_9", {}).get("prediction_log", {}).get(
        "csv_path", "logs/prediction_log.csv"
    )
    p = Path(rel)
    return p if p.is_absolute() else project_root() / p


def _enabled(cfg: Dict[str, Any]) -> bool:
    pl = cfg.get("stage_9", {}).get("prediction_log", {})
    return bool(pl.get("enabled", True))


def classify_log_return(log_ret: float, threshold: float) -> int:
    if log_ret > threshold:
        return 1
    if log_ret < -threshold:
        return 2
    return 0


def recommendation_pnl_proxy(recommendation: str, forward_log_return: float, threshold: float) -> int:
    """
    Proxy arah PnL sederhana (bukan pip): +1 menguntungkan arah sinyal, -1 merugikan, 0 netral.
    """
    rec = recommendation.upper()
    if "ABSTAIN" in rec or rec.startswith("NEUTRAL"):
        return 0
    lr = forward_log_return
    if abs(lr) <= threshold:
        return 0
    if "BUY" in rec:
        return 1 if lr > 0 else -1
    if "SELL" in rec:
        return 1 if lr < 0 else -1
    return 0


def signal_matches_outcome(recommendation: str, outcome_class: int) -> Optional[bool]:
    rec = recommendation.upper()
    if "ABSTAIN" in rec or rec.startswith("NEUTRAL"):
        return None if outcome_class != 0 else True
    if "BUY" in rec:
        return outcome_class == 1
    if "SELL" in rec:
        return outcome_class == 2
    return None


def append_prediction(
    cfg: Dict[str, Any],
    *,
    bar_time_utc: datetime,
    symbol: str,
    close_at_signal: float,
    pred_class_idx: int,
    probs: np.ndarray,
    sentiment: int,
    recommendation: str,
    consensus: Dict[str, Any],
    data_source: str,
    run_dir: Path,
    trade_plan: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not _enabled(cfg):
        return None

    path = log_csv_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and path.stat().st_size > 0:
        try:
            header = path.read_text(encoding="utf-8").splitlines()[0]
            if "lot_size" not in header:
                old_df = pd.read_csv(path, engine="python", on_bad_lines="skip")
                for col in FIELDNAMES:
                    if col not in old_df.columns:
                        old_df[col] = ""
                old_df = old_df.reindex(columns=FIELDNAMES, fill_value="")
                with path.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
                    w.writeheader()
                    for rec in old_df.to_dict("records"):
                        w.writerow({k: rec.get(k, "") for k in FIELDNAMES})
        except Exception:
            pass

    log_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    tp = trade_plan or {}
    row = {
        "log_id": log_id,
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "bar_time_utc": bar_time_utc.isoformat(),
        "symbol": symbol,
        "close_at_signal": f"{close_at_signal:.5f}",
        "pred_class": CLASS_NAMES[pred_class_idx],
        "prob_flat": f"{float(probs[0]):.6f}",
        "prob_up": f"{float(probs[1]):.6f}",
        "prob_down": f"{float(probs[2]):.6f}",
        "sentiment": str(int(sentiment)),
        "recommendation": recommendation,
        "lean_delta": f"{float(consensus.get('lean_delta', 0)):.6f}",
        "combined_score": f"{float(consensus.get('combined', 0)):.6f}",
        "data_source": data_source,
        "run_dir": run_dir.name,
        "lot_size": f"{float(tp.get('lot_size', 0.0)):.4f}",
        "confidence_tier": str(tp.get("confidence_tier", "UNKNOWN")),
        "p_up_binary": f"{float(tp.get('p_up', 0.0)):.6f}",
        "p_down_binary": f"{float(tp.get('p_down', 0.0)):.6f}",
        "tp_rr": f"{float(tp.get('rr_actual', tp.get('risk_reward', 0.0))):.4f}",
        "sl_pips": f"{float(tp.get('sl_distance_pips', 0.0)):.2f}",
        "tp_pips": f"{float(tp.get('tp_distance_pips', 0.0)):.2f}",
        "model_source": str(tp.get("model_source", "unknown")),
        "outcome_resolved": "0",
        "outcome_bar_time_utc": "",
        "close_at_outcome": "",
        "forward_log_return": "",
        "outcome_class": "",
        "ml_correct": "",
        "signal_correct": "",
        "signal_pnl_proxy": "",
    }

    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)
    return log_id


def _load_price_series(cfg: Dict[str, Any], *, skip_mt5: bool) -> Optional[pd.DataFrame]:
    """Ambil seri waktu close untuk resolve outcome."""
    if not skip_mt5:
        try:
            import MetaTrader5 as mt5

            from utils.mt5_export import mt5_timeframe_const, resolve_symbol

            from utils.mt5_connection import initialize_mt5, shutdown_mt5

            ok, mt5 = initialize_mt5(cfg)
            if ok:
                sym = resolve_symbol(mt5, str(cfg.get("project", {}).get("symbol", "XAUUSD")))
                tf = str(cfg.get("project", {}).get("timeframe", "H1")).upper()
                tf_const = mt5_timeframe_const(mt5, tf)
                rates = mt5.copy_rates_from_pos(sym, tf_const, 0, 300)
                shutdown_mt5(mt5)
                if rates is not None and len(rates) > 0:
                    from utils.trading_calendar import drop_weekend_bars

                    df = pd.DataFrame(rates)
                    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
                    df = df[["time", "close"]]
                    if bool(cfg.get("stage_1", {}).get("exclude_weekend_bars", True)):
                        df, _ = drop_weekend_bars(df)
                    return df
        except Exception:
            pass

    csv_rel = cfg.get("stage_1", {}).get("input_csv", "data/xauusd_h1.csv")
    csv_path = Path(csv_rel)
    if not csv_path.is_absolute():
        csv_path = project_root() / csv_path
    if not csv_path.is_file():
        return None
    from utils.trading_calendar import drop_weekend_bars

    df = pd.read_csv(csv_path, usecols=["time", "close"])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna().sort_values("time")
    if bool(cfg.get("stage_1", {}).get("exclude_weekend_bars", True)):
        df, _ = drop_weekend_bars(df)
    return df


def resolve_pending_outcomes(
    cfg: Dict[str, Any],
    *,
    skip_mt5: bool = False,
) -> Dict[str, Any]:
    """Isi outcome setelah horizon (1 bar timeframe aktif)."""
    if not _enabled(cfg):
        return {"resolved": 0, "skipped": "disabled"}

    path = log_csv_path(cfg)
    if not path.is_file():
        return {"resolved": 0, "skipped": "no log file"}

    df_log = pd.read_csv(path, engine="python", on_bad_lines="skip")
    for col in FIELDNAMES:
        if col not in df_log.columns:
            df_log[col] = ""
    df_log = df_log.reindex(columns=FIELDNAMES, fill_value="")
    if df_log.empty:
        return {"resolved": 0}

    prices = _load_price_series(cfg, skip_mt5=skip_mt5)
    if prices is None or prices.empty:
        return {"resolved": 0, "error": "no price data"}

    prices = prices.set_index("time")["close"].sort_index()
    thresh = float(cfg.get("risk", {}).get("flat_return_threshold", 0.0005))
    horizon_td = horizon_timedelta(cfg)

    pending = df_log[df_log["outcome_resolved"].astype(str) == "0"]
    resolved_count = 0
    rows_out = df_log.to_dict("records")

    for i, row in enumerate(rows_out):
        if str(row.get("outcome_resolved")) != "0":
            continue
        try:
            t0 = pd.Timestamp(row["bar_time_utc"])
            if t0.tzinfo is None:
                t0 = t0.tz_localize("UTC")
            else:
                t0 = t0.tz_convert("UTC")
        except Exception:
            continue

        t1 = t0 + horizon_td
        if t1 > prices.index.max():
            continue

        # Bar outcome: close pada atau setelah t1
        future = prices[prices.index >= t1]
        if future.empty:
            continue
        close_out = float(future.iloc[0])
        close_in = float(row["close_at_signal"])
        lr = float(np.log(close_out / close_in))
        oc = classify_log_return(lr, thresh)
        pc = CLASS_NAMES.index(str(row["pred_class"])) if row["pred_class"] in CLASS_NAMES else -1
        ml_ok = int(pc == oc) if pc >= 0 else ""
        sig = signal_matches_outcome(str(row["recommendation"]), oc)
        sig_ok = "" if sig is None else int(sig)
        pnl = recommendation_pnl_proxy(str(row["recommendation"]), lr, thresh)

        rows_out[i].update(
            {
                "outcome_resolved": "1",
                "outcome_bar_time_utc": future.index[0].isoformat(),
                "close_at_outcome": f"{close_out:.5f}",
                "forward_log_return": f"{lr:.8f}",
                "outcome_class": CLASS_NAMES[oc],
                "ml_correct": str(ml_ok),
                "signal_correct": str(sig_ok),
                "signal_pnl_proxy": str(pnl),
            }
        )
        resolved_count += 1

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    return {"resolved": resolved_count, "pending_left": int((pd.read_csv(path)["outcome_resolved"] == "0").sum())}


def summarize_period(
    cfg: Dict[str, Any],
    *,
    days: int = 7,
) -> Dict[str, Any]:
    """Ringkasan akurasi untuk laporan mingguan."""
    path = log_csv_path(cfg)
    if not path.is_file():
        return {"error": "no log", "days": days}

    df = pd.read_csv(path)
    if df.empty:
        return {"error": "empty log", "days": days}

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    df = df[df["ts_utc"] >= cutoff]
    resolved = df[df["outcome_resolved"].astype(str) == "1"].copy()

    out: Dict[str, Any] = {
        "days": days,
        "symbol": str(cfg.get("project", {}).get("symbol", "XAUUSD")),
        "horizon_label": horizon_label_id(cfg),
        "n_predictions": int(len(df)),
        "n_resolved": int(len(resolved)),
        "n_pending": int(len(df) - len(resolved)),
    }
    if resolved.empty:
        out["note"] = (
            f"Belum ada outcome ter-resolve — tunggu {out['horizon_label']} atau jalankan resolve."
        )
        return out

    out["ml_accuracy"] = float((resolved["ml_correct"].astype(int) == 1).mean())
    sig = resolved[resolved["signal_correct"].astype(str).isin(["0", "1"])]
    if len(sig):
        out["signal_accuracy"] = float((sig["signal_correct"].astype(int) == 1).mean())
        out["signal_n"] = int(len(sig))
    pnl = resolved["signal_pnl_proxy"].astype(int)
    out["pnl_proxy_sum"] = int(pnl.sum())
    out["pnl_proxy_win_rate"] = float((pnl > 0).mean()) if len(pnl) else 0.0

    def _sig_acc(s: pd.Series) -> float:
        v = pd.to_numeric(s, errors="coerce").dropna()
        return float((v == 1).mean()) if len(v) else 0.0

    by_rec = (
        resolved.groupby("recommendation")
        .agg(
            n=("log_id", "count"),
            signal_acc=("signal_correct", _sig_acc),
            pnl_sum=("signal_pnl_proxy", lambda s: pd.to_numeric(s, errors="coerce").fillna(0).astype(int).sum()),
        )
        .reset_index()
    )
    out["by_recommendation"] = by_rec.to_dict("records")
    return out


def format_accuracy_message(summary: Dict[str, Any], days: int) -> str:
    """Format laporan akurasi untuk Telegram (/akurasi & laporan mingguan)."""
    hlab = str(summary.get("horizon_label", "24 jam ke depan"))
    symbol = str(summary.get("symbol", "XAUUSD"))
    if summary.get("error"):
        return (
            f"*Laporan akurasi {days} hari*\n\n"
            f"Belum cukup data: {summary.get('error')}\n"
            f"Jalankan /analisis, tunggu {hlab}, lalu /akurasi lagi."
        )
    if summary.get("n_resolved", 0) == 0 and summary.get("n_predictions", 0) > 0:
        return (
            f"*{symbol} — Laporan akurasi {days} hari*\n\n"
            f"• Prediksi tercatat: *{summary['n_predictions']}*\n"
            f"• Outcome selesai: *0* (pending: {summary.get('n_pending', 0)})\n\n"
            f"{summary.get('note', 'Belum ada outcome ter-resolve.')}\n"
            f"Tunggu {hlab} atau pastikan MT5 aktif, lalu /akurasi lagi."
        )

    lines = [
        f"*{symbol} — Laporan akurasi {days} hari*",
        "",
        f"• Prediksi tercatat: *{summary['n_predictions']}*",
        f"• Outcome selesai: *{summary['n_resolved']}* (pending: {summary['n_pending']})",
    ]
    if summary.get("n_resolved", 0) > 0:
        lines.append(f"• Akurasi model (ML): *{summary['ml_accuracy']:.1%}*")
        if summary.get("signal_n"):
            lines.append(
                f"• Akurasi sinyal BUY/SELL: *{summary['signal_accuracy']:.1%}* "
                f"(n={summary['signal_n']})"
            )
        lines.append(
            f"• PnL proxy (arah): win rate *{summary.get('pnl_proxy_win_rate', 0):.1%}* "
            f"| total skor *{summary.get('pnl_proxy_sum', 0):+d}*"
        )
        lines.append("")
        lines.append("_Per rekomendasi:_")
        for row in summary.get("by_recommendation", [])[:8]:
            lines.append(
                f"• {row['recommendation']}: n={row['n']} "
                f"acc={row.get('signal_acc', 0):.0%} pnl_sum={row.get('pnl_sum', 0):+d}"
            )
    lines.append("")
    lines.append("_Catatan: PnL proxy = arah benar/salah, bukan uang riil._")
    return "\n".join(lines)
