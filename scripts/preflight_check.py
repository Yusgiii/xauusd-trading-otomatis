# noqa: D100
"""Cek kesiapan sistem sebelum menjalankan bot / pipeline."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config_loader import load_pipeline_config
from utils.paths import project_root, resolve_pipeline_run_dir
from utils.telegram_notify import is_placeholder, telegram_get_me


def _ok(msg: str) -> str:
    return f"[OK] {msg}"


def _warn(msg: str) -> str:
    return f"[WARN] {msg}"


def _fail(msg: str) -> str:
    return f"[FAIL] {msg}"


def main() -> int:
    lines: list[str] = []
    errors = 0
    warns = 0

    lines.append("=== PREFLIGHT XAUUSD Pipeline ===\n")

    # Config
    cfg_path = project_root() / "configs" / "pipeline.yaml"
    secrets_path = project_root() / "configs" / "pipeline.secrets.yaml"
    if not cfg_path.is_file():
        lines.append(_fail(f"Config hilang: {cfg_path}"))
        errors += 1
    else:
        lines.append(_ok(f"Config: {cfg_path.name}"))

    if not secrets_path.is_file():
        lines.append(_fail(f"Secrets hilang: salin dari pipeline.secrets.yaml.example"))
        errors += 1
    else:
        lines.append(_ok("pipeline.secrets.yaml ada"))

    try:
        cfg = load_pipeline_config(cfg_path)
    except Exception as exc:
        lines.append(_fail(f"Load config: {exc}"))
        return 1

    risk = cfg.get("risk", {})
    proj = cfg.get("project", {})

    # Secrets fields
    for key in ("telegram_token", "telegram_chat_id", "newsapi_key", "gemini_api_key"):
        v = str(risk.get(key, ""))
        if is_placeholder(v):
            lines.append(_fail(f"risk.{key} belum diisi"))
            errors += 1
        else:
            lines.append(_ok(f"risk.{key} terisi"))

    # Telegram
    tok = str(risk.get("telegram_token", ""))
    if not is_placeholder(tok):
        me = telegram_get_me(tok)
        if me.get("ok"):
            u = me.get("result", {}).get("username", "?")
            lines.append(_ok(f"Telegram bot: @{u}"))
        else:
            lines.append(_fail(f"Telegram token invalid: {me}"))
            errors += 1

    cid = str(risk.get("telegram_chat_id", ""))
    if cid.startswith("@"):
        lines.append(_fail("telegram_chat_id harus angka, bukan @username"))
        errors += 1

    # Data CSV
    tf = str(proj.get("timeframe", "H1")).upper()
    csv_rel = cfg.get("stage_1", {}).get("input_csv", f"data/xauusd_{tf.lower()}.csv")
    csv_path = Path(csv_rel)
    if not csv_path.is_absolute():
        csv_path = project_root() / csv_path
    if csv_path.is_file():
        import pandas as pd
        from utils.trading_calendar import weekend_mask

        cdf = pd.read_csv(csv_path, usecols=["time"])
        cdf["time"] = pd.to_datetime(cdf["time"], utc=True, errors="coerce")
        n = len(cdf)
        wk = int(weekend_mask(cdf["time"]).sum())
        if wk:
            lines.append(
                _fail(
                    f"CSV masih punya {wk} bar Sabtu/Minggu — jalankan: "
                    "python scripts/fetch_ohlcv_from_mt5.py --force"
                )
            )
            errors += 1
        else:
            lines.append(_ok(f"Data CSV: {csv_path.name} ({n} bar, tanpa weekend)"))
        min_bars = 800 if tf == "D1" else 5000
        if n < min_bars:
            lines.append(_warn(f"Bar CSV sedikit — pertimbangkan fetch_ohlcv_from_mt5 ({tf})"))
            warns += 1
    else:
        lines.append(_warn(f"CSV belum ada: {csv_path} (Stage 1 bisa auto-fetch MT5)"))
        warns += 1

    # Model
    try:
        run_dir = resolve_pipeline_run_dir(
            None,
            output_root=str(cfg.get("experiment", {}).get("output_root", "artifacts")),
        )
        lines.append(_ok(f"Model: {run_dir.name}/stage_5/xgb_model.joblib"))
    except FileNotFoundError:
        lines.append(_fail("Tidak ada run dengan xgb_model.joblib — jalankan: python run_pipeline.py"))
        errors += 1

    # Python packages
    pkgs = [
        ("pandas", None),
        ("numpy", None),
        ("yaml", "pyyaml"),
        ("xgboost", None),
        ("sklearn", "scikit-learn"),
        ("optuna", None),
        ("joblib", None),
        ("requests", None),
        ("feedparser", None),
        ("MetaTrader5", None),
        ("google.genai", "google-genai"),
    ]
    lines.append("")
    lines.append("Dependensi:")
    for mod, pip_name in pkgs:
        name = pip_name or mod
        try:
            importlib.import_module(mod)
            lines.append(_ok(name))
        except ImportError:
            lines.append(_fail(f"{name} — pip install {pip_name or mod}"))
            errors += 1

    # MT5 quick test
    lines.append("")
    lines.append("MetaTrader5:")
    try:
        from utils.mt5_connection import initialize_mt5, shutdown_mt5
        from utils.mt5_export import mt5_timeframe_const, resolve_symbol

        ok, mt5 = initialize_mt5(cfg)
        if ok:
            sym = str(proj.get("symbol", "XAUUSD"))
            try:
                resolved = resolve_symbol(mt5, sym)
                tf_const = mt5_timeframe_const(mt5, tf)
                rates = mt5.copy_rates_from_pos(resolved, tf_const, 0, 5)
                if rates is not None and len(rates) > 0:
                    lines.append(_ok(f"MT5 connect + {resolved} {tf} ({len(rates)} bar sample)"))
                else:
                    lines.append(_warn(f"MT5 OK tapi tidak ada bar {resolved} — buka chart {tf}"))
                    warns += 1
            finally:
                shutdown_mt5(mt5)
        else:
            lines.append(_warn("MT5 tidak initialize — buka terminal MT5 & login"))
            warns += 1
    except ImportError:
        lines.append(_warn("MetaTrader5 tidak terpasang"))
        warns += 1
    except Exception as exc:
        lines.append(_warn(f"MT5: {exc}"))
        warns += 1

    # NewsAPI quick
    lines.append("")
    lines.append("NewsAPI:")
    try:
        from utils.news_fetch import fetch_newsapi_headlines

        h, lab = fetch_newsapi_headlines(
            str(risk.get("newsapi_key", "")),
            query="XAUUSD",
            max_items=1,
        )
        if lab == "newsapi" and h and not h[0].startswith("("):
            lines.append(_ok("NewsAPI merespons"))
        else:
            lines.append(_warn(f"NewsAPI: {lab} {h[:1]}"))
            warns += 1
    except Exception as exc:
        lines.append(_warn(f"NewsAPI: {exc}"))
        warns += 1

    # Gemini quick
    lines.append("")
    lines.append("Gemini:")
    try:
        from utils.gemini_client import gemini_news_sentiment

        s9 = cfg.get("stage_9", {})
        score, note, _conf = gemini_news_sentiment(
            ["Japan GDP data", "BoE rates"],
            str(risk.get("gemini_api_key", "")),
            model=str(s9.get("gemini_model", "gemini-2.5-flash")),
            fallback_models=list(s9.get("gemini_fallback_models") or []),
        )
        if "tidak tersedia" in note.lower() or "error" in note.lower():
            lines.append(_warn(f"Gemini: {note[:100]}"))
            warns += 1
        else:
            lines.append(_ok(f"Gemini skor={score} | {note[:60]}..."))
    except Exception as exc:
        lines.append(_warn(f"Gemini: {exc}"))
        warns += 1

    # Summary
    lines.append("")
    lines.append("--- RINGKASAN ---")
    if errors == 0 and warns == 0:
        lines.append(_ok("Siap dijalankan (Stage 9 service / Task Scheduler)."))
        code = 0
    elif errors == 0:
        lines.append(_warn(f"Bisa jalan dengan {warns} peringatan (lihat di atas)."))
        code = 0
    else:
        lines.append(_fail(f"{errors} error wajib diperbaiki sebelum produksi."))
        if warns:
            lines.append(_warn(f"+ {warns} peringatan"))
        code = 1

    print("\n".join(lines))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
