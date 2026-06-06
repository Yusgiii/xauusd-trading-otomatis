# noqa: D100
"""Unduh riwayat OHLCV dari MT5 sesuai timeframe di configs/pipeline.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_pipeline_config
from utils.mt5_export import export_ohlcv_csv
from utils.paths import project_root


def main() -> None:
    ap = argparse.ArgumentParser(description="Ekspor OHLCV dari MetaTrader 5")
    ap.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "pipeline.yaml",
    )
    ap.add_argument("--symbol", type=str, default=None)
    ap.add_argument("--bars", type=int, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--timeframe", type=str, default=None, help="Override timeframe (D1, H1, …)")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Timpa CSV meski sudah ada (unduh ulang semua riwayat MT5)",
    )
    args = ap.parse_args()

    cfg = load_pipeline_config(args.config)
    proj = cfg["project"]
    s1 = cfg["stage_1"]
    symbol = args.symbol or str(proj["symbol"])
    tf = (args.timeframe or str(proj.get("timeframe", "D1"))).upper()
    bars = args.bars or int(s1.get("mt5_fetch_bars", 100_000))

    out = args.output
    if out is None:
        rel = Path(str(s1.get("input_csv", f"data/{symbol.lower()}_{tf.lower()}.csv")))
        out = rel if rel.is_absolute() else project_root() / rel

    if out.is_file() and not args.force:
        import pandas as pd

        old = pd.read_csv(out)
        print(f"SKIP | {out} sudah ada ({len(old)} bar). Gunakan --force untuk unduh ulang.")
        return

    n = export_ohlcv_csv(symbol, out, timeframe=tf, n_bars=bars)
    if out.is_file():
        import pandas as pd

        df = pd.read_csv(out)
        t0 = pd.to_datetime(df["time"]).min()
        t1 = pd.to_datetime(df["time"]).max()
        print(f"OK | {symbol} {tf} | {n} bar | {t0.date()} -> {t1.date()} | -> {out}")
    else:
        print(f"OK | {symbol} {tf} | {n} bar | -> {out}")


if __name__ == "__main__":
    main()
