# noqa: D100
"""Resolve outcome bar yang sudah tersedia + tampilkan ringkasan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config_loader import load_pipeline_config
from utils.paths import project_root
from utils.prediction_log import log_csv_path, resolve_pending_outcomes, summarize_period


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--skip-mt5", action="store_true")
    args = ap.parse_args()

    cfg = load_pipeline_config(project_root() / "configs" / "pipeline.yaml")
    print(f"Log: {log_csv_path(cfg)}")
    res = resolve_pending_outcomes(cfg, skip_mt5=args.skip_mt5)
    print("Resolve:", json.dumps(res, indent=2))
    print("Summary:", json.dumps(summarize_period(cfg, days=args.days), indent=2, default=str))


if __name__ == "__main__":
    main()
