"""Walk-forward retrain scheduler + drift monitor (PSI-based)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from stage_1_data import run_stage_1
from stage_2_labeling import run_stage_2
from stage_3_features import run_stage_3
from stage_4_validation import run_stage_4
from stage_5_training import run_stage_5
from stage_6_ensemble import run_stage_6
from utils.config_loader import load_pipeline_config
from utils.paths import project_root


def population_stability_index(base: np.ndarray, curr: np.ndarray, bins: int = 10) -> float:
    """Hitung PSI antara distribusi baseline vs current."""
    base = base[np.isfinite(base)]
    curr = curr[np.isfinite(curr)]
    if len(base) < 30 or len(curr) < 30:
        return 0.0
    cuts = np.quantile(base, np.linspace(0.0, 1.0, bins + 1))
    cuts[0] = -np.inf
    cuts[-1] = np.inf
    b = np.histogram(base, bins=cuts)[0].astype(float)
    c = np.histogram(curr, bins=cuts)[0].astype(float)
    b = np.maximum(b / (b.sum() + 1e-12), 1e-6)
    c = np.maximum(c / (c.sum() + 1e-12), 1e-6)
    return float(np.sum((c - b) * np.log(c / b)))


def compute_drift_status(df_old: pd.DataFrame, df_new: pd.DataFrame, features: List[str]) -> Dict[str, float]:
    psi_map: Dict[str, float] = {}
    for f in features[:10]:
        if f in df_old.columns and f in df_new.columns:
            psi_map[f] = population_stability_index(
                df_old[f].to_numpy(dtype=float),
                df_new[f].to_numpy(dtype=float),
            )
    return psi_map


def append_retrain_log(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrain scheduler + PSI drift check")
    ap.add_argument("--config", type=Path, default=project_root() / "configs" / "pipeline.yaml")
    ap.add_argument("--dummy", action="store_true", help="Run end-to-end with dummy drift check")
    ap.add_argument("--skip-task1", action="store_true")
    ap.add_argument("--skip-task2", action="store_true")
    ap.add_argument("--skip-task3", action="store_true")
    ap.add_argument("--skip-task4", action="store_true")
    ap.add_argument("--skip-task5", action="store_true")
    ap.add_argument("--skip-task6", action="store_true")
    ap.add_argument("--skip-fix1", action="store_true")
    ap.add_argument("--skip-fix2", action="store_true")
    args = ap.parse_args()

    cfg = load_pipeline_config(args.config)
    root = project_root()
    out_root = root / str(cfg.get("experiment", {}).get("output_root", "artifacts"))
    run_dir = out_root / f"run_retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    df1, _ = run_stage_1(cfg, run_dir=run_dir)
    df2, _ = run_stage_2(df1, cfg, run_dir=run_dir)
    df3, _ = run_stage_3(df2, cfg, run_dir=run_dir)
    df_train = df3.loc[df3["keep_for_training"] == 1].reset_index(drop=True)
    run_stage_4(df_train, cfg, run_dir=run_dir)
    meta5 = run_stage_5(
        df_train,
        cfg,
        run_dir=run_dir,
        task_flags={
            "skip_task1": bool(args.skip_task1),
            "skip_task2": bool(args.skip_task2),
            "skip_task3": bool(args.skip_task3),
            "skip_task4": bool(args.skip_task4),
            "skip_task5": bool(args.skip_task5),
            "skip_task6": bool(args.skip_task6),
            "skip_fix1": bool(args.skip_fix1),
            "skip_fix2": bool(args.skip_fix2),
        },
    )
    run_stage_6(df_train, cfg, run_dir=run_dir)

    feat_path = run_dir / "stage_5" / "feature_registry.json"
    reg = json.loads(feat_path.read_text(encoding="utf-8")) if feat_path.is_file() else {}
    feats = list(reg.get("active_features", []))

    # Baseline vs current dummy split untuk deteksi drift.
    split = max(int(len(df_train) * 0.7), 1)
    df_old = df_train.iloc[:split].copy()
    df_new = df_train.iloc[split:].copy()
    if args.dummy and len(df_new):
        # Simulasikan drift ringan untuk uji end-to-end.
        for c in feats[:2]:
            if c in df_new.columns:
                df_new[c] = df_new[c] * 1.15

    psi_map = compute_drift_status(df_old, df_new, feats)
    high = [k for k, v in psi_map.items() if v > 0.2]
    drift_status = "HIGH_DRIFT" if len(high) >= 2 else "NORMAL"

    runtime_risk = {
        "drift_status": drift_status,
        "high_drift_features": high,
        "risk_multiplier": 0.7 if drift_status == "HIGH_DRIFT" else 1.0,
        "psi": psi_map,
    }
    (root / "logs" / "runtime_risk.json").write_text(json.dumps(runtime_risk, indent=2), encoding="utf-8")

    log_item = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": run_dir.as_posix(),
        "drift_status": drift_status,
        "high_drift_features": high,
        "psi": psi_map,
        "holdout_macro_f1": meta5.get("holdout_macro_f1"),
        "holdout_balanced_accuracy": meta5.get("holdout_balanced_accuracy"),
    }
    append_retrain_log(root / "logs" / "retrain_log.jsonl", log_item)
    print(json.dumps(log_item, indent=2))


if __name__ == "__main__":
    main()

