# noqa: D100
"""Orchestrator — pipeline XAUUSD directional forecasting (configs/pipeline.yaml)."""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from stage_1_data import run_stage_1
from stage_2_labeling import run_stage_2
from stage_3_features import run_stage_3
from stage_4_validation import run_stage_4
from stage_5_training import run_stage_5
from stage_6_ensemble import run_stage_6
from stage_9_live_demo import run_stage_9
from utils.config_loader import load_pipeline_config
from utils.experiment import append_experiment_log, write_json
from utils.paths import project_root


def _check_dependencies() -> None:
    """Cek versi dependency kritis sebelum pipeline jalan."""
    import sklearn
    from packaging import version

    sk_ver = version.parse(sklearn.__version__)
    sk_min = version.parse("1.3.0")
    sk_max = version.parse("1.5.0")

    if not (sk_min <= sk_ver < sk_max):
        raise RuntimeError(
            f"scikit-learn versi {sklearn.__version__} tidak kompatibel.\n"
            f"Gunakan: pip install 'scikit-learn>=1.3.0,<1.5.0'\n"
            f"Atau: pip install scikit-learn==1.3.2"
        )


_check_dependencies()


def main() -> None:
    ap = argparse.ArgumentParser(description="XAUUSD directional ML pipeline (config timeframe)")
    ap.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs" / "pipeline.yaml",
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional explicit run directory (default: artifacts/run_<UTC timestamp>)",
    )
    ap.add_argument("--stage9", action="store_true", help="Jalankan laporan live sekali setelah training")
    ap.add_argument("--dry-run-stage9", action="store_true", help="Stage9 tanpa kirim Telegram")
    ap.add_argument("--skip-task1", action="store_true", help="Skip task1 profit-aware objective & threshold")
    ap.add_argument("--skip-task2", action="store_true", help="Skip task2 meta-label execution filter")
    ap.add_argument("--skip-task3", action="store_true", help="Skip task3 probability calibration")
    ap.add_argument("--skip-task4", action="store_true", help="Skip task4 regime-aware training")
    ap.add_argument("--skip-task5", action="store_true", help="Skip task5 feature validation & cleanup")
    ap.add_argument("--skip-task6", action="store_true", help="Skip task6 scheduler related hooks")
    ap.add_argument("--skip-fix1", action="store_true", help="Skip fix1 hybrid calibration")
    ap.add_argument("--skip-fix2", action="store_true", help="Skip fix2 meta-model precision optimization")
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_pipeline_config(args.config)
    root = project_root()
    if args.run_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = root / cfg["experiment"]["output_root"] / f"run_{ts}"
    else:
        run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, run_dir / "pipeline_config_used.yaml")

    df1, _ = run_stage_1(cfg, run_dir=run_dir)
    df2, _ = run_stage_2(df1, cfg, run_dir=run_dir)
    df3, _ = run_stage_3(df2, cfg, run_dir=run_dir)

    df_train = df3.loc[df3["keep_for_training"] == 1].reset_index(drop=True)

    _, _ = run_stage_4(df_train, cfg, run_dir=run_dir)
    task_flags = {
        "skip_task1": bool(args.skip_task1),
        "skip_task2": bool(args.skip_task2),
        "skip_task3": bool(args.skip_task3),
        "skip_task4": bool(args.skip_task4),
        "skip_task5": bool(args.skip_task5),
        "skip_task6": bool(args.skip_task6),
        "skip_fix1": bool(args.skip_fix1),
        "skip_fix2": bool(args.skip_fix2),
    }
    meta5 = run_stage_5(df_train, cfg, run_dir=run_dir, task_flags=task_flags)
    meta6 = run_stage_6(df_train, cfg, run_dir=run_dir)

    summary: Dict[str, Any] = {
        "run_dir": run_dir.as_posix(),
        "pipeline": f"xauusd_directional_{cfg.get('project', {}).get('timeframe', 'h1').lower()}",
        "n_train_rows": len(df_train),
        "stage_5": {
            "cv_macro_f1": meta5.get("optuna", {}).get("xgb", {}).get("best_value"),
            "holdout_macro_f1": meta5.get("holdout_macro_f1"),
            "holdout_balanced_accuracy": meta5.get("holdout_balanced_accuracy"),
            "binary_models": meta5.get("binary_models"),
        },
        "stage_6": meta6,
    }
    write_json(run_dir / "run_summary.json", summary)
    append_experiment_log(root / cfg["experiment"]["output_root"], {"run": summary})

    from utils.paths import write_active_run_pointer

    write_active_run_pointer(run_dir, output_root=str(cfg.get("experiment", {}).get("output_root", "artifacts")))

    if args.stage9:
        run_stage_9(config_path=args.config, run_dir=run_dir, dry_run=args.dry_run_stage9)

    print(f"Pipeline selesai | run_dir={run_dir}")


if __name__ == "__main__":
    main()
