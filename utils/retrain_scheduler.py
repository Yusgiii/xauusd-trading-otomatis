"""Auto-retrain scheduler berdasarkan drift level."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from utils.paths import read_active_run_pointer

log = logging.getLogger("retrain_scheduler")

LAST_RETRAIN_STATE_FILE = "last_auto_retrain.json"
NEEDS_RESTART_FLAG = "needs_restart.flag"


def load_runtime_risk(logs_dir: Path) -> Dict[str, Any]:
    """Load runtime_risk.json, return empty dict jika tidak ada."""
    p = logs_dir / "runtime_risk.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def reset_drift_after_retrain(logs_dir: Path) -> None:
    """
    Reset runtime_risk.json setelah retrain sukses.
    Set drift ke NORMAL sementara sampai inferensi live update ulang.
    """
    risk_path = logs_dir / "runtime_risk.json"
    try:
        existing: Dict[str, Any] = {}
        if risk_path.exists():
            existing = json.loads(risk_path.read_text(encoding="utf-8"))

        existing.update(
            {
                "drift_level": "NORMAL",
                "drift_status": "NORMAL",
                "risk_multiplier": 1.0,
                "last_reset": datetime.now(timezone.utc).isoformat(),
                "reset_reason": "post_auto_retrain",
            }
        )

        risk_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        log.info("Drift reset ke NORMAL setelah retrain")
        print("[auto_retrain] Drift reset ke NORMAL", flush=True)
    except Exception as exc:
        log.warning("Gagal reset drift: %s", exc)


def write_restart_flag(project_root: Path, run_dir: Optional[Path], reason: str = "post_retrain_restart") -> None:
    """Tandai service perlu restart bersih (bukan hot-reload model)."""
    logs = project_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": run_dir.name if run_dir else "",
        "run_dir_path": str(run_dir) if run_dir else "",
        "retrain_time": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    (logs / NEEDS_RESTART_FLAG).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_last_retrain_time(logs_dir: Path) -> Optional[datetime]:
    p = logs_dir / LAST_RETRAIN_STATE_FILE
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        raw = data.get("last_retrain_utc") or data.get("retrain_time")
        if not raw:
            return None
        ts = datetime.fromisoformat(str(raw))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)
    except Exception:
        return None


def save_last_retrain_time(logs_dir: Path, when: Optional[datetime] = None) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = when or datetime.now(timezone.utc)
    iso = ts.astimezone(timezone.utc).isoformat()
    payload = {
        "last_retrain_utc": iso,
        "retrain_time": iso,
        "success": True,
    }
    (logs_dir / LAST_RETRAIN_STATE_FILE).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def should_retrain(
    risk: Dict[str, Any],
    last_retrain_time: Optional[datetime],
    min_retrain_interval_hours: float = 6.0,
    *,
    trigger_on_high_drift: bool = True,
    trigger_on_critical_drift: bool = True,
) -> Tuple[bool, str]:
    """
    Tentukan apakah perlu retrain sekarang.
    Return (should_retrain, reason)
    """
    now = datetime.now(timezone.utc)

    if last_retrain_time is not None:
        elapsed = (now - last_retrain_time).total_seconds() / 3600
        if elapsed < min_retrain_interval_hours:
            return False, f"Retrain terakhir {elapsed:.1f} jam lalu (min {min_retrain_interval_hours}h)"

    drift_level = str(risk.get("drift_level", risk.get("drift_status", "NORMAL"))).upper()

    if trigger_on_critical_drift and "CRITICAL" in drift_level:
        return True, "CRITICAL DRIFT terdeteksi"

    if trigger_on_high_drift and "HIGH" in drift_level:
        return True, "HIGH DRIFT terdeteksi"

    return False, f"Drift normal ({drift_level})"


def is_retrain_allowed_now() -> Tuple[bool, str]:
    """
    Cek apakah waktu sekarang boleh untuk retrain.
    Hindari retrain saat weekend atau dini hari UTC.
    """
    now_utc = datetime.now(timezone.utc)

    if now_utc.weekday() >= 5:
        return False, "Weekend — pasar tutup, retrain ditunda ke Senin"

    if now_utc.hour < 2:
        return False, f"Dini hari UTC ({now_utc.hour}:xx) — ditunda"

    return True, "OK"


def _parse_run_dir_from_stdout(stdout: str, project_root: Path, output_root: str) -> Optional[Path]:
    for line in stdout.splitlines():
        if "run_dir=" not in line:
            continue
        part = line.split("run_dir=")[-1].strip()
        candidate = Path(part)
        if candidate.is_dir() and (candidate / "stage_5" / "xgb_model.joblib").is_file():
            return candidate
        name = candidate.name
        if name.startswith("run_"):
            alt = project_root / output_root / name
            if alt.is_dir():
                return alt
    pointed = read_active_run_pointer(output_root)
    if pointed is not None:
        return pointed
    return None


def run_retrain(
    config_path: Path,
    project_root: Path,
    python_executable: Optional[str] = None,
    *,
    output_root: str = "artifacts",
    telegram_notify_fn: Optional[Callable[[str], None]] = None,
    timeout_seconds: int = 3600,
) -> Tuple[bool, str]:
    """
    Jalankan run_pipeline.py sebagai subprocess.
    Return (success, run_dir_path_or_error_message)
    """
    logs_dir = project_root / "logs"
    py_exe = python_executable or sys.executable
    cmd = [
        py_exe,
        str(project_root / "run_pipeline.py"),
        "--config",
        str(config_path),
    ]

    log.info("Memulai auto-retrain: %s", " ".join(cmd))
    print(f"[auto_retrain] Memulai: {' '.join(cmd)}", flush=True)

    if telegram_notify_fn:
        telegram_notify_fn(
            "🔄 *Auto-Retrain Dimulai*\n"
            "Alasan: HIGH DRIFT terdeteksi\n"
            f"Waktu: {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
            "Bot tetap aktif — sinyal ditahan selama retrain."
        )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(project_root),
        )

        if result.returncode == 0:
            run_dir = _parse_run_dir_from_stdout(
                result.stdout or "",
                project_root,
                output_root,
            )
            reset_drift_after_retrain(logs_dir)
            write_restart_flag(project_root, run_dir)

            run_label = run_dir.as_posix() if run_dir else "unknown"
            log.info("Auto-retrain selesai | run_dir=%s", run_label)
            print(f"[auto_retrain] Selesai | run_dir={run_label}", flush=True)

            if telegram_notify_fn:
                telegram_notify_fn(
                    f"✅ *Auto-Retrain Selesai*\n"
                    f"Run: `{run_dir.name if run_dir else 'unknown'}`\n"
                    "Merestart service dengan model baru..."
                )

            return True, run_label

        error_msg = (result.stderr or result.stdout or "")[-500:] or "Unknown error"
        log.error("Auto-retrain gagal (code=%s): %s", result.returncode, error_msg)
        print(f"[auto_retrain] Gagal: {error_msg[:300]}", flush=True)

        if telegram_notify_fn:
            telegram_notify_fn(
                f"❌ *Auto-Retrain Gagal*\n"
                f"Error: {error_msg[:200]}\n"
                "Jalankan manual: `python run_pipeline.py`"
            )

        return False, error_msg

    except subprocess.TimeoutExpired:
        log.error("Auto-retrain timeout (>%ss)", timeout_seconds)
        print("[auto_retrain] Timeout (>1 jam)", flush=True)
        if telegram_notify_fn:
            telegram_notify_fn("❌ *Auto-Retrain Timeout* (>1 jam) — jalankan manual")
        return False, "timeout"

    except Exception as exc:
        log.error("Auto-retrain exception: %s", exc)
        print(f"[auto_retrain] Exception: {exc}", flush=True)
        if telegram_notify_fn:
            telegram_notify_fn(f"❌ *Auto-Retrain Error*\n{exc}")
        return False, str(exc)
