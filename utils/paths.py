from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


ACTIVE_RUN_POINTER = "active_run.txt"


def active_run_pointer_path(output_root: str = "artifacts") -> Path:
    return project_root() / output_root / ACTIVE_RUN_POINTER


def write_active_run_pointer(run_dir: Path, *, output_root: str = "artifacts") -> None:
    """Catat run yang dipakai Stage 9 / bot (diperbarui setiap run_pipeline selesai)."""
    run_dir = Path(run_dir)
    ptr = active_run_pointer_path(output_root)
    ptr.parent.mkdir(parents=True, exist_ok=True)
    ptr.write_text(run_dir.name + "\n", encoding="utf-8")


def read_active_run_pointer(output_root: str = "artifacts") -> Path | None:
    ptr = active_run_pointer_path(output_root)
    if not ptr.is_file():
        return None
    name = ptr.read_text(encoding="utf-8").strip()
    if not name:
        return None
    run_dir = project_root() / output_root / name
    return run_dir if run_dir.is_dir() else None


def latest_pipeline_run_dir(output_root: str = "artifacts") -> Path | None:
    """Folder run terbaru yang punya model Stage 5."""
    root = project_root() / output_root
    if not root.is_dir():
        return None
    candidates = sorted(
        (p for p in root.glob("run_*") if (p / "stage_5" / "xgb_model.joblib").is_file()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def resolve_pipeline_run_dir(
    preferred: Path | None = None,
    *,
    output_root: str = "artifacts",
) -> Path:
    """
    Pilih folder run yang valid (ada xgb_model.joblib).
    Urutan: active_run.txt → preferred (jika model ada) → run terbaru dengan model.
    """
    model_rel = Path("stage_5") / "xgb_model.joblib"
    preferred = Path(preferred) if preferred is not None else None

    def _has_model(rd: Path) -> bool:
        return (rd / model_rel).is_file()

    pointed = read_active_run_pointer(output_root)
    if pointed is not None and _has_model(pointed):
        if preferred is not None and preferred != pointed and not _has_model(preferred):
            print(
                f"[paths] Run lama {preferred.name} tidak valid; "
                f"pakai {pointed.name} (active_run.txt)",
                flush=True,
            )
        return pointed

    if preferred is not None and _has_model(preferred):
        return preferred

    latest = latest_pipeline_run_dir(output_root)
    if latest is not None:
        if preferred is not None and preferred != latest:
            print(
                f"[paths] Model tidak ada di {preferred.name}; "
                f"menggunakan {latest.name}",
                flush=True,
            )
        return latest

    if preferred is not None:
        missing = preferred / model_rel
        raise FileNotFoundError(
            f"Model tidak ditemukan: {missing}. "
            "Jalankan: python run_pipeline.py lalu restart bot "
            "(scripts\\restart_stage9_service.bat)."
        )
    raise FileNotFoundError(
        "Tidak ada run dengan xgb_model.joblib. Jalankan: python run_pipeline.py"
    )
