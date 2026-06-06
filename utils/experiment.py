from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def append_experiment_log(root: Path, record: Dict[str, Any]) -> None:
    logf = root / "experiments" / "experiment_log.jsonl"
    logf.parent.mkdir(parents=True, exist_ok=True)
    rec = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with logf.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
