from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from utils.paths import project_root


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_pipeline_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    secrets_path = path.parent / "pipeline.secrets.yaml"
    if secrets_path.is_file():
        with secrets_path.open("r", encoding="utf-8") as f:
            secrets = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, secrets)
    else:
        alt = project_root() / "configs" / "pipeline.secrets.yaml"
        if alt.is_file() and alt != secrets_path:
            with alt.open("r", encoding="utf-8") as f:
                secrets = yaml.safe_load(f) or {}
            cfg = _deep_merge(cfg, secrets)

    return cfg
