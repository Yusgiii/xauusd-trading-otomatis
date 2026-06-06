# noqa: D100
"""Uji koneksi Gemini API."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.config_loader import load_pipeline_config
from utils.gemini_client import gemini_news_sentiment
from utils.paths import project_root


def main() -> None:
    cfg = load_pipeline_config(project_root() / "configs" / "pipeline.yaml")
    key = cfg["risk"]["gemini_api_key"]
    s9 = cfg.get("stage_9", {})
    headlines = [
        "Japan GDP grows 2.1% beating expectations",
        "Bank of England holds rates steady",
    ]
    score, note, conf = gemini_news_sentiment(
        headlines,
        key,
        model=str(s9.get("gemini_model", "gemini-2.5-flash")),
        fallback_models=list(s9.get("gemini_fallback_models") or []),
    )
    print("score:", score)
    print("note:", note)
    print("confidence:", conf)


if __name__ == "__main__":
    main()
