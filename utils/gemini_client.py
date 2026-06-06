"""Sentimen berita pair aktif via Google Gemini API (google-genai)."""

from __future__ import annotations

import json
import logging
import re
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.paths import project_root
from utils.telegram_notify import is_placeholder

log = logging.getLogger("gemini_client")

_DEFAULT_CACHE_FILE = "logs/gemini_cache.json"
_DEFAULT_CACHE_MAX_AGE_MINUTES = 60
_GEMINI_FAILURE_KEYWORDS = (
    "error",
    "429",
    "quota",
    "gagal",
    "tidak tersedia",
    "belum dikonfigurasi",
    "resource_exhausted",
)

GEMINI_PROMPT_TEMPLATE = """
Kamu adalah analis berita XAUUSD (Gold vs USD) profesional.

Berikut headlines berita terkini:
{headlines}

Analisis dampak berita-berita ini terhadap harga XAUUSD dalam {horizon_text}.

PENTING: Hanya pertimbangkan berita yang RELEVAN untuk gold/XAUUSD:
- Relevan: Fed/suku bunga, inflasi, DXY/dollar strength, geopolitik, resesi, safe-haven demand,
  data ekonomi AS (NFP, CPI, GDP), kebijakan bank sentral global, COT gold positioning
- TIDAK relevan: crypto, saham individual, komoditas non-gold, berita perusahaan spesifik

Berikan skor sentimen dari -2 hingga +2:
+2 = Sangat Bullish gold (berita sangat mendukung kenaikan harga gold)
+1 = Bullish gold (berita cenderung mendukung kenaikan)
 0 = Netral (tidak ada dampak jelas, atau berita tidak relevan)
-1 = Bearish gold (berita cenderung menekan harga gold)
-2 = Sangat Bearish gold (berita sangat menekan harga gold, mis: Fed hawkish + DXY kuat)

Juga berikan:
- confidence: 0.0-1.0 (seberapa yakin kamu dengan skor ini)
- key_drivers: list maksimal 3 berita paling berpengaruh (kosong jika tidak ada yang relevan)
- irrelevant_count: jumlah berita yang tidak relevan untuk XAUUSD

Response HANYA dalam format JSON:
{{
  "score": <-2 to +2 integer>,
  "confidence": <0.0 to 1.0>,
  "key_drivers": ["berita 1", "berita 2"],
  "irrelevant_count": <integer>,
  "reasoning": "<1-2 kalimat singkat>"
}}
"""


def _clip_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _clip_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _parse_legacy_sentiment(raw: str) -> Tuple[int, str, float]:
    """Fallback parser jika JSON Gemini gagal."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            score = _clip_int(data.get("score", 0), -2, 2)
            conf = _clip_float(data.get("confidence", 0.5), 0.0, 1.0)
            note = str(data.get("reasoning") or data.get("summary", raw[:200]))[:240]
            return score, note, conf
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    low = raw.lower()
    if "bearish" in low or "bear" in low:
        return -1, raw[:200], 0.5
    if "bullish" in low or "bull" in low:
        return 1, raw[:200], 0.5
    return 0, raw[:200], 0.5


def _parse_gemini_json(raw: str) -> Tuple[int, str, float]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return _parse_legacy_sentiment(raw)
    try:
        data = json.loads(m.group())
        score = _clip_int(data.get("score", 0), -2, 2)
        confidence = _clip_float(data.get("confidence", 0.5), 0.0, 1.0)
        key_drivers = data.get("key_drivers", [])
        irrelevant_count = int(data.get("irrelevant_count", 0))
        reasoning = str(data.get("reasoning", ""))

        note = reasoning[:180]
        if isinstance(key_drivers, list) and key_drivers:
            drivers = "; ".join(str(x)[:80] for x in key_drivers[:2])
            note = f"{note} | Drivers: {drivers}"[:240]
        if irrelevant_count > 0:
            note = f"{note} | {irrelevant_count} berita tidak relevan diabaikan"[:240]
        return score, note.strip(), confidence
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return _parse_legacy_sentiment(raw)


def _call_genai_sdk(api_key: str, model: str, prompt: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key.strip())
    resp = client.models.generate_content(model=model, contents=prompt)
    return (resp.text or "").strip()


def _call_legacy_sdk(api_key: str, model: str, prompt: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key.strip())
    m = genai.GenerativeModel(model)
    resp = m.generate_content(prompt)
    return (resp.text or "").strip()


def gemini_news_sentiment(
    headlines: List[str],
    api_key: str,
    *,
    symbol: str = "XAUUSD",
    horizon_text: str = "1 jam ke depan (1 bar H1)",
    model: str,
    fallback_models: List[str] | None = None,
) -> Tuple[int, str, float]:
    """
    Skor sentimen -2..+2 untuk gold/XAUUSD.
    Return: (score, note, confidence).
    """
    if is_placeholder(api_key):
        return 0, "API key Gemini belum dikonfigurasi.", 0.0

    text_block = "\n".join(f"- {h}" for h in headlines[:12]) or "- (tidak ada headline)"
    prompt = textwrap.dedent(
        GEMINI_PROMPT_TEMPLATE.format(headlines=text_block, horizon_text=horizon_text)
    ).strip()

    chain = [model] + [m for m in (fallback_models or []) if m != model]
    errors: List[str] = []

    for m in chain:
        try:
            try:
                raw = _call_genai_sdk(api_key, m, prompt)
            except ImportError:
                raw = _call_legacy_sdk(api_key, m, prompt)
            score, note, conf = _parse_gemini_json(raw)
            return score, note, conf
        except Exception as exc:
            msg = str(exc)
            errors.append(f"{m}: {msg[:120]}")
            if "404" in msg or "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                continue
            return 0, f"Gemini error ({m}): {msg[:180]}", 0.0

    return 0, f"Gemini tidak tersedia ({'; '.join(errors[:2])})", 0.0


def _gemini_cfg(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not cfg:
        return {}
    s9 = cfg.get("stage_9", {})
    g = s9.get("gemini", {}) if isinstance(s9.get("gemini", {}), dict) else {}
    return g


def _cache_path(cfg: Optional[Dict[str, Any]] = None) -> Path:
    rel = str(_gemini_cfg(cfg).get("cache_file", _DEFAULT_CACHE_FILE))
    path = Path(rel)
    if not path.is_absolute():
        path = project_root() / path
    return path


def _cache_max_age_minutes(cfg: Optional[Dict[str, Any]] = None) -> float:
    return float(_gemini_cfg(cfg).get("cache_max_age_minutes", _DEFAULT_CACHE_MAX_AGE_MINUTES))


def _gemini_call_failed(score: int, note: str, confidence: float) -> bool:
    low = str(note).lower()
    if any(k in low for k in _GEMINI_FAILURE_KEYWORDS):
        return True
    if confidence <= 0.0 and score == 0 and "netral" not in low:
        return True
    return False


def _load_gemini_cache(cfg: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Load cache Gemini terakhir yang berhasil."""
    cache_file = _cache_path(cfg)
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        cached_at = datetime.fromisoformat(str(data["cached_at"]))
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        age_minutes = (datetime.now(timezone.utc) - cached_at.astimezone(timezone.utc)).total_seconds() / 60
        if age_minutes <= _cache_max_age_minutes(cfg):
            return data
        log.info("Gemini cache expired (%.0f menit lalu)", age_minutes)
        return None
    except Exception as exc:
        log.warning("Gagal load Gemini cache: %s", exc)
        return None


def _save_gemini_cache(
    score: int,
    note: str,
    confidence: float,
    cfg: Optional[Dict[str, Any]] = None,
) -> None:
    """Simpan hasil Gemini yang berhasil ke cache."""
    try:
        cache_file = _cache_path(cfg)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "score": int(score),
            "note": str(note),
            "confidence": float(confidence),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Gagal simpan Gemini cache: %s", exc)


def gemini_news_sentiment_with_retry(
    headlines: List[str],
    api_key: str,
    *,
    symbol: str = "XAUUSD",
    horizon_text: str = "1 jam ke depan (1 bar H1)",
    model: str = "gemini-2.5-flash",
    fallback_models: Optional[List[str]] = None,
    max_retries: int = 3,
    retry_delay_seconds: float = 5.0,
    cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[int, str, float]:
    """
    Wrapper gemini_news_sentiment dengan retry + cache fallback.

    Urutan:
    1. Coba Gemini API (max_retries, jeda retry_delay_seconds)
    2. Jika semua gagal → cache (max cache_max_age_minutes)
    3. Jika cache tidak ada/expired → fallback keyword
    """
    gemini_cfg = _gemini_cfg(cfg)
    max_retries = int(gemini_cfg.get("max_retries", max_retries))
    retry_delay_seconds = float(gemini_cfg.get("retry_delay_seconds", retry_delay_seconds))

    last_error: Optional[Exception] = None
    last_note = ""

    for attempt in range(1, max_retries + 1):
        try:
            score, note, confidence = gemini_news_sentiment(
                headlines=headlines,
                api_key=api_key,
                symbol=symbol,
                horizon_text=horizon_text,
                model=model,
                fallback_models=fallback_models,
            )
            if _gemini_call_failed(score, note, confidence):
                last_note = note
                log.warning(
                    "Gemini response tidak valid attempt %d/%d: %s",
                    attempt,
                    max_retries,
                    note[:120],
                )
                if attempt < max_retries:
                    log.info("Retry dalam %.0f detik...", retry_delay_seconds)
                    time.sleep(retry_delay_seconds)
                continue

            _save_gemini_cache(score, note, confidence, cfg)
            log.info(
                "Gemini berhasil (attempt %d/%d) | score=%d | conf=%.2f",
                attempt,
                max_retries,
                score,
                confidence,
            )
            return score, note, confidence
        except Exception as exc:
            last_error = exc
            log.warning("Gemini gagal attempt %d/%d: %s", attempt, max_retries, exc)
            if attempt < max_retries:
                log.info("Retry dalam %.0f detik...", retry_delay_seconds)
                time.sleep(retry_delay_seconds)

    log.warning("Semua retry Gemini gagal: %s | last_note=%s", last_error, last_note[:120])

    cache = _load_gemini_cache(cfg)
    if cache is not None:
        cached_at = datetime.fromisoformat(str(cache["cached_at"]))
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)
        age_minutes = (datetime.now(timezone.utc) - cached_at.astimezone(timezone.utc)).total_seconds() / 60
        note_cached = (
            f"{cache['note']} "
            f"[cache {age_minutes:.0f} menit lalu — Gemini tidak tersedia]"
        )
        log.info("Pakai Gemini cache (%.0f menit lalu) | score=%s", age_minutes, cache["score"])
        return int(cache["score"]), note_cached[:240], float(cache["confidence"])

    log.warning("Tidak ada cache Gemini — fallback ke keyword sentiment")
    from utils.news_fetch import simple_headline_sentiment

    kw_score, kw_note = simple_headline_sentiment(headlines)
    return kw_score, f"{kw_note} (fallback keyword — Gemini tidak tersedia)"[:240], 0.3
