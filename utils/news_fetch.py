"""Ambil headline berita untuk Stage 9 (NewsAPI + RSS cadangan)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import requests

from utils.telegram_notify import is_placeholder


def fetch_newsapi_headlines(
    api_key: str,
    *,
    query: str,
    max_items: int = 8,
    language: str = "en",
    timeout: int = 25,
) -> Tuple[List[str], str]:
    """
    NewsAPI v2 everything. Return (headlines, source_label).
    https://newsapi.org/docs/endpoints/everything
    """
    if is_placeholder(api_key):
        return [], "newsapi_skipped"

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": min(max(int(max_items), 1), 20),
        "apiKey": api_key.strip(),
    }
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        data = resp.json()
        if resp.status_code != 200 or data.get("status") != "ok":
            err = data.get("message") or data.get("code") or resp.text[:200]
            return [f"(NewsAPI: {err})"], "newsapi_error"

        titles: List[str] = []
        for art in data.get("articles") or []:
            title = (art.get("title") or "").strip()
            if title and title != "[Removed]":
                src = (art.get("source") or {}).get("name") or ""
                line = f"{title}" + (f" — {src}" if src else "")
                titles.append(re.sub(r"\s+", " ", line))
            if len(titles) >= max_items:
                break
        return titles[:max_items], "newsapi"
    except Exception as exc:
        return [f"(NewsAPI request gagal: {exc})"], "newsapi_error"


def fetch_rss_headlines(rss_urls: List[str], max_items: int) -> List[str]:
    try:
        import feedparser
    except ImportError:
        return []

    headlines: List[str] = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            per = max(max_items // max(1, len(rss_urls)), 2)
            for entry in feed.entries[:per]:
                title = getattr(entry, "title", "") or ""
                title = re.sub(r"\s+", " ", title).strip()
                if title:
                    headlines.append(title)
        except Exception:
            continue
        if len(headlines) >= max_items:
            break
    return headlines[:max_items]


def fetch_all_headlines(cfg: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """Gabungkan NewsAPI (utama) + RSS (cadangan) sesuai stage_9.news_source."""
    s9 = cfg.get("stage_9", {})
    risk = cfg.get("risk", {})
    max_items = int(s9.get("news_max_headlines", 8))
    source = str(s9.get("news_source", "both")).strip().lower()
    meta: Dict[str, Any] = {"news_source": source}

    headlines: List[str] = []
    api_key = str(risk.get("newsapi_key", ""))

    if source in ("newsapi", "both"):
        q = str(
            s9.get(
                "newsapi_query",
                'XAUUSD OR "gold price" OR "spot gold" OR bullion OR "Fed" OR FOMC OR "US dollar"',
            )
        )
        lang = str(s9.get("newsapi_language", "en"))
        items, label = fetch_newsapi_headlines(
            api_key, query=q, max_items=max_items, language=lang
        )
        meta["newsapi"] = label
        headlines.extend(items)

    if source in ("rss", "both") and len(headlines) < max_items:
        rss = fetch_rss_headlines(list(s9.get("news_rss_urls", [])), max_items - len(headlines))
        meta["rss_count"] = len(rss)
        headlines.extend(rss)

    # Dedupe ringkas
    seen: set[str] = set()
    unique: List[str] = []
    for h in headlines:
        key = h.lower()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(h)

    if not unique:
        unique = ["(Tidak ada headline — periksa newsapi_key atau koneksi)"]

    return unique[:max_items], meta


XAUUSD_RELEVANT_KEYWORDS = [
    "fed",
    "federal reserve",
    "interest rate",
    "suku bunga",
    "inflation",
    "inflasi",
    "cpi",
    "pce",
    "gdp",
    "nfp",
    "payroll",
    "unemployment",
    "dollar",
    "dxy",
    "usd",
    "greenback",
    "gold",
    "emas",
    "xauusd",
    "precious metal",
    "bullion",
    "geopolit",
    "war",
    "perang",
    "conflict",
    "konflik",
    "sanction",
    "sanksi",
    "iran",
    "russia",
    "ukraine",
    "china",
    "safe haven",
    "risk off",
    "risk-off",
    "recession",
    "resesi",
    "crisis",
    "ecb",
    "boe",
    "boj",
    "pboc",
    "central bank",
    "bank sentral",
    "fomc",
    "treasury",
]

XAUUSD_IRRELEVANT_KEYWORDS = [
    "bitcoin",
    "btc",
    "crypto",
    "ethereum",
    "solana",
    "nft",
    "stock",
    "saham",
    "equity",
    "nasdaq",
    "s&p",
    "oil",
    "crude",
    "minyak",
    "natural gas",
]


def filter_headlines_for_xauusd(headlines: List[str]) -> Tuple[List[str], List[str]]:
    """
    Filter headlines: return (relevant, irrelevant).
    Relevant jika ada keyword relevan DAN tidak ada keyword tidak relevan.
    """
    relevant: List[str] = []
    irrelevant: List[str] = []
    for h in headlines:
        h_lower = h.lower()
        has_relevant = any(kw in h_lower for kw in XAUUSD_RELEVANT_KEYWORDS)
        has_irrelevant = any(kw in h_lower for kw in XAUUSD_IRRELEVANT_KEYWORDS)

        if has_irrelevant and not has_relevant:
            irrelevant.append(h)
        elif has_relevant:
            relevant.append(h)
        else:
            relevant.append(h)

    return relevant, irrelevant


def simple_headline_sentiment(headlines: List[str]) -> Tuple[int, str]:
    """Skor -1/0/1 dari kata kunci bila Gemini tidak dipakai."""
    text = " ".join(headlines).lower()
    bull = (
        "surge",
        "rally",
        "hawkish",
        "rate hike",
        "strong",
        "gains",
        "rose",
        "bullish",
        "gold rally",
        "xauusd bullish",
        "dollar weakness",
        "fed dovish",
    )
    bear = (
        "fall",
        "drop",
        "dovish",
        "rate cut",
        "weak",
        "slump",
        "decline",
        "bearish",
        "gold slump",
        "xauusd bearish",
        "dollar strength",
        "fed hawkish",
    )
    b = sum(1 for w in bull if w in text)
    s = sum(1 for w in bear if w in text)
    if b > s + 1:
        return 1, f"Sentimen keyword: bullish ({b} vs {s}) — tanpa Gemini"
    if s > b + 1:
        return -1, f"Sentimen keyword: bearish ({s} vs {b}) — tanpa Gemini"
    return 0, f"Sentimen keyword: netral ({b} bull / {s} bear) — tanpa Gemini"
