"""Kirim pesan ke Telegram Bot API."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import requests
from requests import RequestException


def is_placeholder(value: str) -> bool:
    v = (value or "").strip()
    return not v or v.startswith("YOUR_")


def escape_markdown_v1(text: str) -> str:
    """Escape karakter khusus Telegram Markdown (legacy)."""
    for ch in ("_", "*", "[", "`"):
        text = text.replace(ch, f"\\{ch}")
    return text


def send_telegram_message(
    token: str,
    chat_id: str,
    text: str,
    *,
    parse_mode: Optional[str] = "Markdown",
    timeout: int = 30,
) -> Dict[str, Any]:
    if is_placeholder(token):
        return {"ok": False, "error": "telegram_token belum dikonfigurasi"}
    if is_placeholder(chat_id):
        return {"ok": False, "error": "telegram_chat_id belum dikonfigurasi"}

    url = f"https://api.telegram.org/bot{token.strip()}/sendMessage"
    payload: Dict[str, Any] = {
        "chat_id": str(chat_id).strip(),
        "text": text,
        "disable_web_page_preview": True,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except RequestException as exc:
        return {"ok": False, "error": f"telegram_send_failed: {exc}"}
    try:
        data = resp.json()
    except Exception:
        data = {"raw": resp.text}

    if not resp.ok and parse_mode:
        plain = re.sub(r"[*_`\[\]]", "", text)
        try:
            resp2 = requests.post(
                url,
                json={
                    "chat_id": str(chat_id).strip(),
                    "text": plain,
                    "disable_web_page_preview": True,
                },
                timeout=timeout,
            )
        except RequestException as exc:
            return {
                "ok": False,
                "error": f"telegram_send_failed_fallback: {exc}",
                "markdown_error": data,
            }
        try:
            data2 = resp2.json()
        except Exception:
            data2 = {"raw": resp2.text}
        return {
            "ok": resp2.ok,
            "status": resp2.status_code,
            "response": data2,
            "fallback_plain": True,
            "markdown_error": data,
        }

    return {"ok": resp.ok, "status": resp.status_code, "response": data}


def telegram_set_commands(token: str, commands: List[Dict[str, str]]) -> Dict[str, Any]:
    """Daftarkan menu perintah di Telegram (BotFather-style)."""
    url = f"https://api.telegram.org/bot{token.strip()}/setMyCommands"
    resp = requests.post(url, json={"commands": commands}, timeout=20)
    return resp.json()


def telegram_get_me(token: str) -> Dict[str, Any]:
    url = f"https://api.telegram.org/bot{token.strip()}/getMe"
    resp = requests.get(url, timeout=20)
    return resp.json()


def telegram_get_updates(
    token: str,
    *,
    offset: Optional[int] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Long-polling getUpdates."""
    url = f"https://api.telegram.org/bot{token.strip()}/getUpdates"
    params: Dict[str, Any] = {"timeout": int(timeout)}
    if offset is not None:
        params["offset"] = int(offset)
    resp = requests.get(url, params=params, timeout=timeout + 10)
    return resp.json()


def discover_chat_ids(token: str) -> Dict[str, Any]:
    """Ambil chat_id dari pesan terakhir ke bot (user harus /start dulu)."""
    url = f"https://api.telegram.org/bot{token.strip()}/getUpdates"
    resp = requests.get(url, timeout=20)
    data = resp.json()
    chats: Dict[str, Dict[str, Any]] = {}
    if not data.get("ok"):
        return data
    for upd in data.get("result", []):
        msg = upd.get("message") or upd.get("channel_post") or {}
        chat = msg.get("chat") or {}
        cid = chat.get("id")
        if cid is None:
            continue
        key = str(cid)
        chats[key] = {
            "chat_id": key,
            "type": chat.get("type"),
            "title": chat.get("title") or chat.get("username") or chat.get("first_name"),
            "last_text": (msg.get("text") or "")[:80],
        }
    return {"ok": True, "chats": list(chats.values())}
