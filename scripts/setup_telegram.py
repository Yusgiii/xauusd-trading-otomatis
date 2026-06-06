# noqa: D100
"""
Setup & uji Telegram untuk Stage 9.

Langkah:
  1. Buat bot via @BotFather di Telegram → salin token.
  2. Kirim /start ke bot Anda dari akun/chanel tujuan.
  3. Jalankan skrip ini.

Perintah:
  python scripts/setup_telegram.py
  python scripts/setup_telegram.py --test
  python scripts/setup_telegram.py --write-secrets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from utils.paths import project_root
from utils.telegram_notify import (
    discover_chat_ids,
    is_placeholder,
    send_telegram_message,
    telegram_get_me,
)

SECRETS_PATH = project_root() / "configs" / "pipeline.secrets.yaml"
EXAMPLE_PATH = project_root() / "configs" / "pipeline.secrets.yaml.example"


def _prompt(label: str, default: str = "") -> str:
    hint = f" [{default}]" if default else ""
    val = input(f"{label}{hint}: ").strip()
    return val or default


def cmd_discover(token: str) -> None:
    print("\n--- Mencari chat_id (kirim /start ke bot Anda dulu) ---")
    data = discover_chat_ids(token)
    if not data.get("ok"):
        print("Gagal getUpdates:", data)
        return
    chats = data.get("chats") or []
    if not chats:
        print("Belum ada pesan masuk. Buka Telegram, cari bot Anda, ketik /start, lalu jalankan lagi.")
        return
    for i, c in enumerate(chats, 1):
        print(
            f"  {i}. chat_id={c['chat_id']} | {c.get('type')} | "
            f"{c.get('title')} | last: {c.get('last_text', '')!r}"
        )


def cmd_test(token: str, chat_id: str) -> bool:
    print("\n--- Mengirim pesan uji ---")
    text = (
        "*XAUUSD Pipeline — tes Telegram*\n\n"
        "Jika Anda membaca ini, Stage 9 siap dikoneksikan.\n"
        "_Pesan otomatis dari setup_telegram.py_"
    )
    res = send_telegram_message(token, chat_id, text)
    if res.get("ok"):
        print("OK — pesan terkirim.")
        if res.get("fallback_plain"):
            print("(Markdown gagal; pesan dikirim sebagai teks biasa.)")
        return True
    print("Gagal:", res)
    return False


def cmd_write_secrets(token: str, chat_id: str, gemini_key: str = "") -> None:
    payload = {
        "risk": {
            "telegram_token": token,
            "telegram_chat_id": str(chat_id),
        }
    }
    if gemini_key:
        payload["risk"]["gemini_api_key"] = gemini_key

    SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SECRETS_PATH.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, allow_unicode=True)
    print(f"Disimpan → {SECRETS_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Setup Telegram Stage 9")
    ap.add_argument("--token", type=str, default="", help="Bot token dari @BotFather")
    ap.add_argument("--chat-id", type=str, default="", help="Chat ID tujuan")
    ap.add_argument("--discover", action="store_true", help="Cari chat_id dari getUpdates")
    ap.add_argument("--test", action="store_true", help="Kirim pesan uji")
    ap.add_argument("--write-secrets", action="store_true", help="Simpan ke pipeline.secrets.yaml")
    args = ap.parse_args()

    print("=== Setup Telegram — XAUUSD Stage 9 ===\n")
    if not EXAMPLE_PATH.is_file():
        print(f"Contoh config: {EXAMPLE_PATH}")

    token = args.token.strip()
    if not token and SECRETS_PATH.is_file():
        with SECRETS_PATH.open("r", encoding="utf-8") as f:
            sec = yaml.safe_load(f) or {}
        token = str((sec.get("risk") or {}).get("telegram_token", "")).strip()

    if is_placeholder(token):
        print("1) Buka Telegram → @BotFather → /newbot → salin token.\n")
        token = _prompt("Bot token")

    me = telegram_get_me(token)
    if not me.get("ok"):
        print("Token tidak valid:", me)
        sys.exit(1)
    bot = me["result"]
    print(f"Bot OK: @{bot.get('username')} ({bot.get('first_name')})")

    if args.discover or not args.chat_id:
        cmd_discover(token)

    chat_id = args.chat_id.strip()
    if is_placeholder(chat_id):
        chat_id = _prompt("chat_id (dari daftar di atas)")

    if args.test or args.write_secrets:
        if not cmd_test(token, chat_id):
            sys.exit(1)

    if args.write_secrets:
        gemini = _prompt("gemini_api_key (opsional, Enter lewati)", "")
        cmd_write_secrets(token, chat_id, gemini)
        print("\nBerikutnya:")
        print(f"  python stage_9_live_demo.py --run-dir artifacts\\run_20260520_014854 --once")
        return

    print("\nPerintah berguna:")
    print(f"  python scripts/setup_telegram.py --token <TOKEN> --discover")
    print(f"  python scripts/setup_telegram.py --token <TOKEN> --chat-id <ID> --test --write-secrets")


if __name__ == "__main__":
    main()
