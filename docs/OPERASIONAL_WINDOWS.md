# Operasional Windows — XAUUSD H1 Pipeline

## Prasyarat

1. **MetaTrader 5** terpasang, login, simbol **XAUUSD** di Market Watch (timeframe **H1**).
2. **Python 3.11** + dependensi: `pip install -r requirements.txt`
3. **`configs/pipeline.secrets.yaml`** terisi:
   - `telegram_token`, `telegram_chat_id`
   - `newsapi_key`
   - `gemini_api_key`
4. Zona waktu Windows disarankan **(UTC+07:00) Jakarta** agar log konsisten.

---

## Autostart saat login (sekali)

```powershell
cd "D:\trading\trading otomatis 5"
powershell -ExecutionPolicy Bypass -File .\scripts\setup_autostart.ps1
```

Menambah shortcut di folder Startup untuk:
- MetaTrader 5 (jika terdeteksi)
- Bot Telegram (`run_stage9_service_hidden.vbs`)

Hapus autostart:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_autostart.ps1 -Uninstall
```

---

## Task Scheduler (sekali)

Jalankan **PowerShell sebagai Administrator**:

```powershell
cd "D:\trading\trading otomatis 5"
powershell -ExecutionPolicy Bypass -File .\scripts\register_windows_tasks.ps1
```

Script ini:
- Menghapus task lama (`GBPJPY_*`, `XAUUSD_H1_Task` lama jika ada)
- Mendaftarkan **`XAUUSD_H1_Stage9_Bot`** — jalan saat login Windows
- Mencoba **start langsung** setelah register

Cek di `taskschd.msc` → Task Scheduler Library → **`XAUUSD_H1_Stage9_Bot`**

| Task | Trigger | Fungsi |
|------|---------|--------|
| `XAUUSD_H1_Stage9_Bot` | At log on | Bot Telegram on-demand (`/analisa`, `/pairs`, `/akurasi`) |

Log:
- `logs/stage9_service.log`
- `logs/stage9_service.lock` (single instance)

---

## Apa yang dijalankan otomatis?

1. **Bot polling** aktif setelah login dan menunggu perintah user.
2. Analisa dijalankan **hanya** saat command dipanggil (`/analisa ...`).

**Catatan:** Tidak ada lagi scheduler laporan periodik per jam.

---

## Perintah manual

```powershell
cd "D:\trading\trading otomatis 5"

# Training penuh (offline)
python run_pipeline.py

# Bot on-demand (terminal terlihat)
scripts\run_stage9_service.bat

# Satu laporan sekarang
python stage_9_live_demo.py --latest-run

# Tes
python stage_9_live_demo.py --test-telegram
python scripts/preflight_check.py
```

Telegram: `/analisa`, `/analisis`, `/pairs`, `/akurasi`, `/status`, `/help`

---

## Restart service (jika perlu)

```powershell
scripts\restart_stage9_service.bat
```

Atau hentikan proses lama lalu jalankan ulang:

```powershell
wmic process where "CommandLine like '%stage9_service.py%'" call terminate
scripts\run_stage9_service_task.bat
```

---

## Troubleshooting

| Gejala | Solusi |
|--------|--------|
| `/analisa` tidak dibalas | Pastikan service jalan; cek `logs/stage9_service.log`; satu instance saja |
| `getaddrinfo failed` / timeout Telegram | Cek internet/DNS; service tetap hidup setelah patch error handling |
| MT5 error | MT5 terbuka, login, simbol aktif di Market Watch; cek `/pairs` untuk daftar simbol broker |
| Model tidak ditemukan | Jalankan `python run_pipeline.py` dulu |
| Dua proses stage9 | Hentikan duplikat dengan `restart_stage9_service.bat` |

---

## Keamanan

- Jangan commit `pipeline.secrets.yaml`.
- Rotate token jika pernah terbongkar.
