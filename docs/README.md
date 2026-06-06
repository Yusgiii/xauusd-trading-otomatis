# Dokumentasi Sistem — XAUUSD H1 Pipeline

Dokumen ini menjelaskan sistem trading ML **XAUUSD timeframe H1** (prediksi arah per jam) yang sedang aktif di repositori ini.

## Daftar dokumen

| File | Isi |
|------|-----|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Arsitektur teknis, alur data, modul, artefak |
| [OPERASIONAL_WINDOWS.md](OPERASIONAL_WINDOWS.md) | Instalasi Windows, Task Scheduler, autostart, troubleshooting |
| [GEMINI_BRIEFING_SKRIPSI.md](GEMINI_BRIEFING_SKRIPSI.md) | Konteks untuk penulisan skripsi / briefing LLM |
| [contoh_laporan_harian.md](contoh_laporan_harian.md) | Contoh format laporan Telegram (H1) |

## Ringkasan singkat

| Aspek | Nilai |
|--------|--------|
| Instrumen | **XAUUSD** |
| Timeframe | **H1** (1 candle = 1 jam) |
| Horizon prediksi | **1 bar ke depan** |
| Kelas | **FLAT (0)**, **UP (1)**, **DOWN (2)** |
| Model | **XGBoost** multiclass |
| Validasi | Purged walk-forward + holdout 15% + Optuna (48 trial) |
| Live | MT5 → fitur → XGBoost → berita → Gemini → trade plan ATR → Telegram |
| Trigger live | **On-demand** via `/analisa` (tanpa scheduler periodik) |
| Evaluasi live | `logs/prediction_log.csv` + `/akurasi` |

## Entry point

```bash
# Training + artefak
python run_pipeline.py

# Bot Telegram (on-demand command)
python scripts/stage9_service.py --latest-run

# Satu laporan manual
python stage_9_live_demo.py --latest-run
```

## Konfigurasi

- Publik: `configs/pipeline.yaml`
- Rahasia: `configs/pipeline.secrets.yaml` (jangan di-commit)

## Artefak run

- `artifacts/run_<timestamp>/` — output tiap training
- `artifacts/active_run.txt` — pointer run model aktif untuk Stage 9

## Catatan penting

- Sistem **tidak** mengeksekusi order di broker.
- Output `/analisa` sudah memuat rencana `entry/SL/TP/RR` berbasis ATR sebagai decision support.
- Spread/slippage eksekusi riil tidak dimodelkan penuh di backtest terpisah (stage 7–8 tidak ada di pipeline utama).
- Akurasi tinggi pada holdout tidak sama dengan profit trading riil; gunakan prediction log untuk evaluasi operasional.

*Terakhir diperbarui: 2026-05-28*
