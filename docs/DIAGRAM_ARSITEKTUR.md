# Diagram Arsitektur Sistem

Dokumen ini merangkum alur data dan komponen utama sistem trading XAUUSD M15.
Run model aktif: `artifacts/run_20260602_005304` (lihat `artifacts/active_run.txt`).

---

## Gambaran Satu Halaman

```text
                    ┌─────────────────────────────────────────┐
                    │         OFFLINE — TRAINING              │
                    │  run_pipeline.py (orchestrator)         │
                    └─────────────────────────────────────────┘
                                        │
    MetaTrader 5 / CSV ──► Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4
                                        │
                                        ▼
                              Stage 5 (XGBoost + Meta)
                                        │
                                        ▼
                              Stage 6 (Decision Logic)
                                        │
                                        ▼
                              Artifacts (model, threshold, registry)

                    ┌─────────────────────────────────────────┐
                    │         ONLINE — LIVE SERVICE           │
                    │  scripts/stage9_service.py              │
                    └─────────────────────────────────────────┘
                                        │
    MT5 real-time ──► Fitur live ──► Inferensi ──► Filter ──► Telegram
                                        │
                          User "ya" ──► MT5 order ──► Position monitor
                                        │
                                        ▼
                              Notifikasi TP/SL + log performa
```

---

## Pipeline Training (Offline)

```text
[Data MT5 / data/xauusd_m15.csv]
        │
        ▼
┌───────────────────┐
│ Stage 1: Data     │  Cleaning, gap, spread cap, vol regime, weekend filter
│ stage_1_data.py   │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Stage 2: Label    │  UP / DOWN / FLAT + tp_sl_outcome (TP_FIRST/SL_FIRST)
│ stage_2_labeling  │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Stage 3: Features │  11 fitur aktif (shift anti-lookahead)
│ stage_3_features  │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Stage 4: Folds    │  Purged walk-forward (5 split, embargo ≥ horizon)
│ stage_4_validation│
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Stage 5: Train    │  Optuna → XGBoost 3-kelas + meta-filter + kalibrasi
│ stage_5_training  │
└─────────┬─────────┘
          ▼
┌───────────────────┐
│ Stage 6: Decision │  conf_up / conf_down / abstain_tau + signal rate
│ stage_6_ensemble  │
└─────────┬─────────┘
          ▼
[artifacts/run_YYYYMMDD_HHMMSS/]
  ├── stage_5/xgb_model.joblib
  ├── stage_5/meta_model.joblib
  ├── stage_5/feature_registry.json
  ├── stage_5/threshold_config.json
  └── stage_6/stage_6_metadata.json
```

**Entry point:** `python run_pipeline.py`  
**Config:** `configs/pipeline.yaml`  
**Baris training terakhir:** 41.546 baris (`keep_for_training = 1`)

---

## Sistem Live (Online)

```text
[Windows login / manual start]
        │
        ▼
scripts/startup_service.bat  (opsional, Task Scheduler)
        │
        ▼
scripts/stage9_service.py --latest-run
        │
        ├── Thread: Telegram polling (utils/telegram_bot.py)
        │      /analisa  /status  /trades  /akun  + konfirmasi ya/tidak
        │
        └── Thread: Moment alert loop (bar M15)
               │
               ├─ Pasar tutup? → sleep (utils/trading_hours.py)
               ├─ Posisi aktif? → tunggu monitor MT5
               ├─ Bar baru + delay 120 detik → run_daily_report()
               │       │
               │       ├─ MT5 bars → fitur (utils/xauusd_features.py)
               │       ├─ XGBoost predict_proba
               │       ├─ Meta-filter (wajib)
               │       ├─ Sentiment Gemini (utils/gemini_client.py)
               │       ├─ Decision + trade plan (SL/TP ATR)
               │       └─ Kirim Telegram (sinyal atau NO TRADE)
               │
               └─ Auto-retrain? (utils/retrain_scheduler.py)

[User balas "ya"]
        │
        ▼
utils/mt5_execution.py → place_market_order_from_plan()
        │
        ▼
utils/position_monitor.py → monitor posisi MT5 sampai close
        │
        ▼
Notifikasi close (P/L nyata, ticket, balance) + update live_trade_log.csv
```

---

## Auto-Retrain Loop

```text
[Inferensi live / pipeline selesai]
        │
        ▼
logs/runtime_risk.json  ← PSI drift (≥2 fitur PSI>0.2 → HIGH_DRIFT)
        │
        ▼
should_retrain()?  ──no──► lanjut trading
        │
       yes (HIGH_DRIFT / CRITICAL, interval ≥ 6 jam, bukan weekend)
        │
        ▼
run_pipeline.py (subprocess, timeout 3600s)
        │
        ▼
artifacts/run_baru/ + active_run.txt diperbarui
        │
        ▼
Service restart (spawn instance baru / flag needs_restart.flag)
```

---

## Komponen Pendukung

| Komponen | File | Peran |
|----------|------|--------|
| Orchestrator | `run_pipeline.py` | Menjalankan Stage 1–6 berurutan |
| Config | `configs/pipeline.yaml` | Semua hyperparameter |
| Live inference | `stage_9_live_demo.py` | `run_daily_report`, trade plan, drift |
| Service daemon | `scripts/stage9_service.py` | Bot + moment alert + retrain hook |
| Eksekusi | `utils/mt5_execution.py` | Order market + SL/TP |
| Monitor posisi | `utils/position_monitor.py` | Deteksi close dari MT5 (bukan simulasi harga) |
| Telegram | `utils/telegram_bot.py` | Command & konfirmasi user |
| Log trade | `utils/live_tracker.py` | `logs/live_trade_log.csv` |

---

## Batasan Arsitektur (penting untuk skripsi)

1. **Human-in-the-loop:** Order hanya masuk MT5 setelah user membalas `ya` (kecuali konfigurasi eksekusi dimatikan).
2. **Satu instance service:** Lock file `logs/stage9_service.lock` mencegah duplikasi bot.
3. **Model terikat run:** Inferensi memuat artifact dari `artifacts/active_run.txt`, bukan hot-reload otomatis tanpa restart.
