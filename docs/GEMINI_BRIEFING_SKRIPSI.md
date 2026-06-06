# Briefing Sistem XAUUSD H1 — untuk Penulisan Skripsi (Gemini / LLM)

**Versi dokumen:** 2026-05-25  
**Proyek:** `trading otomatis 5` — pipeline prediksi arah **XAUUSD timeframe H1** + bot Telegram  
**Instruksi:** Baca dokumen ini dan daftar file §12. Jangan minta API key; asumsikan rahasia ada di `configs/pipeline.secrets.yaml`.

---

## 1. Judul kerja (usulan)

**“Sistem Prediksi Arah Harga XAU/USD pada Timeframe H1 Berbasis XGBoost dengan Integrasi Sentimen Berita dan Notifikasi Telegram Berbasis Jadwal”**

Alternatif: *Sistem pendukung keputusan trading XAU/USD per jam (FLAT/UP/DOWN) menggunakan machine learning, validasi time-series, dan analisis sentimen berita via API.*

---

## 2. Ringkasan eksekutif

Sistem memproses data OHLCV **jam (H1)** pasangan **XAU/USD** dari MetaTrader 5, melabeli pergerakan **satu bar ke depan** menjadi tiga kelas (**FLAT / UP / DOWN**), melatih klasifikasi **XGBoost** dengan validasi **purged walk-forward** dan **holdout 15%**, mengoptimalkan hyperparameter dengan **Optuna (48 trial)**, lalu pada operasi mengambil bar terbaru dari MT5, memprediksi probabilitas, mengambil berita makro terkait emas/dolar/Fed, menilai sentimen via **Google Gemini** (dengan fallback keyword), menggabungkan ML dan sentimen menjadi **rekomendasi trading**, mengirim laporan ke **Telegram** setiap **60 menit**, dan mencatat setiap prediksi di **prediction log** untuk evaluasi akurasi setelah outcome 1 jam. Sistem **tidak** mengeksekusi order otomatis.

---

## 3. Tujuan dan ruang lingkup

| Aspek | Nilai |
|--------|--------|
| Instrumen | XAUUSD (spot gold vs USD) |
| Timeframe | H1 (1 candle = 1 jam) |
| Horizon | 1 bar H1 ke depan |
| Kelas | 0=FLAT, 1=UP, 2=DOWN |
| Threshold FLAT | \|ln(close_{t+1}/close_t)\| ≤ **0.0022** (0,22% per jam) |
| Model | XGBoost multiclass |
| Validasi | Purged CV 5-fold + embargo 1 bar + holdout 15% |
| Live | MT5 → fitur → XGBoost → berita → Gemini → Telegram |
| Jadwal | `schedule_every_minutes: 60` |
| Backtest terpisah | Stage 7–8 tidak ada di pipeline utama |

**Di luar ruang lingkup:** auto-trading, manajemen risiko otomatis, VPS production tanpa dokumentasi terpisah di repo ini.

---

## 4. Data dan model aktif (kutipan skripsi)

| Item | Nilai |
|------|--------|
| Sumber data | MetaTrader 5, simbol XAUUSD, timeframe H1 |
| Rentang historis | Bergantung unduhan MT5 (broker) |
| Bar training (setelah filter) | ~25.000+ bar H1 (Sen–Jum, weekend dihapus) |
| Holdout | 15% blok waktu terakhir |
| CV macro-F1 (Optuna) | ~0.40 (contoh run) |
| Holdout macro-F1 | ~0.41 |
| Holdout balanced accuracy | ~0.41 |
| Stage 6 abstain τ | ~0.29 (contoh) |
| Signal rate non-FLAT (holdout) | ~32% (target seimbang, tidak over-abstain) |

> Akurasi tinggi pada holdout tidak menjamin profit. Gunakan **prediction log** untuk evaluasi operasional setelah 2–4 minggu.

---

## 5. Arsitektur sistem

### A. Pipeline offline (`run_pipeline.py`)

```
MT5/CSV H1 → Stage1 (clean) → Stage2 (label) → Stage3 (fitur)
  → Stage4 (purged folds) → Stage5 (XGBoost+Optuna) → Stage6 (abstain τ)
```

Output: `artifacts/run_<UTC>/` berisi model, metadata, `stage_6_predictions.parquet`.

### B. Pipeline online (Stage 9)

```
MT5 bar terbaru → fitur (sama training) → XGBoost proba
  → NewsAPI + RSS → Gemini sentimen (horizon: 1 jam ke depan)
  → consensus_matrix → rekomendasi
  → Telegram + append_prediction + resolve outcome (setelah 1 jam)
```

---

## 6. Tahapan metodologi (detail)

### Stage 1 — Data (`stage_1_data.py`)

- Input: `data/xauusd_h1.csv` atau `auto_fetch_mt5: true`
- Kolom: `time, open, high, low, close, spread` (UTC)
- `exclude_weekend_bars: true` — tidak trading Sabtu/Minggu
- `max_gap_minutes: 180` — gap >3 jam dianggap putus
- Anti-leakage: tidak memakai harga masa depan pada bar t

### Stage 2 — Labeling (`stage_2_labeling.py`)

```
r = ln(close[t+1] / close[t])
```
- FLAT: |r| ≤ flat_return_threshold
- UP: r > threshold
- DOWN: r < -threshold
- Filter: spread z-score > 4, wick ratio > 6

### Stage 3 — Fitur (`stage_3_features.py`, `utils/gbpjpy_features.py`)

Fitur stasioner dengan rolling **shift(1)**. Contoh: `log_return`, `distance_to_ema20/50`, `rsi_zscore`, `atr_zscore`, `session_*`, `day_of_week`, `hurst_proxy`, `realized_vol_shift1`, rasio wick/body.

### Stage 4 — Validasi (`stage_4_validation.py`)

- 5 fold purged walk-forward
- `embargo_bars: 1` (sesuai horizon H1)
- Output: `stage_4_fold_indices.json`

### Stage 5 — Training (`stage_5_training.py`)

- XGBoost 3 kelas, objective macro-F1
- Optuna 48 trial; regularisasi L1/L2/gamma/min_child_weight
- `class_balance: inverse` (bukan inverse_sqrt — lebih agresif untuk UP/DOWN)
- Early stopping 60 rounds
- Penalti recall UP/DOWN nol pada CV

### Stage 6 — Keputusan (`stage_6_ensemble.py`)

- Probabilitas mentah XGBoost
- **Abstain τ:** P(FLAT) ≥ τ → FLAT; else argmax(UP, DOWN)
- Pemilihan τ: skor campuran accuracy + balanced accuracy + kedekatan rate FLAT
- Parameter: `abstain_score_accuracy_weight: 0.45`, `abstain_score_balanced_accuracy_weight: 0.55`

### Stage 9 — Operasional

- `run_daily_report()` — satu siklus analisis
- `fetch_all_headlines` — NewsAPI + ForexFactory RSS
- `gemini_news_sentiment` — prompt menyebut horizon **1 jam ke depan (1 bar H1)**
- `consensus_matrix` — hindari BUY/SELL keras saat prediksi dominan FLAT
- `append_prediction` + `resolve_pending_outcomes`

---

## 7. Integrasi sentimen (Gemini)

Prompt inti (lihat `utils/gemini_client.py`):

- Instrumen: XAUUSD
- Horizon: **1 jam ke depan (1 bar H1)** — bukan outlook jangka panjang
- Output JSON: `score` (-1/0/+1), `summary` (1 kalimat Indonesia)
- Fallback: keyword sentiment jika API gagal

Query berita (`stage_9.newsapi_query`): XAUUSD, gold price, spot gold, Fed, FOMC, US dollar, bullion.

---

## 8. Rekomendasi trading (konsensus)

Variabel kunci:

- `lean_delta = P(UP) - P(DOWN)`
- `combined = lean_delta + sentiment_weight * sentiment`
- Jika `pred_class == FLAT` dan `P(FLAT) >= 0.5` → mode konservatif (WEAK/NEUTRAL lebih mungkin daripada BUY/SELL keras)

---

## 9. Evaluasi live

**File:** `logs/prediction_log.csv`

| Kolom | Fungsi |
|------|--------|
| `pred_class` | FLAT/UP/DOWN |
| `recommendation` | Rekomendasi saat prediksi |
| `ml_correct` | Prediksi vs outcome 1 jam kemudian |
| `signal_correct` | Rekomendasi BUY/SELL vs arah aktual |
| `signal_pnl_proxy` | Skor arah ±1 (bukan PnL uang) |

Script: `scripts/resolve_prediction_log.py`, `scripts/weekly_accuracy_report.py`

---

## 10. Konfigurasi (`configs/pipeline.yaml`)

| Bagian | Parameter penting |
|--------|-------------------|
| project | symbol XAUUSD, timeframe H1, horizon_bars 1 |
| risk | flat_return_threshold 0.0022 |
| stage_5 | n_trials 48, holdout 0.15, class_balance inverse |
| stage_6 | abstain_flat_rate_tolerance 0.05, score weights 0.45/0.55 |
| stage_9 | schedule_every_minutes 60, mt5_bars 120 |

---

## 11. Keterbatasan (diskusi Bab 4–5)

1. Bukan jaminan profit; informasi pendukung keputusan.
2. Spread/slippage eksekusi tidak dimodelkan penuh.
3. Kelas FLAT signifikan pada XAUUSD H1 — threshold dan abstain τ mengatur trade-off akurasi vs frekuensi sinyal.
4. Gemini sering fallback (kuota/rate limit).
5. Akurasi live butuh waktu kumpul di prediction log.
6. PC + MT5 harus aktif; koneksi internet untuk Telegram/API.

---

## 12. Usulan struktur BAB skripsi

1. Pendahuluan  
2. Landasan teori (time series, classification, sentiment analysis)  
3. Metodologi (data, label, fitur, validasi, model, live system)  
4. Implementasi  
5. Hasil dan pembahasan  
6. Kesimpulan dan pengembangan (backtest terpisah, VPS, multi-instrument)

---

## 13. Prompt untuk Gemini

```
Saya menyusun skripsi tentang sistem prediksi arah XAUUSD timeframe H1.
Lampiran: GEMINI_BRIEFING_SKRIPSI.md, pipeline.yaml, kode stage_1-6 dan stage_9, run_summary.json.

Tugas:
1. Pahami arsitektur offline (stage 1-6) dan online (stage 9, scheduler 60 menit).
2. Bantu BAB III Metodologi dan BAB IV Implementasi (bahasa Indonesia formal).
3. Buat diagram mermaid alur data dan tabel variabel.
4. Jelaskan mitigasi overfitting dan perbedaan akurasi ML vs evaluasi prediction log.
5. Cantumkan keterbatasan (no auto-trading, threshold FLAT, data broker-limited).

Asumsi: timeframe H1, horizon 1 bar, BUKAN D1. Sistem TIDAK auto-trading.
```

---

## 14. Checklist sebelum kirim ke Gemini

- [ ] GEMINI_BRIEFING_SKRIPSI.md (XAUUSD H1)
- [ ] pipeline.yaml
- [ ] run_summary.json dari run terbaru
- [ ] contoh_laporan_harian.md (template H1)
- [ ] ARCHITECTURE.md
- [ ] OPERASIONAL_WINDOWS.md (XAUUSD H1)
- [ ] Tidak ada referensi GBPJPY D1 sebagai sistem aktif

---

*Dokumentasi selaras dengan implementasi per 2026-05-25.*
