# Parameter Konfigurasi

Sumber: `configs/pipeline.yaml`  
Run referensi hasil evaluasi: `artifacts/run_20260602_005304`

---

## Proyek & Data

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `project.symbol` | XAUUSD | Pair emas spot vs USD |
| `project.timeframe` | M15 | Satu candle = 15 menit |
| `project.horizon_bars` | 12 | Horizont prediksi: 12 × 15 menit = **3 jam** |
| `project.training_label_stride` | 1 | Semua bar eligible dipakai (tanpa subsample) |
| `project.decision_bar` | close | Keputusan pada harga penutupan bar |
| `risk.flat_return_threshold` | 0.0011 | Ambang log-return untuk kelas UP/DOWN (~0,11%) |
| `risk.point_size` | 0.01 | Ukuran point broker (untuk konversi pip) |
| `risk.spread_cap_quantile` | 0.99 | Spread ekstrem dipotong pada kuantil 99% |

---

## Stage 1 — Data Preparation

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `stage_1.input_csv` | data/xauusd_m15.csv | File OHLCV utama |
| `stage_1.auto_fetch_mt5` | true | Unduh dari MT5 jika CSV belum ada |
| `stage_1.mt5_fetch_bars` | 200000 | Maksimum bar historis dari MT5 |
| `stage_1.timezone` | UTC | Semua timestamp dinormalisasi UTC |
| `stage_1.max_gap_minutes` | 60 | Gap > 1 jam = sesi putus (M15) |
| `stage_1.forward_fill_max_bars` | 2 | Isi missing maksimal 2 bar |
| `stage_1.exclude_weekend_bars` | true | Hapus bar akhir pekan |

**Hasil run aktif:** 49.265 bar mentah, periode 2024-04-16 → 2026-05-28 UTC.

---

## Stage 2 — Labeling

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `stage_2.max_spread_zscore` | 4.0 | Bar dengan spread z-score > 4 dibuang |
| `stage_2.wick_anomaly_ratio` | 6.0 | Filter candle wick ekstrem |
| `stage_2.atr_window` | 14 | Window ATR untuk risk labeling |
| `stage_2.risk_labeling.sl_atr_multiplier` | 1.5 | Jarak SL = 1,5 × ATR(14) |
| `stage_2.risk_labeling.tp_rr` | 1.8 | Jarak TP = 1,8 × jarak SL |

---

## Stage 3 — Feature Engineering

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `stage_3.vol_window` | 60 | Window volatilitas realized |
| `stage_3.ema_fast` | 20 | EMA cepat (jarak harga) |
| `stage_3.ema_slow` | 50 | EMA lambat |
| `stage_3.zscore_window` | 60 | Window z-score indikator |
| `stage_3.atr_window` | 14 | ATR untuk fitur & trade plan |
| `stage_3.rsi_window` | 14 | RSI z-score |
| `stage_3.hurst_window` | 32 | Proxy persistensi (Hurst) |

---

## Stage 4 — Validasi

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `stage_4.n_splits` | 5 | Jumlah fold walk-forward |
| `stage_4.embargo_bars` | 1 | Minimal 1 bar (diperkuat ≥ horizon di kode) |
| `stage_4.regime_vol_window` | 24 | Tag regime vol per fold |
| `stage_4.min_train_rows` | 1000 | Minimum bar training per fold |
| `stage_4.min_test_rows` | 100 | Minimum bar test per fold |

---

## Stage 5 — Training & Optuna

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `stage_5.model_type` | xgboost | Model utama klasifikasi 3 kelas |
| `stage_5.n_trials` | 48 | Jumlah trial Optuna |
| `stage_5.n_cv_splits` | 5 | CV internal saat optimasi |
| `stage_5.early_stopping_rounds` | 60 | Early stopping XGBoost |
| `stage_5.final_holdout_frac` | 0.15 | 15% data terakhir = holdout final |
| `stage_5.final_n_estimators` | 1200 | Pohon final setelah tuning |
| `stage_5.class_balance` | inverse | Bobot kelas seimbang |
| `stage_5.train_binary_models_enabled` | true | Model biner UP/DOWN tambahan |
| `stage_5.optuna_weight_expectancy` | 0.35 | Bobot expectancy di objective |
| `stage_5.optuna_weight_sharpe` | 0.20 | Bobot Sharpe simulasi |
| `stage_5.optuna_weight_prec_dir` | 0.25 | Bobot precision directional |
| `stage_5.optuna_weight_f1_dir` | 0.20 | Bobot F1 directional |
| `stage_5.optuna_sim_conf_up` | 0.52 | Simulasi threshold UP saat tuning |
| `stage_5.optuna_sim_conf_down` | 0.52 | Simulasi threshold DOWN |
| `stage_5.optuna_sim_abstain_tau` | 0.42 | Simulasi abstain FLAT |
| `stage_5.min_signal_rate_target` | 0.25 | Target minimal fraksi sinyal |

**Hyperparameter yang di-tune Optuna:** `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`.

---

## Stage 6 — Decision Logic

| Parameter | Nilai (run aktif) | Penjelasan |
|-----------|-------------------|------------|
| `stage_6.use_binary_models` | false | Keputusan dari prob 3-kelas XGBoost |
| `stage_6.min_signal_rate_target` | 0.25 | Minimal 25% bar memicu sinyal |
| `stage_6.max_signal_rate_target` | 0.50 | Maksimal 50% bar memicu sinyal |
| `stage_6.threshold_relax_step` | 0.03 | Langkah relaksasi threshold |
| **Hasil tuning** `conf_up` | **0.70** | P(UP) minimum untuk sinyal beli |
| **Hasil tuning** `conf_down` | **0.65** | P(DOWN) minimum untuk sinyal jual |
| **Hasil tuning** `abstain_tau` | **0.35** | Maksimum P(FLAT) agar tidak abstain |

---

## Stage 9 — Live Trading

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `stage_9.bar_analysis_delay_seconds` | 120 | Tunggu 2 menit setelah bar M15 baru |
| `stage_9.mt5_bars` | 300 | Jumlah bar MT5 untuk inferensi |
| `stage_9.min_rr_to_send` | 1.5 | RR minimum agar sinyal dikirim |
| `stage_9.max_sl_pips` | 150 | Batas SL dalam pip |
| `stage_9.trade_plan.sl_atr_multiplier` | 1.5 | SL live = 1,5 × ATR |
| `stage_9.trade_plan.tp_rr_base` | 1.8 | TP = 1,8 × SL |
| `stage_9.trade_plan.tp_rr_strong` | 2.0 | TP untuk STRONG BUY/SELL |
| `stage_9.trade_plan.min_confidence` | 0.55 | Confidence minimum trade plan |
| `stage_9.execution.lot` | 0.01 | Lot dasar MT5 |
| `stage_9.execution.magic` | 950001 | Magic number order |
| `stage_9.execution.confirm_timeout_sec` | 120 | Pending konfirmasi `ya` (detik) |
| `stage_9.moment_alert.enabled` | true | Analisa otomatis per bar |
| `stage_9.moment_alert.poll_seconds` | 60 | Interval loop moment alert |
| `stage_9.moment_alert.min_confidence` | 0.60 | Confidence minimum alert |
| `stage_9.moment_alert.send_no_trade` | true | Kirim ringkasan NO TRADE |
| `stage_9.moment_alert.no_trade_min_interval_minutes` | 0 | 0 = setiap bar M15 |
| `stage_9.auto_retrain.enabled` | true | Retrain otomatis aktif |
| `stage_9.auto_retrain.min_interval_hours` | 6.0 | Minimal 6 jam antar retrain |
| `stage_9.auto_retrain.trigger_on_high_drift` | true | Retrain saat HIGH_DRIFT |
| `stage_9.gemini.max_retries` | 3 | Retry API Gemini |
| `stage_9.gemini.cache_max_age_minutes` | 60 | Cache sentiment 60 menit |

---

## Lot Sizing (Confidence & Sentiment)

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| `stage_9.lot_sizing.base_lot` | 0.01 | Lot dasar |
| `stage_9.lot_sizing.high_confidence_threshold` | 0.70 | Tier HIGH |
| `stage_9.lot_sizing.medium_confidence_threshold` | 0.55 | Tier MEDIUM |
| `stage_9.lot_sizing.low_confidence_threshold` | 0.45 | Tier LOW |
| `stage_9.lot_sizing.high_lot_multiplier` | 1.5 | HIGH = 1,5× base |
| `stage_9.lot_sizing.medium_lot_multiplier` | 1.0 | MEDIUM = 1× base |
| `stage_9.lot_sizing.low_lot_multiplier` | 0.7 | LOW = 0,7× base |
| `stage_9.lot_sizing.sentiment_reduce_factor` | 0.8 | Kurangi lot jika konflik ringan |
| `stage_9.lot_sizing.sentiment_boost_factor` | 1.1 | Boost lot jika sentiment searah kuat |

---

## Meta-Filter (Artifact, bukan YAML)

| Parameter | Nilai run aktif | Sumber |
|-----------|-----------------|--------|
| Meta threshold | **0.41** | `stage_5/meta_model_report.json` |
| Precision gain UP | **+19,7%** | Baseline 0,577 → filtered 0,773 |
| Precision gain DOWN | **+12,8%** | Baseline 0,562 → filtered 0,690 |

---

## Kalibrasi Probabilitas (Run Aktif)

| Metode | ECE | Dipilih? |
|--------|-----|----------|
| none (raw) | 0,146 | **Ya** |
| isotonic | 0,151 | Tidak |
| temperature | 0,347 | Tidak |

Sumber: `stage_5/calibration_report.json` — model memakai probabilitas raw karena kalibrasi tidak menurunkan ECE.
