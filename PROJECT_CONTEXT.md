# PROJECT CONTEXT

## 1. PROJECT OVERVIEW
- Sistem ini membangun **pipeline trading assistant** end-to-end: training model arah harga (FLAT/UP/DOWN), live inference on-demand via Telegram, validasi berita via Gemini, pembuatan rencana trade ATR (Entry/SL/TP/RR), dan eksekusi order MT5 berbasis konfirmasi user.
- Tujuan utama: menghasilkan sinyal trading jangka pendek yang terstruktur, dengan guardrail risiko (`NO TRADE`, confidence gate, meta-filter, spread/news context), bukan sekadar klasifikasi akurasi.
- Instrumen aktif default: `XAUUSD`; override runtime tersedia via `/analisa EURUSD M15`.
- Timeframe aktif saat ini: `M15`, horizon label default: `4` bar (`~1 jam`), namun sistem tetap support multi-timeframe (`M1..D1`) via parser + MT5 constant mapping.
- Stack utama:
  - Bahasa: Python 3.x
  - Runtime: local Windows + MT5 Terminal + Telegram Bot API
  - ML: XGBoost, Optuna, scikit-learn (calibration + permutation importance)
  - Data: pandas, numpy, pyarrow/parquet, joblib
  - Integrasi: `MetaTrader5`, `requests`, `google-genai` (fallback `google.generativeai`)
- Dependency kritikal operasional:
  - Broker feed: MT5 (atau fallback CSV tail jika MT5 unavailable)
  - Messaging: Telegram bot token/chat_id
  - News: NewsAPI + RSS feed
  - LLM sentiment: Gemini API key

## 2. ARCHITECTURE DIAGRAM (text/ASCII)
```text
[MT5/CSV Historical Bars]
        |
        v
 Stage 1 (stage_1_data.py)
  - clean OHLCV, gap flags, session/day, realized_vol_shift1
  -> stage_1_clean.parquet + stage_1_metadata.json
        |
        v
 Stage 2 (stage_2_labeling.py)
  - label target 3-class + risk-aware tp_sl_outcome + MFE/MAE
  -> stage_2_labeled.parquet + stage_2_metadata.json
        |
        v
 Stage 3 (stage_3_features.py + utils/gbpjpy_features.py)
  - stationary features (shift(1), anti-leakage)
  -> stage_3_featured.parquet + stage_3_metadata.json
        |
        v
 Stage 4 (stage_4_validation.py)
  - purged walk-forward fold indices + embargo
  -> stage_4_fold_indices.json + stage_4_metadata.json
        |
        v
 Stage 5 (stage_5_training.py)
  - Optuna profit-aware objective
  - feature pruning (permutation importance)
  - calibration (none/isotonic/temperature)
  - meta-filter XGB binary
  - regime models
  -> xgb_model.joblib
  -> threshold_config.json
  -> meta_model.joblib + meta_model_report.json
  -> calibration_report.json/.png
  -> regime_models.joblib
  -> feature_registry.json + stage_5_best_config.json
        |
        v
 Stage 6 (stage_6_ensemble.py)
  - choose abstain tau from validation tail
  - generate holdout predictions + tp/sl holdout stats
  -> stage_6_predictions.parquet + stage_6_metadata.json
        |
        v
 [run_pipeline.py]
  -> run_summary.json + active_run pointer update

============================================================
LIVE FLOW (stage_9_live_demo.py via telegram_bot/service)
============================================================
/analisa command
  -> fetch MT5 latest bars (or CSV fallback)
  -> prepare_inference_bars + build_gbpjpy_features
  -> load xgb_model.joblib (and optional regime model)
  -> infer probs FLAT/UP/DOWN
  -> fetch headlines (NewsAPI+RSS)
  -> Gemini sentiment (fallback keyword)
  -> consensus_matrix recommendation
  -> build_trade_plan ATR (Entry/SL/TP/RR + no-trade gates)
  -> apply meta_model filter (if exists)
  -> send Telegram report
  -> if valid signal: ask "ya/tidak" then place MT5 order
  -> append prediction log + resolve pending outcomes
```

## 3. FILE MAP
- `run_pipeline.py`
  - Fungsi: Orkestrasi stage 1-6 (+opsional stage9).
  - Input: `configs/pipeline.yaml`, flag CLI.
  - Output: `artifacts/run_*/...`, `run_summary.json`, update `active_run.txt`.
  - Dependency: semua `stage_*.py`, `utils/config_loader.py`, `utils/paths.py`.

- `stage_1_data.py`
  - Fungsi: ingest + cleaning OHLCV, weekend drop, gap flag, session/calendar.
  - Input: CSV atau MT5 fetch via `utils/mt5_export.ensure_input_csv`.
  - Output: `stage_1_clean.parquet`, `stage_1_metadata.json`.
  - Dependency: `utils/mt5_export.py`, `utils/sessions.py`, `utils/trading_calendar.py`.

- `stage_2_labeling.py`
  - Fungsi: label `target` (0/1/2), risk-aware `tp_sl_outcome`, MFE/MAE.
  - Input: dataframe stage1 + threshold/horizon config.
  - Output: `stage_2_labeled.parquet`, `stage_2_metadata.json`.
  - Dependency: config `risk` + `stage_2`.

- `stage_3_features.py`
  - Fungsi: generate feature matrix anti-leakage.
  - Input: dataframe stage2.
  - Output: `stage_3_featured.parquet`, metadata fitur.
  - Dependency: `utils/gbpjpy_features.py`.

- `utils/gbpjpy_features.py`
  - Fungsi: feature engineering inti (`log_return`, EMA distance, RSI/ATR zscore, hurst proxy, spread_shock, bar_momentum, session_progress).
  - Input: OHLCV dataframe + config stage3.
  - Output: dataframe featured + `feat_cols`.
  - Dependency: `utils/rolling_safe.py`, `utils/sessions.py`.

- `stage_4_validation.py`
  - Fungsi: buat fold indices purged walk-forward + embargo.
  - Input: dataset train.
  - Output: `stage_4_fold_indices.json`, metadata fold.
  - Dependency: numpy/pandas.

- `stage_5_training.py`
  - Fungsi: training inti model + calibrator + meta-filter + regime model + threshold optimization.
  - Input: dataset train berfitur, fold indices stage4.
  - Output: `xgb_model.joblib`, `meta_model.joblib`, `threshold_config.json`, `calibration_report.json`, `feature_registry.json`, `regime_models.joblib`, `stage_5_best_config.json`.
  - Dependency: xgboost, optuna, sklearn, `utils/regime.py`.

- `utils/regime.py`
  - Fungsi: klasifikasi regime (`TREND_UP`, `TREND_DOWN`, `RANGE`, `HIGH_VOL`).
  - Input: OHLC dataframe.
  - Output: series regime.
  - Dependency: pandas/numpy.

- `stage_6_ensemble.py`
  - Fungsi: pilih `abstain_tau`, bangun prediksi holdout dan metrik stage6.
  - Input: dataset train + model bundle stage5.
  - Output: `stage_6_predictions.parquet`, `stage_6_metadata.json`.
  - Dependency: `joblib`, sklearn metrics.

- `stage_9_live_demo.py`
  - Fungsi: live inference satu siklus (data -> model -> sentiment -> consensus -> trade plan -> report).
  - Input: config, `run_dir`, optional symbol/timeframe override.
  - Output: `daily_report.json`, `daily_report.md`, Telegram message payload.
  - Dependency: `utils/news_fetch.py`, `utils/gemini_client.py`, `utils/prediction_log.py`, `utils/mt5_export.py`.

- `utils/telegram_bot.py`
  - Fungsi: polling bot command `/analisa`, `/pairs`, `/akurasi`, `/status`, `/help` + konfirmasi eksekusi `ya/tidak`.
  - Input: updates Telegram.
  - Output: trigger analysis, pesan Telegram, order execution call.
  - Dependency: `utils/mt5_execution.py`, `utils/prediction_log.py`.

- `utils/mt5_execution.py`
  - Fungsi: `place_market_order_from_plan`.
  - Input: `trade_plan` + execution config.
  - Output: payload status order (ok/error, retcode, deal/order).
  - Dependency: package `MetaTrader5`, `utils/mt5_export.resolve_symbol`.

- `scripts/stage9_service.py`
  - Fungsi: service wrapper bot + single-instance lock + background moment alert loop.
  - Input: config + run_dir pointer.
  - Output: long-running bot service, `logs/moment_alert_state.json`.
  - Dependency: `stage_9_live_demo.py`, `utils/telegram_bot.py`.

- `retrain_scheduler.py`
  - Fungsi: periodic retrain + drift detection (PSI) + runtime risk factor.
  - Input: config + flags skip.
  - Output: run retrain artifacts, `logs/runtime_risk.json`, `logs/retrain_log.jsonl`.
  - Dependency: stage1-6 + feature registry stage5.

- File utilitas pendukung lain:
  - `utils/config_loader.py`: merge `pipeline.yaml` + `pipeline.secrets.yaml`
  - `utils/paths.py`: resolve/latest/active run
  - `utils/news_fetch.py`: fetch NewsAPI + RSS + simple sentiment fallback
  - `utils/gemini_client.py`: Gemini sentiment dengan fallback model chain
  - `utils/prediction_log.py`: append + resolve outcome + summary `/akurasi`
  - `utils/telegram_notify.py`: wrapper Telegram API
  - `scripts/preflight_check.py`: diagnosa readiness dependency/secrets/MT5/API
  - `scripts/fetch_ohlcv_from_mt5.py`: fetch data historis ke CSV

## 4. STAGE PIPELINE
- Stage 1 — Data Prep
  - Script: `stage_1_data.py`
  - Tugas: ingest + cleansing + session flags + vol regime base.
  - Output: `stage_1_clean.parquet`, `stage_1_metadata.json`.
  - Flag terkait: via config (`auto_fetch_mt5`, `exclude_weekend_bars`, `max_gap_minutes`).

- Stage 2 — Labeling
  - Script: `stage_2_labeling.py`
  - Tugas: label 3 kelas + risk-aware labels.
  - Output: `stage_2_labeled.parquet`, `stage_2_metadata.json`.
  - Flag: config `risk.flat_return_threshold`, `project.horizon_bars`, `stage_2.risk_labeling`.

- Stage 3 — Feature Engineering
  - Script: `stage_3_features.py`
  - Tugas: transform fitur stasioner + anti-leakage.
  - Output: `stage_3_featured.parquet`, `stage_3_metadata.json`.
  - Flag: config `stage_3.*`.

- Stage 4 — Validation Fold Builder
  - Script: `stage_4_validation.py`
  - Tugas: purged walk-forward fold + embargo.
  - Output: `stage_4_fold_indices.json`, `stage_4_metadata.json`.
  - Flag: config `stage_4.*`.

- Stage 5 — Model Training
  - Script: `stage_5_training.py`
  - Tugas: Optuna search, final fit, threshold sweep, calibration, meta-model, regime models, feature selection.
  - Output: model bundle + reports json/png + registry.
  - CLI skip flags (via `run_pipeline.py`/`retrain_scheduler.py`):
    - `--skip-task1` .. `--skip-task6`
    - `--skip-fix1` (hybrid calibration)
    - `--skip-fix2` (meta-model precision optimization)

- Stage 6 — Decision / Ensemble
  - Script: `stage_6_ensemble.py`
  - Tugas: select `abstain_tau`, generate predictions + holdout stats.
  - Output: `stage_6_predictions.parquet`, `stage_6_metadata.json`.
  - Flag: config `stage_6.use_flat_abstain_search` dll.

- Stage 9 — Live Inference
  - Script: `stage_9_live_demo.py`, service `scripts/stage9_service.py`
  - Tugas: inferensi real-time + Telegram report + optional MT5 execution confirm flow + moment alert.
  - Output: `stage_9/daily_report.json`, `stage_9/daily_report.md`, prediction log.
  - CLI utama:
    - `stage_9_live_demo.py`: `--latest-run`, `--skip-mt5`, `--dry-run`, `--bot`, `--test-telegram`, `--no-telegram`
    - `scripts/stage9_service.py`: `--latest-run`, `--skip-mt5`

## 5. MODEL INVENTORY
- `artifacts/run_*/stage_5/xgb_model.joblib`
  - Jenis: bundle model utama multiclass.
  - Isi penting: `model`, `features`, `thresholds`, `meta`, `calibration`, `regime_enabled`, `regime_models_path`.
  - Input features: dari `feature_registry.active_features` (run terbaru: 14 fitur).
  - Output: `predict_proba -> [P(FLAT), P(UP), P(DOWN)]`.
  - Dipakai: Stage6 + Stage9 live inference.

- `artifacts/run_*/stage_5/meta_model.joblib`
  - Jenis: XGBoost binary classifier (profitable filter).
  - Input: meta-features (`p_flat`, `p_up`, `p_down`, `spread`, `atr_zscore`, `session_london`, `session_ny`, `london_open_proxy`).
  - Output: `p_meta` profitable probability + threshold.
  - Dipakai: Stage9 setelah trade_plan untuk gate execute/no-trade.

- `artifacts/run_*/stage_5/regime_models.joblib`
  - Jenis: dict model per regime.
  - Input: fitur aktif untuk regime terkait.
  - Output: probabilitas kelas sama seperti global model.
  - Dipakai: Stage9 `infer_direction` jika regime model tersedia dan load sukses.

- `artifacts/run_*/stage_5/threshold_config.json`
  - Jenis: konfigurasi threshold hasil sweep.
  - Input: holdout probs + outcome.
  - Output: `conf_up`, `conf_down`, `abstain_tau`, plus metrics.
  - Dipakai: tersimpan di bundle; referensi decision layer (manual/lanjutan).

- `artifacts/run_*/stage_5/calibration_report.json`
  - Jenis: evaluasi calibration methods.
  - Output: selected method + ECE/MCE/Brier.
  - Dipakai: menentukan wrapper model di `xgb_model.joblib`.

- `artifacts/run_*/stage_5/feature_registry.json`
  - Jenis: registry fitur aktif + permutation importance.
  - Dipakai: dokumentasi + drift monitor (`retrain_scheduler.py`) + auditability.

- `artifacts/run_*/stage_6/stage_6_metadata.json`
  - Jenis: metadata decision stage6 (`abstain_tau`, holdout metrics, tp/sl holdout stats).
  - Dipakai: evaluasi kualitas sinyal offline.

- `logs/runtime_risk.json`
  - Jenis: runtime risk control artifact dari retrain scheduler.
  - Output: `drift_status`, `risk_multiplier`, high drift features.
  - Dipakai: operasional risk monitoring (belum auto-wired ke lot sizing stage9 secara langsung).

## 6. DECISION LOGIC
- BUY / SELL / ABSTAIN ditentukan dari beberapa layer:
  1) **Model probability layer**: model menghasilkan P(FLAT), P(UP), P(DOWN).
  2) **Consensus layer** (`consensus_matrix`): gabungan `lean_delta = P(UP)-P(DOWN)` + sentiment score menghasilkan `STRONG BUY/SELL`, `BUY/SELL`, `WEAK`, atau `NEUTRAL / ABSTAIN`.
  3) **Trade plan layer** (`build_trade_plan`): side diturunkan dari rekomendasi; jika FLAT dominan atau confidence kurang -> no-trade reason.
  4) **Meta-filter layer**: jika `p_meta < threshold_meta`, paksa `is_no_trade = true`.

- Cara kerja `abstain_tau` (Stage6):
  - Rule: jika `P(FLAT) >= tau`, prediksi `FLAT`, selain itu pilih argmax antara `UP/DOWN`.
  - `tau` dicari dengan sweep di validation tail untuk menyeimbangkan accuracy + balanced_accuracy + kesesuaian flat-rate prediksi terhadap distribusi aktual.
  - Run terbaru: `tau ≈ 0.268`.

- Pemakaian `threshold_config.json`:
  - Menyimpan hasil threshold sweep profitability (`conf_up/conf_down/abstain_tau`) + metrik expectancy/sharpe.
  - Disimpan ke bundle stage5, menjadi referensi threshold kebijakan decision.
  - `meta_threshold_optimal` juga ditulis saat meta filter aktif.

- Pengaruh meta-filter:
  - Stage9 membentuk `x_meta` dari proba + market context.
  - Prediksi `p_meta` dibandingkan threshold (`~0.52` run terbaru).
  - Jika tidak lolos: trade plan langsung ditandai no-trade dengan reason `meta_filter x < t`.

- Kondisi `NO TRADE` (semua yang terdeteksi eksplisit):
  - Side tidak valid (`NONE`) karena bias lemah/FLAT dominan.
  - Confidence directional `< stage_9.trade_plan.min_confidence` (atau per_symbol override).
  - Meta-filter reject (`p_meta < threshold`).
  - Trade plan tanpa SL/TP valid (guard di flow eksekusi).
  - Execution disabled (`stage_9.execution.enabled=false`) walau sinyal ada.
  - User membalas `tidak` atau timeout konfirmasi.

- UNCLEAR:
  - `threshold_config.best.conf_up/conf_down` saat ini belum dipakai langsung secara eksplisit di `stage_9_live_demo.py` untuk gating rekomendasi final (lebih dominan `consensus_matrix + trade_plan min_conf + meta_filter`).

## 7. LIVE INFERENCE FLOW
1. **Trigger**
   - User kirim `/analisa` atau `/analisa SYMBOL TF` ke bot.
2. **Validation**
   - `telegram_bot.validate_mt5_inputs` cek symbol + timeframe di MT5 (jika mode MT5 live).
3. **Data Fetch**
   - Ambil `stage_9.mt5_bars` terbaru dari MT5 via `fetch_mt5_bars`; fallback CSV tail bila MT5 unavailable/skip.
4. **Preprocess + Features**
   - `prepare_inference_bars` menambahkan kolom kalender/vol yang dibutuhkan.
   - `build_gbpjpy_features` membuat fitur sesuai training.
5. **Model Inference**
   - Load `xgb_model.joblib`.
   - Jika regime enabled + model regime ada, pilih model regime; else pakai global model.
   - Hitung `probs` dan class prediction.
6. **News + Sentiment**
   - Ambil headline via `fetch_all_headlines` (NewsAPI+RSS).
   - Kirim ke Gemini (`gemini_news_sentiment`) dengan fallback models/keyword.
7. **Decision**
   - Jalankan `consensus_matrix` untuk rekomendasi tekstual.
   - Hitung `trade_plan` ATR: side, entry, SL, TP, RR, confidence, no-trade reasons.
   - Jalankan meta-filter (`meta_model.joblib`) untuk final allow/deny.
8. **Output**
   - Format markdown report + kirim ke Telegram.
   - Simpan `daily_report.json` + `daily_report.md`.
   - Catat ke `logs/prediction_log.csv`; jalankan resolver outcome pending.
9. **Execution (opsional)**
   - Jika trade valid: bot minta konfirmasi `ya/tidak`.
   - Jika `ya`: `place_market_order_from_plan` kirim order market MT5.

## 8. KNOWN ISSUES & CURRENT STATE
- State run terbaru (berdasarkan `artifacts/run_20260528_091616`):
  - Pipeline: `xauusd_directional_m15`
  - `n_train_rows`: 41,553
  - Stage5 holdout: macro-F1 `0.3632`, balanced acc `0.3834`
  - Stage6 holdout: accuracy `0.4194`, balanced acc `0.4053`, signal_rate `0.5908`

- Target DoD yang **tercapai** (implementasi):
  - Profit-aware objective + threshold sweep aktif.
  - Hybrid calibration (none/isotonic/temperature) aktif + auto-select.
  - Meta-model XGBoost binary + threshold optimization aktif.
  - Regime-aware training + optional regime inference aktif.
  - Feature pruning + feature registry aktif.
  - Integration sanity artifact (`logs/integration_check.json`) valid (`no_nan=true`, `sum_to_1=true`).

- Target DoD yang **belum tercapai** (kinerja metrik):
  - Calibration quality target ketat (mis. ECE < 0.05) belum tercapai; run terbaru ECE terbaik `0.157` (selected `none`).
  - Precision gain directional dari meta-filter belum >= +5%; run terbaru:
    - delta UP `+0.0025`
    - delta DOWN `+0.0083`

- Flag/warning kondisi operasional saat ini:
  - `logs/runtime_risk.json`: `HIGH_DRIFT`, high drift pada `realized_vol_shift1` dan `spread_shock`, risk multiplier `0.7`.
  - Daily report contoh menunjukkan banyak sinyal berakhir `NO TRADE` karena confidence gate (contoh `0.41 < 0.57`).
  - Dokumentasi `docs/*.md` sebagian masih menyebut H1/scheduler lama; implementasi kode saat ini sudah M15 on-demand + moment alert.

- File log/artifact relevan terakhir:
  - `artifacts/run_20260528_091616/run_summary.json`
  - `artifacts/run_20260528_091616/stage_5/stage_5_best_config.json`
  - `artifacts/run_20260528_091616/stage_5/calibration_report.json`
  - `artifacts/run_20260528_091616/stage_5/meta_model_report.json`
  - `logs/integration_check.json`
  - `logs/runtime_risk.json`
  - `logs/retrain_log.jsonl`

## 9. QUICK START FOR NEW AI ASSISTANT
1. Project ini adalah trading assistant ML berbasis XGBoost untuk FX/metal, aktif default `XAUUSD M15` horizon 4 bar (~1 jam).
2. Pipeline training utama: `run_pipeline.py` -> stage1..6; model final di `artifacts/run_*/stage_5/xgb_model.joblib`.
3. Live flow ada di `stage_9_live_demo.py`, trigger via Telegram command `/analisa` (on-demand, bukan scheduler periodik).
4. Fitur dibangun di `utils/gbpjpy_features.py`; semua rolling pakai shift(1) untuk cegah leakage.
5. Label utama 3 kelas (`target`: FLAT/UP/DOWN) + label risk-aware (`tp_sl_outcome`, MFE/MAE) dibuat di `stage_2_labeling.py`.
6. Stage5 sudah include profit-aware Optuna objective, calibration hybrid, meta-filter XGB binary, regime models, dan feature pruning.
7. NO TRADE dipicu oleh side none/flat-dominant, confidence di bawah min_conf, atau meta-filter reject.
8. Eksekusi order MT5 tidak otomatis penuh; perlu konfirmasi user `ya/tidak` di `utils/telegram_bot.py` lalu `utils/mt5_execution.py`.
9. Drift monitor ada di `retrain_scheduler.py`; cek `logs/runtime_risk.json` (saat ini HIGH_DRIFT).
10. Fokus improvement saat ini: turunkan ECE calibration dan naikkan precision gain meta-filter agar lolos target kualitas sinyal.
