# Dokumentasi Sistem Trading XAUUSD Berbasis Machine Learning

**Tujuan dokumen:** Penjelasan teknis dan konseptual untuk penyusunan skripsi akademik.  
**Run evaluasi referensi:** `artifacts/run_20260602_005304` (aktif: `artifacts/active_run.txt`)  
**Tanggal penyusunan:** Juni 2026  

---

## BAB 1 — Pendahuluan Sistem

### 1.1 Gambaran Umum

Sistem ini adalah **platform trading semi-otomatis** untuk pair **XAUUSD** (emas vs dolar AS) pada timeframe **M15** (15 menit). Secara konseptual, sistem menjawab pertanyaan: *“Apakah harga emas cenderung naik, turun, atau relatif datar dalam 3 jam ke depan?”* Jawaban statistik itu kemudian difilter, digabung dengan sentimen berita, dan disampaikan ke trader melalui **Telegram**.

**Apa yang dilakukan sistem?**

1. **Offline:** Mengunduh/membersihkan data historis, membuat label, melatih model XGBoost, menentukan threshold keputusan, dan menyimpan artifact model.
2. **Online:** Setiap bar M15 baru, sistem mengambil data dari MetaTrader 5, menghitung fitur, menjalankan inferensi, lalu mengirim sinyal atau notifikasi NO TRADE.
3. **Eksekusi:** Order **tidak** otomatis penuh — user harus membalas **`ya`** di Telegram; baru kemudian order market dengan SL/TP dikirim ke MT5.
4. **Monitoring:** Thread terpisah memantau **posisi MT5 nyata** sampai close, lalu mengirim notifikasi profit/loss dan balance.

**Untuk apa sistem dibuat?**

- Mendisiplinkan keputusan trading dengan aturan yang konsisten (bukan emosi).
- Menggabungkan **prediksi kuantitatif** (ML) dengan **konteks berita** (Gemini AI).
- Menjaga validitas penelitian lewat **purged walk-forward** dan meta-filter precision.
- Mendukung operasi harian: moment alert, `/analisa` on-demand, `/status`, `/trades`, `/akun`.

**Keunggulan dibanding trading manual**

| Aspek | Manual | Sistem ini |
|--------|--------|------------|
| Konsistensi aturan | Bergantung mood | Threshold & filter tertulis di config |
| Kecepatan analisa | Menit–jam | Detik setelah bar + delay 2 menit |
| Dokumentasi sinyal | Sering tidak ada | Log CSV + Telegram |
| Risk sizing | Ad hoc | Tier confidence + drift multiplier |
| Validasi strategi | Sering overfit | Walk-forward + holdout 15% |

Analogi sederhana: sistem ini seperti **asisten analis** yang setiap 15 menit menyiapkan laporan singkat; trader tetap menjadi **pilot** yang menyetujui lepas landas (`ya`).

### 1.2 Arsitektur Sistem

```
Data OHLCV MT5 / CSV
        ↓
Stage 1: Data Preparation (cleaning, spread, vol)
        ↓
Stage 2: Labeling (UP / DOWN / FLAT + TP/SL outcome)
        ↓
Stage 3: Feature Engineering (11 fitur aktif, shift anti-leakage)
        ↓
Stage 4: Purged Walk-Forward (5 fold, embargo)
        ↓
Stage 5: XGBoost Training + Optuna + Meta-filter
        ↓
Stage 6: Decision Logic (threshold + signal rate 25–50%)
        ↓
Stage 9: Live Inference (stage_9_live_demo.py + stage9_service.py)
        ↓
Telegram Bot → Konfirmasi User → MT5 Execution → Position Monitor
```

Diagram lengkap: lihat [`docs/DIAGRAM_ARSITEKTUR.md`](DIAGRAM_ARSITEKTUR.md).

**Orkestrator:** `run_pipeline.py` menjalankan Stage 1–6 berurutan, menulis `run_summary.json`, dan memperbarui pointer run aktif.

### 1.3 Teknologi yang Digunakan

| Teknologi | Versi / catatan | Peran dalam sistem |
|-----------|-----------------|-------------------|
| **Python** | 3.11 (lingkungan pengembangan Windows) | Bahasa utama seluruh pipeline |
| **pandas / numpy** | ≥2.0 / 1.26.4 | Manipulasi OHLCV & fitur |
| **XGBoost** | ≥2.0 | Klasifikasi 3 kelas (FLAT, UP, DOWN) |
| **scikit-learn** | 1.3.2 (dibatasi <1.5) | Metrik, kalibrasi, meta-model |
| **Optuna** | ≥3.6 | Bayesian hyperparameter tuning |
| **MetaTrader 5** | Paket `MetaTrader5` | Data historis, tick/bar live, eksekusi order |
| **Telegram Bot API** | via `requests` | Notifikasi & konfirmasi user |
| **Gemini AI** | `gemini-2.5-flash` (+ fallback) | Sentimen headline berita emas/USD |
| **PyYAML** | ≥6.0 | Konfigurasi `pipeline.yaml` |
| **joblib** | ≥1.3 | Serialisasi model |
| **feedparser** | ≥6.0 | RSS berita (ForexFactory, dll.) |

---

## BAB 2 — Data dan Preprocessing

### 2.1 Sumber Data

| Atribut | Nilai (run aktif) |
|---------|-------------------|
| Symbol | XAUUSD |
| Timeframe | M15 (15 menit) |
| Bar mentah Stage 1 | **49.265** bar |
| Periode | **2024-04-16** s/d **2026-05-28** (UTC) |
| Bar training (`keep_for_training=1`) | **41.546** bar |
| Spread median | 112 points (cap kuantil 99% = 252) |

Data diambil dari `data/xauusd_m15.csv` atau diunduh otomatis dari MT5 (`stage_1.auto_fetch_mt5: true`, hingga 200.000 bar).

### 2.2 Data Cleaning (Stage 1)

File: `stage_1_data.py`.

Proses utama:

1. **Validasi kolom** `time, open, high, low, close, spread`.
2. **Timezone UTC** — semua timestamp diseragamkan.
3. **Gap detection** — gap > 60 menit ditandai (546 bar gap pada run aktif).
4. **Forward fill** — missing OHLC maksimal 2 bar berturut-turut.
5. **Spread cap** — spread di atas kuantil 99% dipangkas untuk mengurangi anomali likuiditas.
6. **Weekend filter** — opsi `exclude_weekend_bars` (aktif di config).
7. **Fitur awal anti-leakage** — `realized_vol_shift1`, `hour_of_day`, `day_of_week` dihitung dengan rolling **shift(1)**.

Tujuan cleaning: data yang dipakai model mencerminkan kondisi pasar yang bisa diulang saat live, tanpa lonjakan artefak spread.

### 2.3 Labeling Sistem (Stage 2)

File: `stage_2_labeling.py`. Target utama: arah harga **3 kelas** setelah horizon \(H\) bar.

**Rumus log-return forward:**

\[
r_{t \to t+H} = \ln\left(\frac{close_{t+H}}{close_t}\right)
\]

**Aturan kelas** (threshold \(\tau\) = `flat_return_threshold`):

| Kelas | Kondisi |
|-------|---------|
| **UP** | \(r_{t \to t+H} > \tau\) |
| **DOWN** | \(r_{t \to t+H} < -\tau\) |
| **FLAT** | \(|r_{t \to t+H}| \le \tau\) |

**Parameter aktual:**

- \(H\) = **12 bar** → pada M15 = **3 jam**
- \(\tau\) = **0,0011** (~0,11% pergerakan)

**Distribusi kelas** (bar yang lolos filter, dari `stage_2_metadata.json`):

| Kelas | Jumlah bar |
|-------|------------|
| FLAT | 12.644 |
| UP | 15.857 |
| DOWN | 13.045 |
| DROP_FILTER | 7.719 (tidak dipakai training) |

Fraksi data training: **84,3%** dari seluruh bar.

### 2.4 TP/SL Outcome Labeling

Label tambahan **`tp_sl_outcome`** mengevaluasi kualitas setup risiko (bukan target klasifikasi utama):

- **SL distance** \(= 1{,}5 \times \mathrm{ATR}(14)\)
- **TP distance** \(= 1{,}8 \times\) SL distance (RR = 1,8)

Simulasi pada bar forward: mana yang tersentuh lebih dulu — TP atau SL.

| Outcome | Jumlah (kept rows) |
|---------|-------------------|
| TP_FIRST | 13.960 |
| SL_FIRST | 4.416 |
| NO_HIT_OR_FLAT | 23.170 |

**TP_FIRST rate** (hanya yang kena TP atau SL):  
\(13960 / (13960 + 4416) \approx \mathbf{76{,}0\%}\) — ini pada **semua bar labeled**, bukan hanya saat model mengeluarkan sinyal.

---

## BAB 3 — Feature Engineering

### 3.1 Fitur Aktif (11 fitur)

Sumber: `artifacts/run_20260602_005304/stage_5/feature_registry.json`.

| Fitur | Deskripsi | Relevansi untuk Gold |
|-------|-----------|---------------------|
| **daily_range_pos** | Posisi close dalam range harian (0–1), shift(1) | Mean-reversion intraday emas |
| **distance_to_ema50** | Jarak relatif harga ke EMA(50) | Tren jangka menengah |
| **distance_to_ema20** | Jarak ke EMA(20) | Tren jangka pendek |
| **bar_momentum** | Momentum bar terakhir (shift) | Kontinuitas pergerakan |
| **hour_of_day** | Jam UTC | Pola likuiditas sesi Asia/London/NY |
| **realized_vol_shift1** | Volatilitas realized rolling | Regime vol — penting untuk emas |
| **rsi_zscore** | RSI dinormalisasi z-score | Overbought/oversold relatif |
| **atr_zscore** | ATR dinormalisasi z-score | Regime volatilitas absolut |
| **hurst_proxy** | Proxy persistensi vs mean-reversion | Karakter jangka pendek emas |
| **asian_range_breakout_up** | Breakout atas range sesi Asia | Emas sering bereaksi saat London open |
| **asian_range_breakout_down** | Breakout bawah range Asia | Sisi bearish struktur intraday |

Fitur dengan importance negatif (mis. `log_return`, `session_ny`) **tidak aktif** setelah feature validation Stage 5.

### 3.2 Anti-Lookahead Bias

**Lookahead bias** = model secara tidak sah “melihat masa depan” lewat fitur.

Contoh salah: menghitung RSI pada bar \(t\) memakai close bar \(t\) lalu memprediksi return dari \(t\) ke \(t+H\) — informasi close \(t\) sudah termasuk dalam label horizon yang dimulai dari \(t\).

**Solusi:** Semua rolling statistic memakai **`.shift(1)`** — fitur di bar \(t\) hanya memakai data sampai bar \(t-1\).

Analogi: Anda membuat keputusan di **buka** bar baru, hanya dengan informasi yang sudah tersedia **setelah** bar sebelumnya selesai.

### 3.3 Fitur XAUUSD-Specific

**Asian range breakout**  
Range ditentukan dari high/low pada jam **02–09 UTC** (proxy sesi Asia). Breakout = close menembus range tersebut. Emas sering mengalami ekspansi volatilitas saat transisi ke London.

**London open proxy** (tersedia di registry, tidak masuk 11 fitur aktif terakhir)  
Mengkodekan jam sekitar pembukaan London — likuiditas dan arah struktur sering berubah.

**daily_range_pos**  
Jika harga berada di ekstrem atas range harian, secara statistik ada kecenderungan pullback (mean-reversion); sebaliknya di ekstrem bawah.

Implementasi detail: `utils/gbpjpy_features.py` (dipanggil via `utils/xauusd_features.py`).

---

## BAB 4 — Model Machine Learning

### 4.1 Algoritma: XGBoost

**Gradient Boosting** membangun pohon regresi/klasifikasi secara berurutan; setiap pohon baru memperbaiki residual error ensemble sebelumnya.

**XGBoost (Extreme Gradient Boosting)** menambah regularisasi dan struktur data efisien sehingga cocok untuk:

- Data tabular time-series yang sudah difiturkan
- Kelas tidak seimbang (dengan `scale_pos_weight` / inverse frequency)
- Interpretasi importance fitur

**Perbandingan singkat:**

| Algoritma | Kelebihan | Kekurangan untuk proyek ini |
|-----------|-----------|----------------------------|
| Random Forest | Robust, mudah | Kurang fine-tune probabilitas |
| Neural Network | Fleksibel | Butuh data besar, interpretasi sulit |
| **XGBoost** | Kuat di tabular, cepat | Perlu validasi temporal ketat |

### 4.2 Validasi: Purged Walk-Forward

**Masalah random split:** Bar trading berkorelasi dalam waktu; split acak membuat model “melihat” pola masa depan lewat fitur yang overlap dengan label horizon.

**Walk-Forward:** Training pada window masa lalu → test pada window berikutnya → geser, 5 fold (`stage_4.n_splits`).

**Purging & Embargo:**  
Setelah training berakhir, sistem memberi **embargo** minimal `horizon_bars` (diperkuat di kode: `embargo >= horizon`) agar label di perbatasan train/test tidak overlap.

```
Timeline ─────────────────────────────────────────────►

Fold k:
[======== Train (expand) ========][embargo][== Test ==]
                                 ↑
                          purge overlap label H
```

File: `stage_4_validation.py`.

### 4.3 Hyperparameter Optimization: Optuna

**Bayesian Optimization (TPE)** mencari kombinasi hyperparameter yang memaksimalkan **objective komposit** (bukan hanya F1):

- 35% expectancy simulasi trading
- 20% Sharpe simulasi
- 25% precision directional
- 20% F1 directional

Parameter yang dioptimasi (contoh): `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`.

`n_trials = 48` per run pipeline.

### 4.4 Kalibrasi Probabilitas

Probabilitas raw tree-based model sering **over/under-confident**.

| Metode | ECE (holdout) | Dipilih |
|--------|---------------|---------|
| Raw (none) | **0,146** | **Ya** |
| Isotonic | 0,151 | Tidak |
| Temperature | 0,347 | Tidak |

Sumber: `calibration_report.json`. Sistem live memakai probabilitas **raw** karena kalibrasi tidak menurunkan ECE pada run ini.

**ECE** mengukur rata-rata |confidence − accuracy| per bin — semakin kecil semakin terkalibrasi.

### 4.5 Meta-Filter (Two-Stage Model)

**Meta-model** memprediksi apakah sinyal directional dari model utama berkualitas tinggi (proxy profitabilitas).

Input meta: `p_flat`, `p_up`, `p_down`, spread, `atr_zscore`, sesi London/NY, dll.

**Hasil run aktif** (`meta_model_report.json`):

| Arah | Precision baseline | Setelah filter | Gain |
|------|-------------------|----------------|------|
| UP | 0,577 | 0,773 | **+19,7%** |
| DOWN | 0,562 | 0,690 | **+12,8%** |

Threshold meta: **0,41**. Trade yang lolos: **1.231** dari **6.232** prediksi holdout.

Di live (`apply_meta_filter` di `stage_9_live_demo.py`), meta-filter **wajib** — skor di bawah threshold → NO TRADE.

---

## BAB 5 — Sistem Pengambilan Keputusan

### 5.1 Decision Logic (Stage 6)

File: `stage_6_ensemble.py`. Mode: **directional_thresholds**.

| Parameter | Nilai run aktif |
|-----------|-----------------|
| `conf_up` | 0,70 |
| `conf_down` | 0,65 |
| `abstain_tau` | 0,35 |

**Aturan intuitif:**

- **BUY** jika \(P(UP) \ge conf\_up\), \(P(FLAT) < abstain\_tau\), dan \(P(UP) \ge P(DOWN)\).
- **SELL** jika \(P(DOWN) \ge conf\_down\), \(P(FLAT) < abstain\_tau\), dan \(P(DOWN) > P(UP)\).
- Selain itu → FLAT / abstain.

**Signal rate constraint:** target **25%–50%** bar menghasilkan sinyal directional (`min_signal_rate_target`, `max_signal_rate_target`). Pada holdout: **49,7%**.

### 5.2 Sentiment Analysis (Gemini AI)

File: `utils/gemini_client.py`.

1. Headline diambil (NewsAPI + RSS, query emas/Fed/USD).
2. Gemini menilai sentimen skala **−2 … +2** (sangat bearish → sangat bullish untuk emas).
3. Retry 3× + cache 60 menit + fallback keyword jika API gagal.

Integrasi: sentimen mempengaruhi rekomendasi teks dan bisa memicu **konflik** dengan ML.

### 5.3 Multi-Layer Filter

Urutan penapisan keputusan live:

1. **Model confidence** — probabilitas vs `conf_up` / `conf_down` / `min_confidence` trade plan.
2. **Meta-filter** — `meta_score >= 0.41`.
3. **Sentiment conflict** — rekomendasi mengandung KONFLIK / NO TRADE.
4. **Drift monitor** — HIGH_DRIFT mengurangi lot; CRITICAL memblokir eksekusi.

### 5.4 Conflict Detection

**Konflik** terjadi bila sinyal ML BUY kuat tetapi sentimen ≤ −1 (bearish), atau SELL kuat dengan sentimen ≥ +1.

**Hasil:** `no_trade_reasons` berisi *"konflik ML vs sentiment"* → user menerima NO TRADE, bukan order.

Contoh narasi: Model memprediksi SELL (emas turun) tetapi headline menekankan Fed hawkish + safe-haven bid — Gemini bullish → sistem menolak eksekusi.

---

## BAB 6 — Risk Management

### 6.1 ATR-Based Stop Loss

\[
SL_{distance} = m_{sl} \times \mathrm{ATR}(14), \quad m_{sl} = 1{,}5
\]

Entry BUY: \(SL = entry - SL_{distance}\), \(TP = entry + RR \times SL_{distance}\).

### 6.2 Risk-Reward Ratio

**RR = 1,8** — setiap 1R risiko, target profit 1,8R jika TP tersentuh.

**Winrate minimum agar break-even** (tanpa biaya):

\[
w_{min} = \frac{1}{RR + 1} = \frac{1}{2{,}8} \approx 35{,}7\%
\]

Dengan winrate lebih tinggi, expectancy positif.

### 6.3 Confidence-Based Lot Sizing

| Tier | Syarat confidence | Multiplier |
|------|-------------------|------------|
| HIGH | ≥ 0,70 | 1,5× base (0,01 → 0,015) |
| MEDIUM | ≥ 0,55 | 1,0× |
| LOW | ≥ 0,45 | 0,7× |

Sentiment adjustment: faktor 0,8 (konflik ringan) atau 1,1 (searah kuat).

### 6.4 Drift Monitor

**PSI (Population Stability Index)** membandingkan distribusi fitur training lama vs data terbaru.

- PSI > **0,2** pada fitur → dianggap drift tinggi pada fitur tersebut.
- Jika **≥ 2 fitur** drift → status **`HIGH_DRIFT`** (`retrain_scheduler.py`).
- Respon live: `risk_multiplier = 0,7` (lot −30%); retrain otomatis jika interval ≥ 6 jam.
- **`CRITICAL`**: eksekusi diblokir total.

File runtime: `logs/runtime_risk.json`.

---

## BAB 7 — Sistem Live Trading

### 7.1 Arsitektur Live System

Komponen utama: `scripts/stage9_service.py` + `stage_9_live_demo.py`.

| Fitur | Implementasi |
|-------|----------------|
| Bar-aware | Analisa hanya saat bar M15 baru (`is_new_bar`) |
| Delay | 120 detik setelah open bar (`bar_analysis_delay_seconds`) |
| Trading hours | `utils/trading_hours.py` — Sen 06:00 WIB s/d Sab 05:00 WIB |
| Weekend | Loop sleep, notifikasi pasar tutup sekali |

### 7.2 Alur Eksekusi

```
Bar M15 baru
    → tunggu 120 detik
    → ambil ~300 bar MT5
    → feature engineering (sama seperti training)
    → XGBoost predict_proba
    → meta-filter (wajib)
    → Gemini sentiment
    → build_trade_plan (SL/TP ATR)
    → Telegram: sinyal STRONG/BUY/SELL atau NO TRADE
    → user "ya" → place_market_order_from_plan()
    → position_monitor thread → notifikasi close MT5
```

**Penting:** Monitor TP/SL memakai **`positions_get(ticket)`**, bukan simulasi sentuh level harga — mencegah notifikasi palsu tanpa posisi.

### 7.3 Auto-Retrain

| Kondisi | Nilai |
|---------|-------|
| Trigger | HIGH_DRIFT / CRITICAL (dari `runtime_risk.json`) |
| Interval minimum | 6 jam |
| Waktu diizinkan | Bukan weekend; bukan dini hari UTC (<02:00) |
| Proses | Subprocess `run_pipeline.py`, timeout 3600s |
| Setelah sukses | Pointer run baru + restart service |

### 7.4 Telegram Bot Interface

| Command | Fungsi |
|---------|--------|
| `/analisa` | Analisa on-demand + prompt konfirmasi `ya` |
| `/status` | Model aktif, drift, run_id |
| `/trades` | Ringkasan 7 hari (CSV + backup MT5 history) |
| `/akun` | Balance, equity, posisi terbuka MT5 |

File: `utils/telegram_bot.py`.

---

## BAB 8 — Evaluasi Performa

### 8.1 Metrik

| Metrik | Arti | Kenapa dipakai |
|--------|------|----------------|
| **Macro F1** | Rata-rata F1 per kelas (bobot sama) | Kelas FLAT/UP/DOWN imbalanced |
| **Balanced Accuracy** | Rata-rata recall per kelas | Tidak tertipu accuracy mayoritas FLAT |
| **ECE** | Kalibrasi probabilitas | Kepercayaan pada skor % |
| **TP_FIRST rate** | Proporsi simulasi TP sebelum SL | Proxy langsung profitability setup |
| **Signal rate** | % bar dengan sinyal BUY/SELL | Mengontrol over-trading |

### 8.2 Hasil Backtest (Holdout 15%)

Run: `run_20260602_005304`.

| Metrik | Nilai |
|--------|-------|
| Stage 5 holdout macro F1 | **0,463** |
| Stage 5 holdout balanced accuracy | **0,472** |
| Stage 6 holdout accuracy | **0,434** |
| Stage 6 holdout balanced accuracy | **0,460** |
| Signal rate holdout | **49,7%** |
| TP_FIRST rate (directional only) | **40,5%** |
| SL_FIRST rate (directional only) | **11,2%** |
| Unresolved rate (directional) | **48,2%** |
| Meta precision gain UP | **+19,7%** |
| Meta precision gain DOWN | **+12,8%** |

**Catatan:** Accuracy klasifikasi arah ≠ profit trading; TP_FIRST dan meta-filter lebih dekat ke kualitas eksekusi.

### 8.3 Ekspektasi Matematis

Untuk trade directional yang **benar-benar** kena TP atau SL pada holdout:

\[
\text{winrate} = \frac{TP\_FIRST}{TP\_FIRST + SL\_FIRST} = \frac{0{,}405}{0{,}405 + 0{,}112} \approx 0{,}783
\]

\[
\text{Expectancy (R)} = w \times RR - (1-w) \times 1 = 0{,}783 \times 1{,}8 - 0{,}217 \approx \mathbf{+1{,}19R}
\]

Ini **teoretis pada label simulasi**, belum termasuk spread, slippage, dan 48% unresolved.

Pada **seluruh bar labeled** (bukan hanya saat sinyal):

\[
13960 / (13960+4416) \approx 76{,}0\% \text{ TP\_FIRST}
\]

### 8.4 Perbandingan dengan Baseline

| Skenario | Karakter |
|----------|----------|
| Random entry | Winrate ~50% → expectancy negatif dengan RR 1,8 tanpa edge |
| Tanpa meta-filter | Lebih banyak sinyal, precision UP/DOWN lebih rendah |
| Dengan meta-filter | Sinyal lebih sedikit, precision +13–20% (holdout) |

---

## BAB 9 — Kesimpulan Teknis

### 9.1 Kontribusi Sistem

1. **Integrasi ML + sentiment** untuk emas intraday dengan filter konflik berita.
2. **Purged walk-forward** mengurangi leakage temporal pada label horizon 12 bar.
3. **Two-stage model** (XGBoost + meta-filter) meningkatkan precision sinyal.
4. **Operasi live terstruktur:** bar-aware timing, human confirmation, monitor posisi MT5 nyata.
5. **Auto-retrain** responsif terhadap drift distribusi (PSI).

### 9.2 Keterbatasan

- Model terikat distribusi historis 2024–2026; regime krisis baru belum tentu tercakup.
- Sentiment bergantung API Gemini/News — gagal API → fallback keyword lebih kasar.
- **Unresolved ~48%** pada sinyal directional holdout: banyak setup tidak kena TP/SL dalam horizon simulasi.
- Eksekusi live belum sepenuhnya otomatis — bergantung respons user Telegram.
- Validasi profit riil account perlu dilanjutkan di luar metrik klasifikasi.

### 9.3 Pengembangan Selanjutnya

- Multi-symbol (BTCUSD, EURUSD) — parameter `per_symbol` sudah ada di config.
- Data tick / order book untuk fitur mikrostruktur.
- Reinforcement learning untuk threshold dinamis per regime.
- Update `runtime_risk.json` langsung dari batch fitur live (bukan hanya pasca-pipeline).
- Penutupan otomatis opsional setelah timeout 4 jam dengan kebijakan risk tetap.

---

## Lampiran — File Penting

| Topik | Path |
|-------|------|
| Konfigurasi | `configs/pipeline.yaml` |
| Parameter tabel | `docs/PARAMETER_KONFIGURASI.md` |
| Diagram | `docs/DIAGRAM_ARSITEKTUR.md` |
| Istilah | `docs/GLOSSARY.md` |
| Service Windows | `docs/OPERASIONAL_WINDOWS.md` |

---

*Dokumen ini dihasilkan dari analisis kode sumber dan metadata run `run_20260602_005304`. Perbarui angka evaluasi setelah setiap `run_pipeline.py` sukses.*
