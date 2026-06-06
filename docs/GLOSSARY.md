# Glossary Istilah Teknis

Daftar istilah yang dipakai dalam sistem trading XAUUSD M15 ini. Penjelasan ditujukan untuk pembaca skripsi yang belum familiar dengan machine learning atau trading algoritmik.

---

## A

**Abstain / Abstain Tau**  
Keputusan model untuk *tidak* mengambil sisi BUY atau SELL karena probabilitas kelas FLAT cukup tinggi. Parameter `abstain_tau` (misalnya 0,35) membatasi nilai maksimum P(FLAT) agar sinyal directional boleh keluar.

**ATR (Average True Range)**  
Indikator volatilitas: rata-rata jarak pergerakan harga “nyata” per bar, memperhitungkan gap. Di sistem ini, jarak stop loss sering dihitung sebagai `sl_atr_multiplier × ATR(14)`.

**Artifact**  
File hasil training (model `.joblib`, threshold, registry fitur) yang disimpan di folder `artifacts/run_*` dan dipakai ulang saat live trading.

---

## B

**Balanced Accuracy**  
Rata-rata *recall* per kelas. Lebih adil daripada *accuracy* biasa ketika kelas tidak seimbang (banyak FLAT vs sedikit UP/DOWN).

**Bar (Candlestick)**  
Satu periode OHLCV (open, high, low, close, volume/spread). Untuk M15, satu bar = 15 menit perdagangan.

**Bayesian Optimization (Optuna)**  
Metode mencari hyperparameter terbaik dengan mencoba kombinasi secara cerdas (bukan grid search brutal), mempelajari trial sebelumnya untuk memilih trial berikutnya.

---

## C

**Calibration (Kalibrasi Probabilitas)**  
Penyesuaian probabilitas model agar angka 70% benar-benar terjadi ~70% di data baru. Diukur dengan **ECE**.

**Confidence Tier**  
Kategori kekuatan sinyal (HIGH / MEDIUM / LOW) berdasarkan skor probabilitas; mempengaruhi ukuran lot.

**Cooldown (Trading)**  
Periode setelah ada posisi terbuka di mana sistem tidak mengirim sinyal baru sampai posisi close (TP/SL/manual).

---

## D

**Directional Label**  
Kelas prediksi arah harga: UP (naik), DOWN (turun), atau FLAT (pergerakan kecil).

**Drift (Distribusi)**  
Perubahan statistik data live dibanding data training. Sistem memakai **PSI**; status **HIGH_DRIFT** memicu pengurangan lot atau retrain.

**Deal (MT5)**  
Catatan eksekusi di MetaTrader 5; deal dengan `entry = OUT` menandakan penutupan posisi.

---

## E

**ECE (Expected Calibration Error)**  
Rata-rata selisih antara probabilitas prediksi dan frekuensi kejadian aktual per bin. Semakin kecil semakin baik.

**Embargo**  
Jarak waktu (dalam jumlah bar) antara akhir data training dan awal data test pada walk-forward, mencegah label horizon “menembus” ke training.

**Ensemble**  
Kombinasi beberapa model atau tahap keputusan; di sini Stage 6 menggabungkan probabilitas XGBoost dengan aturan threshold.

**Expectancy**  
Nilai harian rata-rata profit per trade dalam satuan R (risk unit):  
`E = (winrate × RR) − (1 − winrate) × 1`.

---

## F

**Feature Engineering**  
Proses membuat variabel input (fitur) dari harga mentah, misalnya jarak ke EMA atau posisi dalam range harian.

**FLAT (Kelas)**  
Label ketika pergerakan log-return ke depan berada di antara −threshold dan +threshold.

**Fold (Cross-Validation)**  
Satu putaran train/test pada walk-forward; proyek ini memakai 5 fold expanding window.

---

## G

**Gradient Boosting**  
Teknik ML yang menambah pohon keputusan kecil secara berurutan; setiap pohon baru memperbaiki kesalahan pohon sebelumnya.

**Gemini AI**  
Model bahasa Google yang menganalisis headline berita untuk skor sentimen (−2 hingga +2) terhadap emas/USD.

---

## H

**Holdout**  
Bagian data paling akhir (15%) yang tidak dipakai saat tuning, hanya untuk evaluasi final.

**Horizon (H)**  
Berapa bar ke depan label dihitung. Di config: `horizon_bars = 12` → 3 jam pada M15.

**Human-in-the-Loop**  
User harus membalas `ya` di Telegram sebelum order dikirim ke MT5.

---

## L

**Lookahead Bias**  
Kesalahan memakai informasi masa depan saat memprediksi masa lalu/lalu. Dicegah dengan `shift(1)` pada fitur rolling.

**Log-Return**  
`ln(close_t+H / close_t)` — bentuk return yang lebih stabil secara statistik untuk threshold labeling.

**Lot**  
Ukuran volume kontrak di MT5 (misalnya 0,01 lot).

---

## M

**Macro F1**  
Rata-rata F1-score per kelas (UP, DOWN, FLAT) dengan bobot sama; mengukur performa multiclass secara adil.

**Meta-Filter**  
Model sekunder (stacking) yang memprediksi apakah sinyal dari model utama layak dieksekusi; meningkatkan precision.

**M15**  
Timeframe 15 menit.

**Moment Alert**  
Notifikasi otomatis setiap bar M15 (setelah delay 2 menit) berisi sinyal trade atau ringkasan NO TRADE.

**MT5 (MetaTrader 5)**  
Platform broker untuk data harga, eksekusi order, dan monitoring posisi.

---

## N

**NO TRADE**  
Keputusan tidak membuka posisi karena filter (meta, drift, konflik sentiment, confidence rendah, dll.).

---

## O

**OHLCV**  
Open, High, Low, Close, Volume (dan spread di data FX).

**Optuna**  
Library hyperparameter tuning berbasis Bayesian/TPE.

**Order Ticket**  
Nomor unik order/posisi di MT5; dipakai monitor posisi sampai close.

---

## P

**Precision**  
Dari semua prediksi “profitabel” meta-filter, berapa persen yang benar-benar layak — penting untuk mengurangi false signal.

**PSI (Population Stability Index)**  
Mengukur pergeseran distribusi fitur antara baseline (training) vs data terkini. PSI > 0,2 pada banyak fitur → **HIGH_DRIFT**.

**Purged Walk-Forward**  
Validasi time series: training hanya di masa lalu, test di masa depan, dengan **purge/embargo** agar label horizon tidak bocor.

---

## R

**Risk-Reward (RR)**  
Rasio jarak TP terhadap SL. Config: `tp_rr = 1.8` artinya target profit 1,8× risiko 1R.

**Recall**  
Proporsi kasus positif yang berhasil terdeteksi model.

**Retrain**  
Menjalankan ulang `run_pipeline.py` untuk model baru; dipicu drift tinggi (min. interval 6 jam).

---

## S

**Sentiment Score**  
Skor integer −2 … +2 dari analisis berita Gemini; bisa memicu **KONFLIK** dengan sinyal ML.

**Signal Rate**  
Persentase bar yang menghasilkan prediksi BUY/SELL (bukan FLAT/abstain).

**SL_FIRST / TP_FIRST**  
Label risk: apakah stop loss atau take profit tersentuh lebih dulu dalam horizon simulasi setelah entry.

**Spread**  
Selisih bid-ask; spread ekstrem dapat membuat bar dibuang di Stage 2.

**Stage (1–9)**  
Tahapan pipeline: data → label → fitur → validasi → train → decision → (live stage 9).

**STRONG BUY / STRONG SELL**  
Sinyal dengan probabilitas directional sangat tinggi; bisa memakai TP RR lebih besar (`tp_rr_strong`).

---

## T

**Temperature Scaling**  
Metode kalibrasi: membagi logit probabilitas dengan parameter suhu per kelas.

**Threshold (conf_up / conf_down)**  
Batas probabilitas minimum untuk mengeluarkan sinyal BUY atau SELL.

**Time Series Split**  
Pembagian data berurutan waktu; tidak boleh diacak seperti klasifikasi i.i.d.

**TP/SL**  
Take Profit / Stop Loss — level harga target untung dan batas rugi.

**Two-Stage Model**  
Arsitektur: model utama (arah) + meta-filter (kualitas sinyal).

---

## U

**Unresolved (TP/SL)**  
Dalam simulasi labeling, harga tidak mencapai TP maupun SL dalam horizon → `NO_HIT_OR_FLAT`.

**UP / DOWN**  
Kelas naik/turun berdasarkan log-return forward vs threshold.

---

## W

**Walk-Forward Validation**  
Metode evaluasi: model dilatih pada window masa lalu, diuji pada window berikutnya, digeser berulang.

**Winrate (TP_FIRST)**  
`TP_FIRST / (TP_FIRST + SL_FIRST)` pada trade yang benar-benar kena TP atau SL.

---

## X

**XAUUSD**  
Simbol trading emas (XAU) terhadap dolar AS (USD).

**XGBoost (Extreme Gradient Boosting)**  
Implementasi gradient boosting yang efisien; model utama klasifikasi 3 kelas dalam proyek ini.

---

## Z

**Z-Score**  
Nilai fitur dinormalisasi: (x − μ) / σ pada window rolling; memudahkan model membandingkan skala.
