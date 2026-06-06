# Contoh Laporan Telegram — XAUUSD H1

Template output bot untuk skripsi / dokumentasi.  
Sistem aktif: **XAUUSD**, timeframe **H1**, horizon **1 jam ke depan**.

---

**XAUUSD Sinyal H1 — 2026-05-25 19:00 WIB**

*Target: pergerakan 1 jam ke depan (1 bar H1)*

**Model XGBoost (bar acuan terakhir)**
- Prediksi kelas: **DOWN**
- P(FLAT)=31.7% | P(UP)=32.0% | P(DOWN)=36.4%
- Bias arah (UP−DOWN): **-4.4%** → cond. **DOWN**

**Sentimen berita (AI)**
- Skor: **+1** (Bullish)
- Meningkatnya ketergantungan pada Dolar AS akibat perang Iran mengindikasikan sentimen risk-off global, yang cenderung menekan GBP sementara mendukung JPY sebagai safe-haven, sehingga GBPJPY diproyeksikan melemah.

**Rekomendasi:** **SELL**
- Mode: **Directional**

**Catatan:**
- Horizon prediksi: 1 jam ke depan (1 bar H1) (dari close bar acuan).
- Kelas terbesar DOWN (36%).
- Bias UP−DOWN: -4.4% | skor gabungan: +0.31
- Berita: mendukung sisi bullish.

_Headlines:_
- Iran War Drives Rise in Dollar Reliance for Global Trade — Financial Post
- Developments related to US-Iran negotiations, oil prices to dictate sentiment in markets: Analysts — BusinessLine
- Bitcoin's Fed cut trade flips as bond market turns into the risk — CryptoSlate
- … +3 lainnya

---

## Variasi rekomendasi (konsensus)

| Rekomendasi | Arti singkat |
|-------------|-------------|
| STRONG BUY / SELL | Sinyal kuat (prob + sentimen selaras) |
| BUY / SELL | Sinyal sedang |
| WEAK BUY / WEAK SELL | Bias arah condong, FLAT masih dominan |
| NEUTRAL / ABSTAIN | Tidak ada edge arah yang cukup kuat |

---

*File ini menggantikan template lama GBPJPY D1. Sesuaikan angka probabilitas dengan run model terbaru di `artifacts/run_*/stage_6/`.*
