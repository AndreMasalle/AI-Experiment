# AI Model Experiment & Evaluation — Sentiment Analysis Ulasan Pelanggan

Eksperimen perbandingan dua pendekatan AI untuk klasifikasi sentimen ulasan pelanggan e-commerce:
1. **Model Klasik Machine Learning** — TF-IDF + Logistic Regression (Scikit-learn)
2. **LLM API (Gemini)** — zero-shot prompting tanpa training

---

## 1. Problem Statement & Dataset

### 1.1 Objective
Membangun fitur otomatis untuk mengklasifikasikan sentimen ulasan pelanggan (positif/negatif) pada halaman produk e-commerce. Hasil eksperimen menjadi bahan pertimbangan tim dalam memilih pendekatan yang akan dikembangkan ke sistem produksi.

### 1.2 Target / Label
- **Label biner:** `positif` atau `negatif`
- **Input:** kolom `review_text`
- **Output:** prediksi sentimen per ulasan

### 1.3 Dataset
- **File:** `data/customer_reviews_sentiment.csv`
- **Jumlah baris:** 200
- **Kolom:** `review_id`, `product_name`, `review_text`, `sentiment`
- **Distribusi:** ~55% positif, ~45% negatif
- **Bahasa:** Indonesia (formal & informal)

### 1.4 Batasan & Asumsi
- Dataset kecil dan mengandung banyak `review_text` duplikat (~40–50 kalimat unik dari 200 baris).
- Untuk mencegah **data leakage**, split menggunakan **GroupShuffleSplit** dengan `review_text` sebagai group — baris dengan kalimat yang sama tidak dicampur antara train dan test.
- Eksperimen dijalankan pada **free tier Gemini API** — ada rate limit dan kemungkinan error 503/429.
- Diasumsikan setiap ulasan memiliki satu sentimen dominan dan label ground truth benar.

---

## 2. Ringkasan Eksperimen

### 2.1 Pendekatan 1 — Model Klasik (Scikit-learn)
- **Preprocessing:** TF-IDF (`lowercase=True`, `ngram_range=(1,2)`)
- **Model:** Logistic Regression (`max_iter=1000`)
- **Split:** GroupShuffleSplit 80/20 (test size = 41 baris)
- **Training:** < 5 detik
- **Inference:** < 1 detik untuk seluruh test set

### 2.2 Pendekatan 2 — LLM API (Gemini)
- **Prompt:** zero-shot, instruksi "balas HANYA satu kata: positif atau negatif"
- **Temperature:** 0.2 (dipilih agar output deterministik — task klasifikasi tidak butuh kreativitas)
- **Model:** `gemini-3.7-flash` (fallback: `3.6-flash`, `3.5-flash`)
- **Strategi:** paralelisasi dengan `ThreadPoolExecutor` (2 worker) + retry untuk 429/503 + fallback antar model
- **Waktu inference:** ~2 menit untuk 41 review
- **Normalisasi output:** handle output bahasa Indonesia & Inggris (`positif`/`positive`, `negatif`/`negative`)

---

## 3. Hasil Evaluasi & Perbandingan

### 3.1 Tabel Ringkasan

| Metrik | Model Klasik | LLM API (Gemini) |
|--------|--------------|------------------|
| Accuracy | 0.512 | **0.927** |
| Precision (weighted) | 0.59 | **0.94** |
| Recall (weighted) | 0.51 | **0.93** |
| F1-Score (weighted) | 0.52 | **0.93** |

### 3.2 Confusion Matrix

**Model Klasik:**

```
[[ 9 5]
[15 12]]
```

text
→ 15 false negative (review positif diprediksi negatif)

**LLM API (Gemini):**

```
[[14 0]
[ 3 24]]
```


→ Hanya 3 kesalahan (review positif diprediksi negatif)

### 3.3 Catatan Penting
Pada percobaan awal dengan split acak, model klasik mencapai **accuracy 1.00**. Setelah beralih ke **GroupShuffleSplit**, performanya turun ke **0.51**. Ini membuktikan bahwa skor sempurna sebelumnya adalah artefak data leakage — kalimat review yang sama muncul di train dan test. Hasil dengan GroupShuffleSplit lebih representatif.

---

## 4. Analisis Trade-off & Limitation

### 4.1 Performa
**LLM API (Gemini) jauh lebih unggul** di semua metrik dengan selisih accuracy ~41 poin persentase. LLM mampu generalisasi ke kalimat baru, sedangkan model klasik hanya menghafal pola dari training.

### 4.2 Trade-off Effort, Kecepatan, Biaya

| Aspek | Model Klasik | LLM API (Gemini) |
|-------|--------------|------------------|
| Effort implementasi | Perlu training + tuning | Tanpa training, cukup prompt |
| Kecepatan inference | Sangat cepat (< 1 detik) | ~2 menit untuk 41 review |
| Biaya | Gratis | Free tier terbatas; produksi berbayar per token |
| Skalabilitas | Sangat baik | Bergantung kuota API |
| Maintenance | Perlu retrain jika data berubah | Cukup update prompt |

### 4.3 Limitation

**Model Klasik:**
- Generalisasi rendah pada kalimat baru.
- Sangat bergantung pada kualitas & keragaman dataset.
- Tidak bisa menangani sarkasme, bahasa gaul, atau typo yang tidak muncul saat training.

**LLM API:**
- Output tidak selalu konsisten (perlu normalisasi EN/ID).
- Bergantung pada koneksi internet dan kestabilan server (503/429).
- Latency lebih tinggi.
- Biaya API membengkak untuk volume besar.

---

## 5. Rekomendasi Technical Approach

**Rekomendasi utama: LLM API (Gemini) untuk deployment awal.**

Alasan:
1. Performa jauh lebih unggul (Accuracy 0.93 vs 0.51 pada data yang tidak bocor).
2. Tidak memerlukan training atau feature engineering.
3. Lebih tahan terhadap variasi bahasa.
4. Iterasi cepat — cukup ubah prompt untuk domain baru.

**Model klasik tetap relevan** sebagai komplementer jika volume prediksi sangat tinggi dan biaya API menjadi kendala utama, atau jika latency rendah (< 1 detik) menjadi syarat mutlak.

**Strategi deployment:**
1. **Jangka pendek:** LLM API sebagai solusi utama.
2. **Jangka menengah:** Kumpulkan data produksi untuk retrain model klasik.
3. **Jangka panjang:** Hybrid — model klasik untuk volume besar & confidence tinggi, LLM untuk kasus ambigu.

---

## 6. Cara Menjalankan

### 6.1 Struktur Project
```
model-experiment-assignment/
├── data/
├── notebook/
├── documentation/
├── .gitignore
├── README.md
└── requirements.txt
```

### 6.2 Prasyarat
- Python 3.10+
- Gemini API key (dapatkan gratis di https://aistudio.google.com/apikey)

### 6.3 Setup

**1. Clone / download project, lalu masuk ke folder:**
```bash
cd model-experiment-assignment
```

**2. Buat virtual environment**

python -m venv .venv
> **Windows:**
.venv\Scripts\activate
> 
> **Linux/Mac:**
source .venv/bin/activate

**3. Install Dependencies**

pip install -r requirements.txt

**4. Buat file .env di root project**

GEMINI_API_KEY=api_key_kamu_di_sini

### 6.4 Catatan
- Notebook menggunakan **GroupShuffleSplit** dengan `review_text` sebagai group untuk mencegah data leakage.
- Cell inference LLM memerlukan waktu ~2 menit pada kondisi server normal.
- Kalau server Gemini overload (503/429), notebook otomatis retry dan fallback antar model.