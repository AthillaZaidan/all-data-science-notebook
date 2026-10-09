# All Data Science Notebook

Kumpulan notebook template siap pakai untuk lomba data (Kaggle-style). Setiap domain punya dua notebook:

1. **EDA**: analisis lengkap + grafik, lalu menulis `eda-output/eda_decisions.json`.
2. **Pipeline**: membaca `eda_decisions.json`, lalu menjalankan baseline → model utama → ensemble → evaluasi → interpretasi → export.

Jadi alurnya selalu **EDA dulu, baru pipeline**. Semua keputusan pipeline (metrik, skema CV, transformasi target, kolom yang dibuang, dsb.) diambil dari temuan EDA dan dicatat di tabel keputusan, sehingga mudah dijelaskan di laporan dan presentasi.

| Domain | EDA | Pipeline | Baseline wajib | Model yang bisa dipilih |
|---|---|---|---|---|
| Tabular | [`tabular/eda_tabular.ipynb`](tabular/eda_tabular.ipynb) | [`tabular/pipeline_tabular.ipynb`](tabular/pipeline_tabular.ipynb) | Linear / Logistic Regression | LightGBM, XGBoost, CatBoost (+ Optuna) |
| NLP | [`nlp/eda_nlp.ipynb`](nlp/eda_nlp.ipynb) | [`nlp/pipeline_nlp.ipynb`](nlp/pipeline_nlp.ipynb) | TF-IDF (word + char) + Linear | IndoBERT, RoBERTa, DeBERTa-v3, ModernBERT, XLM-R, mDeBERTa, dll. |
| Forecasting | [`forecasting/eda_forecasting.ipynb`](forecasting/eda_forecasting.ipynb) | [`forecasting/pipeline_forecasting.ipynb`](forecasting/pipeline_forecasting.ipynb) | Naive, Seasonal Naive, Moving Average, ETS | LightGBM, XGBoost, CatBoost (global model) |
| Multimodal: tabular + teks | [`multimodal/tabular-text/eda_multimodal.ipynb`](multimodal/tabular-text/eda_multimodal.ipynb) | [`multimodal/tabular-text/pipeline_multimodal.ipynb`](multimodal/tabular-text/pipeline_multimodal.ipynb) | Linear / Logistic Regression | LightGBM, XGBoost, CatBoost + TF-IDF, embedding transformer, stacking NLP |
| Multimodal: forecasting + teks | [`multimodal/forecast-text/eda_forecast_text.ipynb`](multimodal/forecast-text/eda_forecast_text.ipynb) | [`multimodal/forecast-text/pipeline_forecast_text.ipynb`](multimodal/forecast-text/pipeline_forecast_text.ipynb) | Naive, Seasonal Naive, Moving Average, ETS | LightGBM, XGBoost, CatBoost + fitur teks ter-lag (embedding / TF-IDF) |
| Tabular foundation model | pakai EDA tabular | [`tabular-foundation/pipeline_foundation.ipynb`](tabular-foundation/pipeline_foundation.ipynb) | Linear / Logistic Regression | **Causilo** (in-context learning) + LightGBM / XGBoost / CatBoost |
| CV: klasifikasi gambar | analisis data di section 4 | [`computer-vision/classification/pipeline_cv_classification.ipynb`](computer-vision/classification/pipeline_cv_classification.ipynb) | Linear probe di fitur DINOv3 beku | **DINOv3 ViT-L/16** fine-tuning (LLRD) + blend |
| CV: segmentasi | analisis data di section 4 | [`computer-vision/segmentation/pipeline_cv_segmentation.ipynb`](computer-vision/segmentation/pipeline_cv_segmentation.ipynb) | Linear head di patch DINOv3 beku | **DINOv3 ViT-L/16** + decoder multi-layer + blend |

---

## Quick Start

### 1. Pilih lingkungan

Notebook otomatis mendeteksi lingkungan:

| Lingkungan | Deteksi | Path data yang dipakai | Folder output |
|---|---|---|---|
| Kaggle | `/kaggle/input` ada | `KAGGLE_PATH` | `/kaggle/working/eda-output`, `/kaggle/working/pipeline-output` |
| Colab | `/content` ada | `COLAB_PATH` | `eda-output/`, `pipeline-output/` |
| Lokal | selain di atas | `LOCAL_PATH` | `eda-output/`, `pipeline-output/` (relatif ke folder notebook) |

Cell `%pip install` di bagian **Initialization** menginstal semua dependency. Kalau di lokal, cukup jalankan sekali di venv.

> **Catatan macOS:** LightGBM dan XGBoost butuh `libomp`. Install dengan `brew install libomp`. Kalau tidak tersedia, notebook akan melewati model tersebut dengan peringatan dan tetap jalan dengan model lain.

### 2. Taruh data dan arahkan path ke folder

Path data cukup diisi **folder dataset**. Notebook mencari sendiri file di dalamnya berdasarkan nama:

| File | Dikenali dari nama | Dipakai untuk |
|---|---|---|
| Train | `train*` (kalau tidak ada: file terbesar) | data latih |
| Test | `test*` | prediksi + `submission.csv` |
| Sample submission | mengandung `sample` atau `submission` | template `submission.csv` (urutan ID, nama kolom, label vs probabilitas) |

```python
KAGGLE_PATH = '/kaggle/input/<dataset-slug>'   # folder dataset di Kaggle
COLAB_PATH  = '/content/drive/MyDrive/<folder>'
LOCAL_PATH  = 'data'                           # folder data/ di samping notebook
READ_KWARGS = {}                               # misal {'sep': ';'} atau {'sheet_name': 0}
```

- **Lokal:** taruh `train.csv`, `test.csv`, `sample_submission.csv` di `<domain>/data/` (misal `tabular/data/`), lalu Run All. Tidak perlu edit path sama sekali.
- **Kaggle:** tambahkan dataset/kompetisi lewat *Add Input*. Kalau `KAGGLE_PATH` dibiarkan placeholder, notebook mencari di seluruh `/kaggle/input`.
- **Path salah / tidak ada:** notebook memberi `[warn]` lalu mencari otomatis. Di lokal urutannya `data/` → `input/` → `dataset/` → folder notebook. Di Kaggle `/kaggle/input`, di Colab `/content`. Folder `eda-output/` dan `pipeline-output/` tidak ikut dicari.
- `KAGGLE_PATH` / `LOCAL_PATH` tetap boleh diisi path file langsung (`.../train.csv`). Test dan sample submission dicari di folder yang sama. `TEST_*_PATH` hanya perlu diisi kalau nama file test-nya tidak diawali `test`.
- Di awal notebook tercetak file mana yang dipakai:

```
train             : data/train.csv
test              : data/test.csv
sample submission : data/sample_submission.csv
```

Format yang didukung: `.csv`, `.tsv`, `.parquet`, `.xlsx`/`.xls`, `.json` (NLP juga `.jsonl`).

Kalau file EDA tidak ada, kolom target ditebak dari `sample_submission` (kolom keduanya), lalu dari satu-satunya kolom yang ada di train tapi tidak ada di test. Untuk notebook EDA, tetap set nama kolom target dan kolom lain yang relevan (lihat bagian per domain di bawah).

### 3. Jalankan EDA → Pipeline

1. Run All `eda_*.ipynb` → menghasilkan `eda-output/` (grafik, tabel, `eda_summary.md`, `eda_decisions.json`).
2. Run All `pipeline_*.ipynb` di folder yang sama. Pipeline membaca `eda-output/eda_decisions.json` lewat setting `EDA_DIR`.

Di Kaggle, kalau EDA dan pipeline dijalankan di notebook terpisah, upload `eda-output/` sebagai dataset lalu arahkan `EDA_DIR` ke `/kaggle/input/<nama-dataset>/eda-output`.

### Opsi: satu notebook EDA + pipeline

Tiap folder juga punya **`full_<domain>.ipynb`**, yaitu EDA dan pipeline digabung dalam satu notebook yang jalan dari awal sampai `submission.csv`:

| Domain | Notebook gabungan |
|---|---|
| Tabular | [`tabular/full_tabular.ipynb`](tabular/full_tabular.ipynb) |
| NLP | [`nlp/full_nlp.ipynb`](nlp/full_nlp.ipynb) |
| Forecasting | [`forecasting/full_forecasting.ipynb`](forecasting/full_forecasting.ipynb) |
| Multimodal tabular + teks | [`multimodal/tabular-text/full_multimodal.ipynb`](multimodal/tabular-text/full_multimodal.ipynb) |
| Multimodal forecasting + teks | [`multimodal/forecast-text/full_forecast_text.ipynb`](multimodal/forecast-text/full_forecast_text.ipynb) |
| Foundation model | [`tabular-foundation/full_foundation.ipynb`](tabular-foundation/full_foundation.ipynb) |

- **Part 1 (EDA)** menulis `eda-output/eda_decisions.json`. **Part 2 (pipeline)** membacanya lewat `EDA_DIR`, jadi semua keputusan `'auto'` tetap berasal dari temuan EDA.
- **Path data cukup diisi sekali** di Settings Part 1. Pipeline otomatis memakai file yang sudah ditemukan di Part 1.
- Setting model (pilihan model, tuning, ensemble) ada di cell Settings section *Pipeline Initialization*.
- Cocok untuk Kaggle: satu notebook, satu kali *Run All*.

---

## Cara Kerja Keputusan `'auto'`

Hampir semua setting pipeline bisa diisi `'auto'`. Nilainya ditentukan dengan prioritas:

```
nilai yang kamu isi sendiri  >  eda_decisions.json  >  dihitung dari data
```

Setiap keputusan dicatat di tabel **Decisions** (setting, nilai, sumber `user`/`eda`/`computed`, alasan), yang ditampilkan di notebook dan disimpan ke `pipeline-output/final/decisions.csv` serta `report.md`. Kalau file EDA tidak ada, pipeline tetap jalan dan semua keputusan ditandai `computed`.

Untuk mengunci keputusan secara manual, ganti `'auto'` dengan nilai eksplisit, misalnya `METRIC = 'rmse'` atau `CV_SCHEME = 'group'`.

---

## Tabular

### Settings penting

**EDA (`eda_tabular.ipynb`)**

| Setting | Isi |
|---|---|
| `TARGET` | nama kolom target. `None` = EDA tanpa target |
| `TASK` | `'auto'`, `'regression'`, `'binary'`, `'multiclass'` |
| `ID_COLS`, `DROP_COLS`, `DATE_COLS` | kolom ID, kolom yang dibuang, kolom tanggal |
| `TIME_COL` | kolom waktu untuk analisis tren dan cek drift (boleh `None`) |
| `GROUP_COL` | kolom segmen untuk analisis per grup |
| `CV_GROUP` | kolom yang dicek untuk leakage (ID berulang) |
| `NUM_COLS`, `CAT_COLS`, `TEXT_COLS` | `'auto'` atau daftar kolom |
| `SAMPLE_N` | jumlah baris sampel untuk plot berat |

**Pipeline (`pipeline_tabular.ipynb`)**

| Setting | Isi |
|---|---|
| `TARGET`, `TASK`, `ID_COLS`, `DROP_COLS`, `SENTINELS` | `'auto'` = ambil dari EDA |
| `CV_SCHEME` | `'auto'`, `'kfold'`, `'stratified'`, `'group'`, `'time'` |
| `CV_GROUP`, `TIME_COL`, `DEDUP`, `METRIC` | `'auto'` = ambil dari EDA |
| `TARGET_TRANSFORM`, `CLASS_WEIGHT`, `DROP_WEAK` | `'auto'` = ambil dari EDA |
| `HOLDOUT_SIZE` | porsi holdout untuk evaluasi akhir (default 0.2) |
| `BASELINE` | `'linear'` (Linear / Logistic Regression), selalu dijalankan |
| `MODELS` | subset dari `['lightgbm', 'xgboost', 'catboost']` |
| `MODEL_PARAMS` | override hyperparameter per model |
| `TUNE`, `TUNE_MODEL`, `N_TRIALS` | tuning Optuna (default mati) |
| `ENSEMBLE` | blend bobot optimal dari OOF |
| `SUBMISSION_ID` | kolom ID untuk `submission.csv` |
| `SUBMISSION_FORMAT` | `'auto'` (ikut `sample_submission`; kalau tidak ada: probabilitas untuk ROC AUC / PR AUC / log loss, selain itu label), `'label'`, `'proba'` |
| `USE_GPU` | `True` kalau pakai GPU Kaggle/Colab |

### Yang dibaca pipeline dari EDA

| Keputusan | Bukti dari EDA |
|---|---|
| Target, task, ID, peran kolom | profil kolom |
| Kolom yang dibuang | kolom konstan / hampir konstan |
| Sentinel → NaN | nilai placeholder di luar rentang (misal `-999`) |
| Skema CV + kolom group | ID berulang dan selisih skor split acak vs per grup |
| Split berdasarkan waktu | kolom waktu + drift train/test kuat (adversarial AUC > 0.7) |
| Transformasi target, metrik, class weight | skewness target, jenis task, imbalance |
| Dedup | baris duplikat persis |
| Fitur lemah | mutual information ≈ 0 dan tidak signifikan |

---

## NLP

### Settings penting

**EDA (`eda_nlp.ipynb`)**

| Setting | Isi |
|---|---|
| `TEXT_COL`, `LABEL_COL` | kolom teks dan label (`LABEL_COL` boleh kelas, angka, atau `None`) |
| `GROUP_COL`, `TIME_COL`, `ID_COL` | opsional |
| `LANGUAGE` | `'en'`, `'id'`, atau `'both'`. Stopword Indonesia sudah termasuk slang |
| `EXTRA_STOPWORDS` | stopword tambahan sesuai domain |
| `REMOVE_PATTERNS` | regex yang dibuang sebelum analisis, misal `[r'\[[^\]]*\]', r'http\S+']` |
| `TOPIC_K`, `MAX_DOCS`, `TSNE_SAMPLE` | knob analisis topik, sampel, peta semantik |

Isi EDA: kualitas teks, panjang, vocabulary, word cloud (keseluruhan, per label, kata khas), n-gram, kata pembeda per label (log-odds), sentimen (VADER), topic modeling (NMF), peta t-SNE, duplikat dan label noise.

**Pipeline (`pipeline_nlp.ipynb`)**

| Setting | Isi |
|---|---|
| `TEXT_COL`, `LABEL_COL`, `TASK` | `'auto'` = ambil dari EDA |
| `TEXT_PAIR_COL` | kolom teks kedua untuk task pasangan kalimat (opsional) |
| `TOP_K_CLASSES` | batasi ke K kelas terbanyak (`None` = semua) |
| `DEBUG_SAMPLE` | jumlah baris untuk uji cepat (`None` = semua) |
| `MODELS` | `['auto']` = rekomendasi EDA, atau daftar key dari `MODEL_REGISTRY` |
| `MAX_LENGTH`, `TRUNCATION` | `'auto'` dari distribusi panjang token; `'head_tail'` menyimpan awal + akhir teks |
| `EPOCHS`, `BATCH_SIZE`, `LR`, `PATIENCE` | hyperparameter fine-tuning |
| `TRAIN_FOLDS` | fold yang dilatih, misal `[0]` (cepat) atau `[0,1,2,3,4]` (penuh) |
| `MIXED_PRECISION` | `'auto'` = aktif kalau ada GPU |
| `SAVE_WEIGHTS` | simpan bobot model ke `final/models/` |
| `ID_COL`, `SUBMISSION_FORMAT` | kolom ID di `submission.csv`; format `'auto'` / `'label'` / `'proba'` seperti di tabular |

Pilihan model di `MODEL_REGISTRY`:

| Key | Checkpoint | Cocok untuk |
|---|---|---|
| `indobert`, `indobert-large`, `indolem`, `indoroberta` | IndoBenchmark / IndoLEM / Flax | Bahasa Indonesia |
| `roberta`, `twitter-roberta` | RoBERTa base, RoBERTa sentimen Twitter | Bahasa Inggris, teks medsos |
| `deberta-v3`, `modernbert` | DeBERTa-v3 base, ModernBERT base | Bahasa Inggris, akurasi tinggi / konteks panjang |
| `xlm-roberta`, `mdeberta`, `multilingual-e5` | multilingual | Campuran bahasa |
| `distilbert`, `bert-tiny` | model kecil | Uji cepat di CPU |

Model lain dari Hugging Face bisa ditambahkan langsung ke `MODEL_REGISTRY`. Untuk transformer, gunakan GPU (Kaggle T4/P100 atau Colab).

---

## Forecasting

### Format data

Format panjang (long format): satu baris per tanggal per series.

```
date,store,item,sales,promo
2023-01-01,S1,I1,12,0
2023-01-01,S1,I2,5,1
...
```

Single series cukup `ID_COLS = []`. Kalau tidak ada data sama sekali dan `DEMO_IF_MISSING = True`, notebook membuat data demo di `demo-data/`. Kalau ada `sample_submission` dengan kolom ID yang juga ada di file test (misal `id`), `submission.csv` mengikuti urutan dan nama kolomnya.

### Settings penting

**EDA (`eda_forecasting.ipynb`)**

| Setting | Isi |
|---|---|
| `DATE_COL`, `TARGET_COL` | kolom tanggal dan target |
| `ID_COLS` | kolom pembentuk series, misal `['store', 'item']` |
| `EXOG_COLS` | variabel eksogen (`'auto'` = semua kolom numerik lain) |
| `AGG` | agregasi kalau ada tanggal duplikat (`'sum'` / `'mean'`) |
| `FREQ`, `HORIZON`, `SEASONAL_PERIODS` | `'auto'` = diinferensi; horizon dari panjang test kalau ada |

Isi EDA: frekuensi dan kelengkapan, intermittency (ADI/CV²), transformasi target, dekomposisi STL, pola musiman, ACF/PACF dan periodogram, stasioneritas (ADF + KPSS), anomali dan level shift, driver kalender/eksogen dan hari spesial, struktur antar series, forecastability baseline.

**Pipeline (`pipeline_forecasting.ipynb`)**

| Setting | Isi |
|---|---|
| `FREQ`, `HORIZON`, `SEASONAL_PERIODS`, `TRANSFORM`, `FILL_MISSING` | `'auto'` = ambil dari EDA |
| `STRATEGY` | `'direct'` (lag ≥ horizon, satu model) atau `'recursive'` |
| `LAGS`, `ROLL_WINDOWS` | `'auto'` = lag signifikan dari ACF EDA |
| `EVENT_DATES`, `EVENT_WINDOW` | tanggal spesial (default: Lebaran 2021–2027) dan jendela hari sekitarnya |
| `N_FOLDS`, `METRIC` | backtest expanding window; metrik `wape` / `smape` / `mase` / `rmse` / `mae` (`'auto'`: WAPE kalau ada banyak nol, selain itu sMAPE) |
| `MODELS` | subset dari `['lightgbm', 'xgboost', 'catboost']` |
| `BASELINES` | subset dari `['naive', 'seasonal_naive', 'moving_average', 'ets']` |
| `OBJECTIVE`, `WEIGHT_BY_VOLUME`, `NON_NEGATIVE` | `'auto'` = dari EDA (misal Tweedie untuk data intermittent) |
| `INTERVAL` | tingkat prediction interval empiris (default 0.8) |

---

## Tipe Target yang Didukung

Berlaku untuk pipeline **tabular**, **multimodal**, dan **NLP**.

| Tipe | Cara set | Metrik default | Aturan keputusan yang di-tune di OOF |
|---|---|---|---|
| Regresi | otomatis (angka dengan > 20 nilai unik) | RMSE | - |
| Biner | otomatis | ROC AUC (tabular), macro F1 (NLP) | threshold, kalau metrik berbasis label (`'f1'`, `'accuracy'`, ...) |
| Multiclass | otomatis | macro F1 | skala per kelas: `argmax(p × s)`, mendongkrak macro F1 di data imbalance |
| **Ordinal** (rating 1–5, rendah/sedang/tinggi) | `TASK = 'ordinal'` atau `METRIC = 'qwk'`. Label string pakai `ORDINAL_ORDER = ['rendah', 'sedang', 'tinggi']` | QWK | cut-point antar kelas (model dilatih sebagai regresi di urutan kelas) |
| **Multi-label** (satu baris bisa banyak label) | `MULTILABEL_SEP = '|'` kalau satu kolom berisi `"a|b"`, atau `LABEL_COL` / `TARGET` = list kolom 0/1 | F1 per label (tabular), micro F1 (NLP) | threshold per label |
| **Multi-target** (beberapa kolom target, tabular) | `TARGET = ['y1', 'y2']`, atau otomatis kalau `sample_submission` punya > 1 kolom target | per target | per target |

- `TUNE_DECISION = True` (default) mengaktifkan tuning di atas. Tuning **hanya pakai prediksi out-of-fold**, lalu dinilai sekali di holdout, jadi skornya tetap jujur.
- Multi-target di tabular: target pertama dapat analisis lengkap (SHAP, error analysis). Target berikutnya pakai fitur dan fold yang sama, dengan baseline, GBM, blend, dan aturan keputusan masing-masing. Hasilnya `multi_target_metrics.csv`, plus `multilabel_scores.json` (micro / macro / samples F1) untuk multi-label.
- Format `submission.csv` tetap mengikuti `sample_submission`. Untuk multi-label satu kolom, label digabung lagi dengan separator yang sama.

---

## Tabular Foundation Model (Causilo)

[Causilo](https://github.com/nums-ai/causilo) (Nums AI) adalah *tabular foundation model*: transformer yang dilatih di sekitar 36 juta tabel sintetis. Cara kerjanya *in-context learning*: baris training dipakai sebagai konteks, lalu prediksi keluar dalam satu forward pass. Model ini **tidak dilatih ulang** di data lomba, jadi tidak ada learning rate, early stopping, atau tuning.

`tabular-foundation/pipeline_foundation.ipynb` adalah pipeline tabular yang sama (keputusan EDA, fold, baseline, ensemble, tuning keputusan, ordinal, multi-target, format `sample_submission`), ditambah:

| Setting | Isi |
|---|---|
| `MODELS` | default `['causilo', 'lightgbm']`. Causilo dinilai di fold yang sama dan bisa di-blend dengan GBM |
| `CAUSILO_ESTIMATORS` | anggota ensemble Causilo, `'auto'` = 8 di GPU, 4 di CPU |
| `CAUSILO_DEVICE` | `'auto'` (CUDA kalau ada), `'cuda'`, `'mps'`, `'cpu'` |
| `MAX_CONTEXT_ROWS` | batas baris konteks per fit (default 50.000). Data lebih besar disubsample (stratified) |
| `CONTEXT_CURVE` | kurva skor vs jumlah baris training, Causilo vs GBM. Bukti untuk laporan bahwa prior pretrained membantu di data kecil |
| `INTERVAL` | prediction interval dari 999 kuantil Causilo (regresi). Coverage dicek di holdout, interval test disimpan ke `test_intervals.csv` |
| `PERM_IMPORTANCE` | permutation importance untuk Causilo (TreeSHAP tidak berlaku karena tidak ada pohon) |

Hasil uji di data sintetis (CV, tanpa tuning):

| Kasus | Causilo | LightGBM |
|---|---|---|
| Regresi (RMSE ↓) | **1.006** | 1.386 |
| Biner (ROC AUC ↑) | **0.766** | 0.736 |
| Ordinal (QWK ↑) | **0.869** | 0.852 |
| Multiclass 5 kelas (macro F1 ↑) | **0.417** | 0.409 |

Catatan:

- **Jalankan EDA tabular dulu** (`tabular/eda_tabular.ipynb`) di folder kerja yang sama, atau arahkan `EDA_DIR` ke `eda-output` yang sudah ada.
- **Butuh internet saat fit pertama**: checkpoint diunduh dari Hugging Face (`nums-ai/causilo`) dan di-cache. Di Kaggle, aktifkan *Internet* di setting notebook.
- **GPU sangat disarankan** untuk data di atas ~10 ribu baris. Di CPU, 2.000 baris × 5 fold butuh sekitar 30–45 detik.
- **Lisensi**: kode Apache-2.0, tapi **bobot model** pakai *Causilo License v1.0* (riset non-komersial diizinkan, penggunaan komersial/produksi butuh lisensi terpisah). Cek dulu aturan lomba soal model pretrained.

---

## Multimodal: Tabular + Teks (`multimodal/tabular-text/`)

Untuk tabel yang punya kolom angka/kategori **dan** kolom teks bebas (deskripsi produk, ulasan, keluhan, iklan lowongan). Isinya sama dengan notebook tabular, ditambah:

**EDA (`eda_multimodal.ipynb`)**: section 15 **Modality Signal Check** membandingkan skor out-of-fold dari:
- naive (tebak rata-rata atau kelas mayoritas)
- teks saja (TF-IDF + model linear, per kolom teks)
- tabular saja (gradient boosting)
- tabular + teks

Selisih "tabular + teks" dengan "tabular saja" jadi dasar rekomendasi `text_mode` di `eda_decisions.json`:

| Kondisi | `text_mode` |
|---|---|
| Teks menambah skor dengan jelas | `'both'` |
| Teks menambah sedikit | `'tfidf'` |
| Teks tidak punya sinyal | `'none'` |

**Pipeline (`pipeline_multimodal.ipynb`)**

| Setting | Isi |
|---|---|
| `TEXT_MODE` | `'tfidf'` (TF-IDF → SVD), `'embed'` (embedding transformer → PCA), `'both'`, `'none'`, atau `'auto'` (ikut EDA; kalau tidak ada, `'both'` saat ada GPU atau datanya kecil) |
| `EMBED_MODEL` | encoder Hugging Face, default `intfloat/multilingual-e5-base` (Indonesia + Inggris). Alternatif: `indobenchmark/indobert-base-p1`, `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBED_DIM` | jumlah komponen PCA per kolom teks (default 32) |
| `NLP_OOF_DIR` | folder `pipeline-output/final` dari pipeline NLP. Prediksi OOF model BERT jadi fitur `nlp_*` (stacking) |
| `MODALITY_ABLATION` | latih ulang model pertama dengan fitur tabular saja, + tiap keluarga fitur teks, teks saja, dan semua. Hasilnya `modality_ablation.csv` + grafik |

Embedding dihitung sekali per teks unik lalu di-cache di `pipeline-output/cache/`, jadi run kedua langsung cepat.

### Alur stacking dengan model NLP fine-tuned

1. Jalankan `nlp/pipeline_nlp.ipynb` di data yang sama, dengan `TEXT_COL` = kolom teks, `LABEL_COL` = target, `ID_COL` = kolom ID, dan **`TRAIN_FOLDS = [0, 1, 2, 3, 4]`**. Semua fold wajib dilatih supaya tiap baris punya prediksi out-of-fold. Kalau tidak, stacking-nya bocor dan notebook akan memberi peringatan.
2. Di `pipeline_multimodal.ipynb`, set `NLP_OOF_DIR = '../../nlp/pipeline-output/final'`.
3. Prediksi digabung lewat kolom ID yang sama. Tabel ablation menunjukkan seberapa besar tambahan dari model NLP.

---

## Multimodal: Forecasting + Teks (`multimodal/forecast-text/`)

Untuk forecasting yang punya teks bertanggal, misalnya berita, pengumuman promo, ulasan, atau postingan media sosial, yang bisa memberi sinyal **sebelum** target berubah.

**Sumber teks** (otomatis dicari):

| Sumber | Contoh | Cara pakai |
|---|---|---|
| File teks terpisah di folder dataset | `news.csv` (`date`, `store`, `headline`) | nama file mengandung `news`, `text`, `review`, `tweet`, `post`, `berita`, `ulasan`, atau set `TEXT_*_PATH` |
| Kolom teks di data utama | `train.csv` punya kolom `deskripsi` per (tanggal, series) | kolom string rata-rata ≥ 3 kata, atau set `TEXT_COL` |

Kalau teks punya kolom ID series (misal `store`), fitur teks dipetakan ke series yang cocok (boleh sebagian ID saja). Kalau tidak, teks dianggap global dan dipakai semua series.

**EDA (`eda_forecast_text.ipynb`)**: section 16 **Text Signal**:
- volume dokumen per periode vs target
- heatmap korelasi |r| antara fitur teks di t − lag dan target (sudah di-detrend) di t
- lag terbaik **yang ≥ horizon**, dibandingkan dengan batas noise 2/√n

Hasilnya `text_lag`, `text_useful`, dan `text_windows` di `eda_decisions.json`.

**Pipeline (`pipeline_forecast_text.ipynb`)**:

| Setting | Isi |
|---|---|
| `TEXT_FEATURES` | `'embed'` (transformer `EMBED_MODEL` → PCA, di-cache), `'tfidf'` (TF-IDF → SVD), `'auto'` |
| `TEXT_DIM` | jumlah komponen teks per dokumen (default 8) |
| `TEXT_LAG` | `'auto'` = lag dari EDA. **Selalu ≥ horizon**: kalau diisi lebih kecil, dinaikkan otomatis karena teks masa depan belum ada saat forecast dibuat |
| `TEXT_WINDOWS` | jendela rolling untuk fitur teks ter-lag (default dari EDA: 1, M, 4M) |
| `TEXT_ABLATION` | backtest ulang model pertama **tanpa** fitur teks, lalu bandingkan per fold. Hasilnya `text_ablation.csv` + grafik + share importance per keluarga fitur |

Fitur teks: jumlah dokumen, panjang rata-rata, dan komponen embedding per (series, periode), di-lag `TEXT_LAG` lalu di-rolling. Jadi semua nilai di baris masa depan berasal dari teks yang sudah terbit di histori, **tanpa kebocoran**.

Kalau tidak ada data, `DEMO_IF_MISSING = True` membuat panel penjualan demo + `news.csv` berisi pengumuman promo yang menaikkan penjualan 30–36 hari kemudian, jadi efek teksnya bisa langsung terlihat.

---

## Computer Vision: DINOv3 ViT-L (`computer-vision/`)

Dua notebook pipeline untuk data gambar, keduanya memakai backbone **DINOv3 ViT-L/16** (`facebook/dinov3-vitl16-pretrain-lvd1689m`, 300M parameter). Tidak ada notebook EDA terpisah: section 4 (**Data Overview and Decisions**) menganalisis data lalu mengisi semua setting `'auto'` dan mencatatnya di `decisions.csv`.

**Akses bobot DINOv3 (gated).** Sekali saja:

1. Login ke huggingface.co, buka halaman model `facebook/dinov3-vitl16-pretrain-lvd1689m`, lalu terima lisensinya.
2. Buat token *read* di Settings → Access Tokens.
3. Simpan sebagai secret bernama `HF_TOKEN`:
   - **Kaggle:** Add-ons → Secrets
   - **Colab:** ikon kunci
   - **Lokal:** `export HF_TOKEN=...`

Jangan tempel token di notebook. Kalau bobot tidak bisa dimuat (tanpa token atau tanpa internet), notebook otomatis pindah ke `BACKBONE_FALLBACK` (`dinov2-large`, tidak gated). Peralihan ini dicatat di tabel keputusan. Untuk Kaggle offline, upload hasil `save_pretrained` sebagai dataset, lalu isi `BACKBONE` dengan path foldernya.

**GPU.** Linear probe/head jalan di CPU atau Apple Silicon. Fine-tuning ViT-L butuh GPU CUDA, jadi `FINETUNE = 'auto'` hanya aktif kalau ada CUDA. Tanpa GPU, notebook tetap menghasilkan submission dari baseline.

### Klasifikasi gambar (`computer-vision/classification/`)

| Layout data | Contoh |
|---|---|
| Tabel | `train.csv` (nama file / id gambar + label), `test.csv`, `sample_submission.csv`, folder gambar bebas |
| Folder kelas | `train/<kelas>/*.jpg`, `test/*.jpg` |

- **Task:** binary, multiclass, multi-label (beberapa kolom 0/1 atau `MULTILABEL_SEP`), dan regresi (misal umur dari foto).
- **Alur:**
  1. **Linear probe** dengan C di-tune per fold, memakai fitur `[CLS ; mean(patch)]` yang di-cache dan pakai flip TTA.
  2. **Fine-tuning** `UNFREEZE_LAST_N` blok terakhir dengan layer-wise LR decay, augmentasi, AMP, dan early stopping.
  3. **Blend** keduanya.
  4. **Tuning decision rule** di OOF: threshold, class scale, atau threshold per label.
  5. **Evaluasi holdout**.
- **Interpretasi:** *patch evidence map* menunjukkan kontribusi tiap patch ke logit kelas. Ini dihitung eksak dari head linear, tanpa gradien.
- **Error analysis:** gambar dengan kesalahan paling yakin, dan pasangan kelas yang paling sering tertukar.

### Segmentasi (`computer-vision/segmentation/`)

| Layout data | Contoh |
|---|---|
| File mask | `train/images/x.jpg` + `train/masks/x_mask.png` (folder bernama `mask`/`label`/`gt`/`annot`/`seg`, atau suffix `_mask`) |
| Tabel RLE | `train.csv` (`ImageId`, [`ClassId`], `EncodedPixels`), gaya Severstal / Carvana |

- **Nilai mask dibaca otomatis:**
  - 0/255 atau 0/1 → binary
  - 0..K dengan 255 → 255 dianggap *ignore* (gaya VOC)
  - mask RGB → satu kelas per warna, dengan pemetaan ke warna terdekat (tahan JPEG)
  - banyak level abu-abu → di-threshold
- **Override:** `MASK_VALUES`, `IGNORE_INDEX`, `CLASS_NAMES`.
- **RLE:**
  - Urutan piksel (`'F'` kolom, `'C'` baris) dideteksi otomatis dari bentuk mask.
  - Tabel yang hanya berisi gambar ber-objek menambahkan gambar lain di folder yang sama sebagai mask kosong.
- **Alur:**
  1. **Linear head** di patch beku layer terakhir (baseline).
  2. **Decoder multi-layer:** 4 layer DINOv3 digabung di 1/16, naik ke 1/8, lalu digabung dengan cabang CNN resolusi 1/4 untuk detail tepi. Blok terakhir di-fine-tune.
  3. **Blend** keduanya.
  4. **Tuning threshold + ukuran objek minimum** di OOF.
  5. **Evaluasi holdout di resolusi asli.**
- **Loss:** CE/BCE + Dice. Bobotnya `'auto'` dari luas foreground.
- **Metrik:** Dice per gambar untuk binary (mask kosong yang diprediksi kosong = 1), mIoU untuk multiclass.
- **Output:**
  - `masks/*.png` di resolusi asli
  - `submission.csv` dengan RLE yang mengikuti `sample_submission.csv`: satu baris per gambar, per (gambar, kelas), atau id gabungan `gambar_kelas`
  - tanpa sample: `id, rle` (binary) atau `id, class, rle` (multiclass)
  - string mask kosong diatur lewat `EMPTY_RLE`

---

## Output

```
<domain>/
├── eda-output/
│   ├── figures/              # semua grafik, bernomor urut (01_..., 02_...)
│   ├── tables/               # tabel CSV pendukung
│   ├── eda_summary.md        # ringkasan temuan
│   └── eda_decisions.json    # dibaca pipeline
└── pipeline-output/
    ├── baseline/             # metrik, OOF, grafik baseline
    └── final/
        ├── figures/          # leaderboard, learning curve, SHAP, error analysis
        ├── decisions.csv     # tabel keputusan + sumber + alasan
        ├── leaderboard.csv   # skor CV semua model
        ├── final_metrics.*   # skor holdout + bootstrap CI vs baseline
        ├── report.md         # ringkasan siap salin ke laporan
        ├── config.json
        └── submission.csv    # kalau ada file test (format mengikuti sample_submission kalau ada)
```

Folder output tidak di-commit (`.gitignore`) karena bisa dibuat ulang dengan menjalankan notebook.

Grafik memakai palet yang sama di semua notebook (`#3D5A80` primary, `#EE6C4D` accent), jadi bisa langsung dipakai di laporan.

---

## Mengedit Notebook

Setiap `.ipynb` dibuat dari skrip builder `build_*.py` di folder yang sama. Untuk perubahan permanen, edit builder lalu generate ulang:

```bash
python tabular/build_pipeline_tabular.py
```

Notebook gabungan `full_*.ipynb` dibuat dari notebook EDA dan pipeline, jadi generate ulang setelah mengubah builder mana pun:

```bash
python build_full_notebooks.py
```

Perubahan kecil untuk satu lomba (misal path dan nama kolom) cukup diedit langsung di notebook.

---

## Tips Lomba

- **Jalankan EDA dulu.** Tanpa EDA, pipeline tetap jalan tapi keputusan hanya berbasis heuristik, dan tabel keputusan tidak punya bukti untuk laporan.
- **Uji cepat dulu:** untuk NLP set `DEBUG_SAMPLE = 2000` dan `MODELS = ['bert-tiny']`; untuk tabular set `MODELS = ['lightgbm']` dan `TUNE = False`; untuk CV set `BACKBONE = 'dinov3-vits16'`, `DEBUG_SAMPLE = 500`, dan `EPOCHS = 2`.
- **Selalu bandingkan dengan baseline.** `final_metrics` memuat improvement vs baseline beserta bootstrap 95% CI, jadi kamu bisa bilang apakah peningkatannya signifikan.
- **Ambil isi laporan dari `report.md` dan `eda_summary.md`**, lalu pilih grafik dari `figures/`.
