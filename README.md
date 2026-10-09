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
| Multimodal (tabular + teks) | [`multimodal/eda_multimodal.ipynb`](multimodal/eda_multimodal.ipynb) | [`multimodal/pipeline_multimodal.ipynb`](multimodal/pipeline_multimodal.ipynb) | Linear / Logistic Regression | LightGBM, XGBoost, CatBoost + TF-IDF, embedding transformer, stacking NLP |

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

## Multimodal (Tabular + Teks)

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
2. Di `pipeline_multimodal.ipynb`, set `NLP_OOF_DIR = '../nlp/pipeline-output/final'`.
3. Prediksi digabung lewat kolom ID yang sama. Tabel ablation menunjukkan seberapa besar tambahan dari model NLP.

### Bagaimana dengan forecasting + teks?

Belum ada notebook khusus, tapi polanya:

1. **Agregasi teks ke level (series, tanggal)**: jumlah dokumen, rata-rata sentimen, proporsi topik, rata-rata embedding (dikompres PCA) per hari/minggu.
2. **Jadikan eksogen**. Kalau teksnya baru diketahui setelah kejadian (berita, ulasan), fitur ini **harus di-lag** minimal sebesar horizon supaya tidak bocor. Kalau teksnya sudah ada di masa depan (misal deskripsi promo yang sudah dijadwalkan), masukkan sebagai kolom di file test supaya terbaca sebagai `EXOG_KNOWN`.
3. Simpan hasil agregasi sebagai kolom tambahan di train/test, lalu jalankan `forecasting/pipeline_forecasting.ipynb` seperti biasa.

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

Perubahan kecil untuk satu lomba (misal path dan nama kolom) cukup diedit langsung di notebook.

---

## Tips Lomba

- **Jalankan EDA dulu.** Tanpa EDA, pipeline tetap jalan tapi keputusan hanya berbasis heuristik, dan tabel keputusan tidak punya bukti untuk laporan.
- **Uji cepat dulu:** untuk NLP set `DEBUG_SAMPLE = 2000` dan `MODELS = ['bert-tiny']`; untuk tabular set `MODELS = ['lightgbm']` dan `TUNE = False`.
- **Selalu bandingkan dengan baseline.** `final_metrics` memuat improvement vs baseline beserta bootstrap 95% CI, jadi kamu bisa bilang apakah peningkatannya signifikan.
- **Ambil isi laporan dari `report.md` dan `eda_summary.md`**, lalu pilih grafik dari `figures/`.
