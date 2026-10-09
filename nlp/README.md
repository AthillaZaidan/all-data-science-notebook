# NLP

Template untuk data teks: klasifikasi (biner / multiclass), regresi skor, ordinal (rating), multi-label, dan pasangan kalimat. Mendukung bahasa Indonesia (termasuk slang), Inggris, dan campuran.

## Isi folder

| File | Fungsi |
|---|---|
| `eda_nlp.ipynb` | EDA teks + menulis `eda-output/eda_decisions.json` |
| `pipeline_nlp.ipynb` | baseline TF-IDF → fine-tuning transformer (IndoBERT, RoBERTa, DeBERTa, XLM-R, ...) → ensemble → holdout → interpretasi → submission |
| `full_nlp.ipynb` | EDA + pipeline dalam satu notebook |
| `build_eda_nlp.py`, `build_pipeline_nlp.py` | skrip builder notebook |

## Cara pakai cepat

1. Taruh data di `nlp/data/`: satu baris per dokumen, satu kolom teks, satu kolom label.
2. *Run All* `eda_nlp.ipynb`. Kolom teks otomatis = kolom string dengan teks terpanjang; isi `TEXT_COL` / `LABEL_COL` kalau salah.
3. Aktifkan GPU, lalu *Run All* `pipeline_nlp.ipynb`.

Tanpa GPU: set `MODELS = []` (baseline TF-IDF saja), atau `MODELS = ['bert-tiny']` + `DEBUG_SAMPLE = 2000` untuk uji cepat.

---

## EDA (`eda_nlp.ipynb`)

### Alur section

| # | Section | Yang dikerjakan kode |
|---|---|---|
| 1–2 | Introduction, Initialization | settings, `discover_files`, pilih kolom teks & label |
| 3 | Text Toolkit | stopword Inggris + Indonesia (termasuk slang), `clean` (hapus URL, HTML, `REMOVE_PATTERNS`, normalisasi unicode, huruf berulang `"bangeeet"` → `"bangeet"`), `tokens_all` / `content` (token tanpa stopword), persiapan label (numerik dibagi kuantil `LABEL_BINS`) |
| 4 | Corpus Overview | distribusi label, contoh dokumen per label |
| 5 | Text Quality and Noise | dokumen kosong, URL, emoji, mention, huruf kapital, karakter aneh |
| 6 | Length Analysis | panjang kata/karakter, per label (dasar `MAX_LENGTH`) |
| 7 | Vocabulary Statistics | ukuran vocab, kata langka, hukum Zipf/Heaps (dasar ukuran TF-IDF) |
| 8 | Word Clouds | word cloud keseluruhan dan per label |
| 9 | N-gram Analysis | unigram/bigram/trigram terbanyak (`doc_freq`), per label |
| 10 | Distinctive Words per Label | kata pembeda per label (log-odds), perbandingan dua label |
| 11 | Sentiment Analysis | skor sentimen VADER (bahasa Inggris) vs label |
| 12 | Topic Modeling | topik NMF (`TOPIC_K`), campuran topik per label |
| 13 | Semantic Map | t-SNE dari TF-IDF, diwarnai per label |
| 14 | Duplicates and Label Noise | duplikat persis & hampir sama (cosine ≥ `NEAR_DUP_SIM`), teks sama dengan label beda |
| 15 | Segment and Temporal Analysis | per `GROUP_COL` / `TIME_COL` |
| 16 | EDA Summary | rekomendasi model + `eda_decisions.json` |

### Settings EDA

| Setting | Default | Arti |
|---|---|---|
| `*_PATH`, `READ_KWARGS` | | lokasi data (sama seperti folder lain) |
| `TEXT_COL` | `'auto'` | kolom teks; `'auto'` = kolom string dengan teks terpanjang |
| `LABEL_COL` | `'auto'` | label (kelas atau angka); `None` = EDA tanpa label |
| `GROUP_COL`, `TIME_COL`, `ID_COL` | `None` | kolom segmen, waktu, ID (opsional) |
| `LANGUAGE` | `'both'` | stopword `'en'`, `'id'`, atau `'both'` |
| `EXTRA_STOPWORDS` | `[]` | stopword khusus domain, misal `['aplikasi', 'gojek']` |
| `REMOVE_PATTERNS` | `[]` | regex yang dihapus, misal `[r'@\w+', r'\[.*?\]']` |
| `MIN_TOKEN_LEN` | `2` | token lebih pendek diabaikan |
| `MAX_DOCS` | `30_000` | sampel dokumen untuk analisis berat |
| `MAX_LABELS` | `8` | jumlah label yang diplot per label |
| `LABEL_BINS` | `4` | label numerik dibagi jadi 4 kuantil |
| `CLASS_MAX` | `30` | label dengan ≤ 30 nilai unik dianggap kelas |
| `TOP_N` | `20` | jumlah n-gram / kata teratas |
| `WORDCLOUD_WORDS` | `150` | kata per word cloud |
| `TOPIC_K` | `8` | jumlah topik NMF |
| `SENTIMENT_SAMPLE`, `TSNE_SAMPLE`, `NEAR_DUP_SAMPLE` | `5_000`, `3_000`, `5_000` | ukuran sampel tiap analisis |
| `NEAR_DUP_SIM` | `0.90` | ambang cosine untuk "hampir duplikat" |
| `TIME_RANGE` | `(1950, 2025)` | rentang tahun valid |

### Isi `eda_decisions.json`

| Key | Arti | Dipakai pipeline untuk |
|---|---|---|
| `text_col`, `label_col`, `label_kind` | kolom & jenis label | `TEXT_COL`, `LABEL_COL`, `TASK` |
| `language`, `language_share` | bahasa dominan | pilihan model `'auto'` |
| `recommended_model` | model yang disarankan (misal `indobert` untuk Indonesia) | `MODELS = ['auto']` |
| `max_length`, `words_p50`, `words_p95` | panjang teks | `MAX_LENGTH` |
| `tfidf_max_features` | ukuran vocab TF-IDF | `TFIDF_MAX_FEATURES` |
| `remove_patterns`, `extra_stopwords`, `noise_patterns` | pola noise yang ditemukan | `REMOVE_PATTERNS` |
| `imbalance_ratio`, `class_weights` | ketimpangan kelas | `CLASS_WEIGHTS` |
| `duplicate_share`, `near_duplicate_share` | porsi duplikat | `DEDUP` |
| `conflicting_label_groups`, `group_identical_texts` | teks sama, label beda | `GROUP_IDENTICAL` (teks identik dikunci di fold yang sama) |

---

## Pipeline (`pipeline_nlp.ipynb`)

### Alur section

| # | Section | Yang dikerjakan kode |
|---|---|---|
| 1–3 | Introduction, Initialization, Toolkit | settings, data, metrik (`score_all`, `to_label`, `pred_labels`), `decide` |
| 4 | EDA-Informed Decisions | `compute_language` (rasio stopword ID vs EN), `compute_max_length` (persentil 95 panjang × 1.3, dibulatkan ke pangkat 2, maks 512), `compute_tfidf_size` (vocab yang menutup 95% kata × 3) |
| 5 | Data Preparation | `clean_text` (unescape HTML, `REMOVE_PATTERNS`, URL → `URL`, normalisasi unicode), `build_frame`, persiapan target (ordinal / multi-label / regresi), dedup, grouping teks identik, holdout + fold |
| 6 | Baseline: TF-IDF + Linear | `TfidfFeatures` (word 1–2 gram + char 2–5 gram), `LinearHead` (Logistic Regression / Ridge / One-vs-Rest), diagnostik, n-gram paling prediktif, export |
| 7 | Transformer Setup | tokenizer tiap model, cakupan panjang token, `encode` dengan truncation `head_tail` (`special_wrap` menjaga token spesial `[CLS]`/`[SEP]`) |
| 8 | Transformer Fine-Tuning | `TextModel` (encoder → mean pooling → dropout → linear), `run_fold` (AdamW, cosine warmup, mixed precision, evaluasi `EVALS_PER_EPOCH` kali per epoch, early stopping), `hp` menggabungkan `MODEL_OVERRIDES`, learning curve |
| 9 | Ensemble and Model Selection | `blend` bobot optimal di OOF, `tune_decision` |
| 10 | Final Evaluation on Holdout | skor holdout semua kandidat |
| 11 | Model Interpretation | `occlusion`: hapus kata satu per satu dan lihat perubahan prediksi; `highlight` mewarnai kata (oranye = mendorong, biru = melawan) → `occlusion_examples.html` |
| 12 | Error Analysis | prediksi paling salah, confusion matrix |
| 13 | Predictions and Export | `follow_sample`, `export_result`, report |

### Settings pipeline

| Setting | Default | Arti |
|---|---|---|
| `EDA_DIR` | `'eda-output'` | folder `eda_decisions.json` |
| `TEXT_COL`, `LABEL_COL`, `ID_COL` | `'auto'` | kolom teks, label, ID |
| `TEXT_PAIR_COL` | `None` | kolom teks kedua (task pasangan kalimat, misal NLI / similarity) |
| `SUBMISSION_FORMAT` | `'auto'` | `'label'` / `'proba'` |
| `TASK` | `'auto'` | `'classification'`, `'regression'`, `'ordinal'`, `'multilabel'` |
| `ORDINAL_ORDER`, `MULTILABEL_SEP` | `None` | urutan kelas ordinal; separator multi-label |
| `TOP_K_CLASSES` | `6` | jumlah kelas yang ditampilkan di grafik per kelas |
| `DEBUG_SAMPLE` | `None` | subsample untuk uji cepat |
| `REMOVE_PATTERNS`, `DEDUP`, `GROUP_IDENTICAL` | `'auto'` | pembersihan & penanganan duplikat dari EDA |
| `CV_GROUP` | `None` | kolom group untuk fold |
| `N_FOLDS`, `HOLDOUT_SIZE` | `5`, `0.15` | fold CV dan porsi holdout |
| `METRIC` | `'auto'` | macro F1 (klasifikasi), RMSE (regresi), QWK (ordinal), micro F1 (multi-label) |
| `CLASS_WEIGHTS` | `'auto'` | bobot kelas di loss |
| `TUNE_DECISION` | `True` | tuning aturan label di OOF |
| `TFIDF_MAX_FEATURES` | `'auto'` | ukuran vocab TF-IDF word |
| `TFIDF_CHAR` | `True` | tambah TF-IDF karakter 2–5 gram (tahan typo & slang) |
| `BASELINE_C`, `BASELINE_ALPHA` | `4.0`, `1.0` | regularisasi Logistic Regression / Ridge |
| `MODELS` | `['auto']` | key dari `MODEL_REGISTRY`, misal `['indobert', 'xlm-roberta']`; `[]` = baseline saja |
| `MODEL_REGISTRY` | 13 model | key → checkpoint Hugging Face (boleh ditambah, atau isi path folder lokal) |
| `MODEL_OVERRIDES` | `{'indobert-large': {...}}` | hyperparameter khusus per model (`lr`, `batch_size`, `epochs`, `max_length`, ...) |
| `MAX_LENGTH` | `'auto'` | panjang token maksimum |
| `TRUNCATION`, `HEAD_RATIO` | `'head_tail'`, `0.5` | teks panjang: simpan 50% awal + 50% akhir (`'head'` = awal saja) |
| `POOLING` | `'mean'` | mean pooling hidden state terakhir |
| `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM` | `3`, `16`, `1` | budget training; batch efektif = `BATCH_SIZE × GRAD_ACCUM` |
| `LR`, `HEAD_LR_MULT` | `2e-5`, `5.0` | learning rate encoder; head memakai `LR × 5` |
| `WEIGHT_DECAY`, `WARMUP_RATIO`, `LABEL_SMOOTHING`, `DROPOUT` | `0.01`, `0.1`, `0.0`, `0.1` | regularisasi & warmup |
| `EVALS_PER_EPOCH`, `PATIENCE` | `2`, `3` | evaluasi per epoch; early stopping setelah 3 evaluasi tanpa perbaikan |
| `MIXED_PRECISION` | `'auto'` | fp16/bf16 di GPU |
| `GRAD_CHECKPOINT` | `False` | hemat memori GPU (lebih lambat) |
| `TRAIN_FOLDS` | `[0]` | fold yang di-fine-tune; `[0,1,2,3,4]` = semua (wajib kalau OOF dipakai untuk stacking di multimodal) |
| `SAVE_WEIGHTS` | `False` | simpan bobot ke `final/models/` |
| `ENSEMBLE` | `True` | blend baseline + transformer |

### Model di `MODEL_REGISTRY`

| Key | Checkpoint | Cocok untuk |
|---|---|---|
| `indobert` / `indobert-large` | `indobenchmark/indobert-base-p1` / `-large-p1` | Indonesia |
| `indolem`, `indoroberta` | `indolem/indobert-base-uncased`, `flax-community/indonesian-roberta-base` | Indonesia |
| `roberta`, `twitter-roberta` | `FacebookAI/roberta-base`, `cardiffnlp/twitter-roberta-base-sentiment-latest` | Inggris, medsos |
| `deberta-v3`, `modernbert` | `microsoft/deberta-v3-base`, `answerdotai/ModernBERT-base` | Inggris (akurasi tinggi / teks panjang) |
| `xlm-roberta`, `mdeberta`, `multilingual-e5` | multilingual | campuran bahasa |
| `distilbert`, `bert-tiny` | model kecil | uji cepat |

### Output (`pipeline-output/final/`)

| File | Isi |
|---|---|
| `submission.csv` | prediksi test mengikuti `sample_submission` |
| `leaderboard.csv`, `final_metrics.*` | skor CV dan holdout |
| `training_history.csv` | loss & metrik per evaluasi tiap fold |
| `occlusion_examples.html` | contoh kata penting per prediksi (buka di browser) |
| `oof_predictions.csv`, `holdout_predictions.csv`, `test_predictions_full.csv` | prediksi per baris (OOF dipakai untuk stacking di multimodal) |
| `classification_report.csv`, `worst_predictions.csv` | analisis error |
| `decisions.csv`, `report.md`, `config.json` | keputusan, ringkasan, konfigurasi |

## Troubleshooting khusus NLP

| Gejala | Solusi |
|---|---|
| `CUDA out of memory` | `BATCH_SIZE = 8` + `GRAD_ACCUM = 2`, turunkan `MAX_LENGTH`, `GRAD_CHECKPOINT = True`, atau model base (bukan large) |
| Model dilewati (`[warn] ... skipped`) | checkpoint gagal diunduh (internet / nama salah); notebook lanjut dengan model lain |
| Skor transformer kalah dari TF-IDF | data kecil atau label noise; coba `EPOCHS` lebih banyak, `LR = 3e-5`, model bahasa yang sesuai, atau pakai blend (sering menang) |
| Loss tidak turun / prediksi satu kelas | `LR` terlalu besar (coba `1e-5`), atau kelas sangat timpang (cek `CLASS_WEIGHTS`) |
| Teks berisi tag/mention yang mengganggu | tambahkan regex ke `REMOVE_PATTERNS` |
| Training terlalu lama | `TRAIN_FOLDS = [0]`, `EPOCHS = 2`, model `distilbert` / `indobert` base |
