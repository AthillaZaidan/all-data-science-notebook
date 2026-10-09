# Tabular

Template untuk data tabel (baris = sampel, kolom = fitur): regresi, klasifikasi biner, multiclass, ordinal, multi-label, dan multi-target.

## Isi folder

| File | Fungsi |
|---|---|
| `eda_tabular.ipynb` | EDA lengkap + menulis `eda-output/eda_decisions.json` |
| `pipeline_tabular.ipynb` | baseline linear → LightGBM / XGBoost / CatBoost → tuning → ensemble → holdout → SHAP → submission |
| `full_tabular.ipynb` | EDA + pipeline dalam satu notebook (Part 1 dan Part 2) |
| `build_eda_tabular.py`, `build_pipeline_tabular.py` | skrip builder yang menghasilkan notebook di atas. Edit ini kalau mau mengubah template, lalu jalankan `python build_eda_tabular.py` |

## Cara pakai cepat

1. Taruh `train.csv`, `test.csv`, `sample_submission.csv` di `tabular/data/`, atau isi `KAGGLE_PATH` dengan folder dataset.
2. *Run All* `eda_tabular.ipynb`. Kalau target salah tebak, isi `TARGET = 'nama_kolom'`.
3. *Run All* `pipeline_tabular.ipynb`. Hasil di `pipeline-output/final/submission.csv`.

Atau cukup *Run All* `full_tabular.ipynb`.

---

## EDA (`eda_tabular.ipynb`)

### Alur section

| # | Section | Yang dikerjakan kode |
|---|---|---|
| 1 | Introduction | penjelasan tujuan dan cara konfigurasi |
| 2 | Initialization | install library, import, `seed_everything`, `Settings`, lalu `discover_files` mencari train/test/sample dan `resolve_columns` mengisi target & ID `'auto'` |
| 3 | Plotting Toolkit | palet warna, `save_fig` (menyimpan grafik bernomor ke `eda-output/figures/`), `save_table` (CSV ke `eda-output/tables/`), helper plot |
| 4 | Data Overview | `infer_roles` menentukan peran tiap kolom (numeric / categorical / text / id / datetime), profil kolom (tipe, missing, unik, nilai teratas) |
| 5 | Data Quality | missing value, *disguised missing* (sentinel seperti `-999`, dicari `out_of_range_sentinels`), duplikat (`row_hash`), kolom konstan / hampir konstan / kardinalitas tinggi, outlier |
| 6 | Target Analysis | distribusi target, skewness → saran transformasi (`suggest_transform`), imbalance kelas |
| 7 | Numerical Features | statistik deskriptif, histogram, skewness |
| 8 | Categorical Features | frekuensi level, level langka |
| 9 | Feature vs Target | ranking asosiasi (mutual information, korelasi, Kruskal-Wallis lewat `safe_kruskal`), plot fitur teratas vs target |
| 10 | Multivariate Structure | matriks korelasi, pasangan sangat berkorelasi + VIF, clustering fitur, Cramér's V antar kategori (`cramers_v`), pairplot, PCA |
| 11 | Temporal and Segment Analysis | tren volume & target terhadap waktu (`TIME_COL`), heatmap segmen × periode (`GROUP_COL`) |
| 12 | Text Columns Snapshot | ringkasan kolom teks bebas (panjang, kata umum) |
| 13 | Train vs Test Drift | PSI per fitur (`psi`) dan *adversarial validation* (model membedakan train vs test; AUC tinggi = drift) |
| 14 | Baseline Signal Check | CV cepat: split acak vs per grup (deteksi leakage), fitur paling penting |
| 15 | EDA Summary | `eda_summary.md` + `eda_decisions.json` untuk pipeline |

### Settings EDA

| Setting | Default | Arti |
|---|---|---|
| `KAGGLE_PATH` / `COLAB_PATH` / `LOCAL_PATH` | placeholder / `'data'` | folder dataset (atau file train) per lingkungan |
| `TEST_*_PATH` | `None` | file test kalau namanya tidak diawali `test` |
| `READ_KWARGS` | `{}` | argumen tambahan untuk pembaca file, misal `{'sep': ';'}` |
| `TARGET` | `'auto'` | kolom target. `'auto'` = dari `sample_submission` / kolom yang tidak ada di test. `None` = EDA tanpa target |
| `TASK` | `'auto'` | `'regression'`, `'binary'`, `'multiclass'` |
| `ID_COLS` | `'auto'` | kolom ID (dikenali dari nama: `id`, `*_id`, `key`, ...) |
| `DROP_COLS` | `[]` | kolom yang dibuang sebelum analisis |
| `DATE_COLS` | `[]` | kolom tanggal tambahan (yang terlihat seperti tanggal sudah otomatis di-parse) |
| `TIME_COL` | `None` | kolom waktu untuk analisis tren & drift |
| `GROUP_COL` | `None` | kolom segmen untuk analisis per grup |
| `CV_GROUP` | `None` | kolom yang dicurigai bocor antar fold (misal `user_id`) |
| `NUM_COLS`, `CAT_COLS`, `TEXT_COLS` | `'auto'` | override peran kolom |
| `SAMPLE_N` | `20_000` | baris sampel untuk plot berat |
| `DISCRETE_MAX` | `15` | kolom numerik dengan ≤ 15 nilai unik diperlakukan sebagai kategori |
| `CLASS_MAX` | `20` | target dengan ≤ 20 kelas dianggap klasifikasi |
| `TOP_N_CATS` | `15` | jumlah level yang ditampilkan per kategori |
| `MAX_PLOT_COLS` | `30` | batas kolom yang diplot |
| `HIGH_CORR` | `0.80` | ambang pasangan fitur "sangat berkorelasi" |
| `PAIRPLOT_K` | `5` | jumlah fitur teratas di pairplot |
| `MIN_PERIOD_N` | `30` | minimal baris per periode di analisis temporal |
| `TIME_RANGE` | `(1950, 2025)` | rentang tahun valid (deteksi tanggal / tahun aneh) |
| `RUN_BASELINE` | `True` | jalankan section 14 |
| `N_FOLDS` | `5` | fold untuk baseline signal check |

### Isi `eda_decisions.json`

| Key | Arti | Dipakai pipeline untuk |
|---|---|---|
| `target`, `task` | target dan jenis task | `TARGET`, `TASK` |
| `id_cols`, `num_cols`, `cat_cols`, `text_cols`, `datetime_cols` | peran kolom | peran fitur |
| `drop_cols` | kolom konstan / tidak berguna | `DROP_COLS` |
| `sentinels` | `{kolom: [nilai placeholder]}` | `SENTINELS` (diubah jadi NaN) |
| `high_cardinality` | kategori dengan level sangat banyak | penanganan kategori |
| `dedup` | ada baris duplikat persis? | `DEDUP` |
| `cv_group`, `group_vs_random_gap` | kolom group dan selisih skor split acak vs per grup | `CV_GROUP`, `CV_SCHEME` |
| `conflicting_targets` | baris identik dengan target beda | info label noise |
| `time_col`, `adversarial_auc` | kolom waktu, AUC train-vs-test | `TIME_COL`, `CV_SCHEME = 'time'` kalau drift kuat |
| `cv_scheme` | `kfold` / `stratified` / `group` / `time` | `CV_SCHEME` |
| `target_transform` | misal `log1p` untuk target miring | `TARGET_TRANSFORM` |
| `metric` | metrik yang disarankan | `METRIC` |
| `class_weights` | perlu class weight? | `CLASS_WEIGHT` |
| `weak_features` | fitur tanpa sinyal | `DROP_WEAK` |
| `top_features`, `high_corr_pairs` | fitur terkuat, pasangan berkorelasi | laporan |

---

## Pipeline (`pipeline_tabular.ipynb`)

### Alur section

| # | Section | Yang dikerjakan kode |
|---|---|---|
| 1 | Introduction | tujuan, tabel konfigurasi, metrik, pendekatan |
| 2 | Initialization | install, import, `Settings`, `discover_files`, baca data |
| 3 | Toolkit / Metrics | `score_all` (semua metrik sesuai task), `primary` (metrik utama), `better`, `decide` (resolver `'auto'`: user > EDA > computed, dicatat di tabel Decisions) |
| 4 | Data Preparation | baca keputusan EDA, `guess_target`, `infer_roles`, persiapan target (`fwd` / `inv` untuk transformasi `log1p`, `ordinal_order` untuk ordinal), hapus duplikat, split holdout |
| 5 | Feature Engineering | `custom_features` (**tempat menambah fitur sendiri**) dan `FeatureBuilder` (frequency encoding, flag missing, SVD dari TF-IDF kolom teks) |
| 6 | Validation Strategy | `make_folds`: KFold / Stratified / Group / TimeSeries sesuai `CV_SCHEME` |
| 7 | Baseline | `LinearModel` (Linear/Logistic Regression + one-hot + scaling), `cross_validate` (OOF + prediksi holdout & test), diagnostik, koefisien, export ke `pipeline-output/baseline/` |
| 8 | Gradient Boosting | `LGBModel`, `XGBModel`, `CatModel` (turunan `GBMModel`) dengan early stopping per fold; kategori langka digabung ke `(other)` (`tree_frame`) |
| 9 | Hyperparameter Tuning | Optuna (`suggest` mendefinisikan ruang pencarian), kalau `TUNE = True` |
| 10 | Ensemble and Model Selection | `blend` dengan bobot optimal di OOF, leaderboard, tuning aturan keputusan (`tune_decision`: threshold / class scale / cut-point ordinal / threshold per label) |
| 11 | Final Evaluation on Holdout | skor holdout + bootstrap 95% CI vs baseline |
| 12 | Model Interpretation | feature importance, SHAP summary & dependence |
| 13 | Error Analysis | prediksi terburuk, error per segmen / desil target |
| 14 | Predictions and Export | `predict_new` (fungsi inferensi untuk data baru), `follow_sample` (format submission), report, target tambahan (`fit_current_target`) |

### Settings pipeline

| Setting | Default | Arti |
|---|---|---|
| `EDA_DIR` | `'eda-output'` | folder berisi `eda_decisions.json` |
| `TARGET` | `'auto'` | target; list (`['y1', 'y2']`) = multi-target |
| `TASK` | `'auto'` | `'regression'`, `'classification'`, `'ordinal'` |
| `ORDINAL_ORDER` | `None` | urutan kelas ordinal string, misal `['rendah', 'sedang', 'tinggi']` |
| `MULTILABEL_SEP` | `None` | separator kalau satu kolom berisi banyak label (`'a|b'`) |
| `ID_COLS`, `DROP_COLS` | `'auto'` | kolom yang tidak dipakai sebagai fitur |
| `SENTINELS` | `'auto'` | `{kolom: [nilai]}` yang diubah jadi NaN |
| `DATE_COLS` | `[]` | kolom tanggal; diubah jadi fitur `_year`, `_month`, `_dow` (hari dalam minggu), `_doy` (hari dalam tahun) |
| `NUM_COLS`, `CAT_COLS`, `TEXT_COLS` | `'auto'` | override peran kolom |
| `SUBMISSION_ID` | `None` | kolom ID di `submission.csv` (default: dari sample) |
| `SUBMISSION_FORMAT` | `'auto'` | `'label'` / `'proba'`; `'auto'` mengikuti sample |
| `CV_SCHEME` | `'auto'` | `'kfold'`, `'stratified'`, `'group'`, `'time'` |
| `CV_GROUP`, `TIME_COL` | `'auto'` | kolom group / waktu untuk CV |
| `N_FOLDS` | `5` | jumlah fold |
| `HOLDOUT_SIZE` | `0.2` | porsi data untuk evaluasi akhir (tidak dipakai training/tuning) |
| `DEDUP` | `'auto'` | hapus baris duplikat persis |
| `METRIC` | `'auto'` | RMSE (regresi), ROC AUC (biner), macro F1 (multiclass), QWK (ordinal); bisa `'mae'`, `'f1_macro'`, `'logloss'`, ... |
| `TARGET_TRANSFORM` | `'auto'` | `'log1p'` atau `'none'` |
| `CLASS_WEIGHT` | `'auto'` | bobot kelas untuk data imbalance |
| `TUNE_DECISION` | `True` | tuning aturan label di OOF |
| `DROP_WEAK` | `'auto'` | buang fitur lemah menurut EDA (`False` untuk mematikan) |
| `DISCRETE_MAX` | `15` | numerik dengan ≤ 15 nilai unik → kategori |
| `MAX_OHE_LEVELS` | `30` | kategori dengan ≤ 30 level di-one-hot untuk model linear |
| `GBM_MAX_CAT_LEVELS` | `200` | level kategori maksimum untuk GBM; sisanya `(other)` |
| `ADD_FREQ_ENCODING` | `True` | tambah fitur frekuensi tiap level kategori |
| `ADD_MISSING_FLAGS` | `True` | tambah flag 0/1 "nilai ini kosong" |
| `TEXT_SVD_DIM` | `16` | komponen SVD dari TF-IDF per kolom teks (`0` = mati) |
| `BASELINE` | `'linear'` | baseline yang selalu jalan |
| `MODELS` | `['lightgbm', 'xgboost', 'catboost']` | model GBM yang dilatih (`[]` = baseline saja) |
| `N_ESTIMATORS`, `LEARNING_RATE`, `EARLY_STOPPING` | `3000`, `0.05`, `100` | iterasi maksimum, learning rate, patience early stopping |
| `USE_GPU` | `False` | GBM di GPU |
| `MODEL_PARAMS` | `{...: {}}` | override hyperparameter per model, misal `{'lightgbm': {'num_leaves': 63}}` |
| `TUNE`, `TUNE_MODEL`, `N_TRIALS`, `TUNE_TIMEOUT`, `TUNE_FOLDS` | `False`, `'lightgbm'`, `30`, `900`, `3` | tuning Optuna |
| `ENSEMBLE` | `True` | blend semua model |

### Menambah fitur sendiri

Edit fungsi `custom_features` di section 5. `raw` adalah data mentah, `feats` adalah fitur yang sudah dibuat. Fungsi ini dipanggil untuk train, holdout, dan test, jadi fitur selalu konsisten.

```python
def custom_features(raw, feats):
    feats['harga_per_m2'] = raw['harga'] / raw['luas']
    feats['umur_bangunan'] = 2025 - raw['tahun_bangun']
    return feats
```

### Output (`pipeline-output/final/`)

| File | Isi |
|---|---|
| `submission.csv` | prediksi test, format mengikuti `sample_submission` |
| `leaderboard.csv` | skor CV semua model |
| `final_metrics.csv` / `.json` | skor holdout + improvement vs baseline + bootstrap CI |
| `decisions.csv` | semua keputusan `'auto'` + sumber + alasan |
| `oof_predictions.csv`, `holdout_predictions.csv`, `test_predictions_full.csv` | prediksi per baris |
| `ensemble_weights.json` | bobot blend |
| `feature_importance.csv`, `feature_inventory.csv` | importance dan asal tiap fitur |
| `classification_report.csv`, `error_by_*.csv`, `worst_predictions.csv` | analisis error |
| `best_params_*.json`, `optuna_trials_*.csv` | hasil tuning (kalau `TUNE = True`) |
| `multi_target_metrics.csv`, `multilabel_scores.json` | kalau multi-target / multi-label |
| `report.md`, `config.json` | ringkasan siap salin ke laporan, konfigurasi run |
| `figures/`, `models/`, `oof/` | grafik, model tersimpan, OOF per model |

## Troubleshooting khusus tabular

| Gejala | Solusi |
|---|---|
| Fitur ID (misal `customer_no`) jadi fitur paling penting | isi `ID_COLS = ['customer_no']`; ID bukan fitur |
| Skor CV jauh lebih tinggi dari skor leaderboard | kemungkinan leakage: cek section 14 EDA (selisih split acak vs grup), pakai `CV_SCHEME = 'group'` + `CV_GROUP`, atau `'time'` kalau data berurutan waktu |
| Baseline memprediksi konstan | cek kolom kategori dengan level unik per baris (harus jadi ID / dibuang) |
| Kolom angka terbaca sebagai teks | ada karakter non-angka (koma ribuan, `"-"`); bersihkan di `custom_features` atau saat baca (`READ_KWARGS = {'thousands': ','}`) |
| CatBoost lambat | hapus dari `MODELS`, atau `N_ESTIMATORS` lebih kecil |
| Tuning terlalu lama | turunkan `N_TRIALS` / `TUNE_TIMEOUT` |
