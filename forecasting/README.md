# Forecasting

Forecast satu atau banyak time series (format panjang: satu baris per tanggal per series).

## Isi folder

| File | Fungsi |
|---|---|
| `eda_forecasting.ipynb` | EDA time series + `eda-output/eda_decisions.json` |
| `pipeline_forecasting.ipynb` | baseline statistik → LightGBM / XGBoost / CatBoost global → backtest → ensemble → forecast + interval |
| `full_forecasting.ipynb` | EDA + pipeline dalam satu notebook |
| `build_*.py` | skrip builder notebook |

**Cara pakai:** taruh `train.csv` (+ `test.csv` berisi tanggal masa depan) di `forecasting/data/`, lalu jalankan EDA → pipeline. Kalau tidak ada data, `DEMO_IF_MISSING = True` membuat data demo.

## EDA: alur section

| # | Section | Isi |
|---|---|---|
| 4 | Structure and Frequency | `infer_freq` (frekuensi dari selisih tanggal), umur series |
| 5 | Completeness and Intermittency | gap, nol, ADI/CV² (`adi_cv2`) untuk demand intermittent |
| 6 | Target Distribution and Transform | saran `log1p` / `sqrt` |
| 7–8 | Series Overview, Decomposition | STL, kekuatan trend & musiman (`strength`) |
| 9–10 | Seasonal Patterns, Autocorrelation | profil musiman, periodogram, ACF/PACF → lag |
| 11 | Stationarity | ADF + KPSS (`stationarity`, `n_diffs`) |
| 12 | Anomalies and Level Shifts | outlier (z > `OUTLIER_Z`), perubahan level |
| 13 | Exogenous and Calendar Drivers | variabel eksogen, hari spesial |
| 14 | Cross-Series Structure | korelasi antar series |
| 15 | Baseline Forecastability | skor naive/seasonal naive, `spectral_entropy` |
| 16 | EDA Summary | `eda_decisions.json` |

**Key penting `eda_decisions.json`:** `freq`, `horizon`, `seasonal_periods`, `transform`, `fill_missing`, `lags_direct` / `lags_recursive`, `rolling_windows`, `exog_known_future` (eksogen yang diketahui di masa depan), `special_dates`, `cv` (jumlah fold), `metric`, `best_naive_wape`, `global_model_recommended`.

## Pipeline: alur section

| # | Section | Isi |
|---|---|---|
| 4 | EDA-Driven Decisions | `compute_freq`, `compute_transform`, `compute_lags`, `compute_folds` kalau EDA tidak ada |
| 5 | Data Preparation | `prep` (parse tanggal, gabung ID jadi `series`), isi tanggal hilang |
| 6 | Feature Engineering | `build_features`: lag, rolling mean/std, EWM, same-season, intermittency, kalender, `event_features` (jarak ke `EVENT_DATES`), eksogen |
| 7 | Backtesting Design | fold expanding window (`val_mask`, `train_mask`) |
| 8 | Statistical Baselines | `baseline_forecast`: naive, seasonal naive, moving average, ETS |
| 9 | Gradient Boosting | kelas `GBM`, `volume_weights`, `recursive_predict` (strategi recursive) |
| 10–12 | Ensemble, Diagnostics, Interpretation | blend, error per series / kelas demand, importance per keluarga fitur (`family`) |
| 13–14 | Final Forecast, Export | refit penuh, forecast + interval, `to_submission` |

## Settings pipeline

| Setting | Default | Arti |
|---|---|---|
| `DATE_COL`, `TARGET_COL`, `ID_COLS` | `'auto'` | kolom tanggal, target, pembentuk series (`[]` = satu series) |
| `EXOG_COLS` | `'auto'` | covariate; `AGG` = agregasi tanggal duplikat (`'sum'`) |
| `FREQ`, `HORIZON`, `SEASONAL_PERIODS` | `'auto'` | frekuensi, langkah forecast, periode musiman |
| `TRANSFORM`, `FILL_MISSING` | `'auto'` | transformasi target, cara isi tanggal kosong |
| `STRATEGY` | `'direct'` | `'direct'` (lag ≥ horizon) atau `'recursive'` |
| `LAGS`, `ROLL_WINDOWS` | `'auto'` | lag dan jendela rolling |
| `EVENT_DATES`, `EVENT_WINDOW` | Lebaran 2021–2027, `14` | tanggal spesial dan jendela hari di sekitarnya |
| `N_FOLDS`, `METRIC` | `'auto'` | fold backtest; `wape`/`smape`/`mase`/`rmse`/`mae` |
| `NON_NEGATIVE` | `'auto'` | potong prediksi negatif |
| `MODELS`, `BASELINES` | 3 GBM, 4 baseline | model yang dijalankan |
| `ETS_MAX_SERIES` | `200` | ETS hanya untuk ≤ 200 series (lambat) |
| `OBJECTIVE`, `WEIGHT_BY_VOLUME` | `'auto'` | loss (misal Tweedie untuk data banyak nol), bobot per volume series |
| `N_ESTIMATORS`, `LEARNING_RATE`, `EARLY_STOPPING` | `3000`, `0.03`, `150` | budget GBM |
| `REFIT_IN_FOLD` | `True` | refit tiap fold dengan iterasi terbaik |
| `USE_GPU`, `ENSEMBLE`, `INTERVAL` | `False`, `True`, `0.8` | GPU, blend, level prediction interval |

**Output (`pipeline-output/final/`):** `submission.csv`, `forecast_full.csv` (forecast + interval), `leaderboard.csv`, `backtests/`, `error_by_series.csv`, `error_by_demand_class.csv`, `feature_importance.csv`, `decisions.csv`, `report.md`.

## Troubleshooting

| Gejala | Solusi |
|---|---|
| Frekuensi salah (misal harian terbaca mingguan) | isi `FREQ = 'D'` |
| Horizon salah | isi `HORIZON`, atau sediakan `test.csv` berisi tanggal masa depan |
| Series pendek, error fold | turunkan `N_FOLDS` atau `HORIZON` |
| Prediksi negatif | `NON_NEGATIVE = True` |
| Libur nasional selain Lebaran | tambahkan ke `EVENT_DATES` |
