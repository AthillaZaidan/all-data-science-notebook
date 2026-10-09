# Tabular Foundation Model (Causilo)

Pipeline tabular yang sama (lihat [tabular/README.md](../tabular/README.md)) dengan tambahan model **Causilo**: transformer pretrained yang memprediksi lewat in-context learning (tanpa training ulang). Pakai EDA dari `tabular/eda_tabular.ipynb`.

| File | Fungsi |
|---|---|
| `pipeline_foundation.ipynb` | baseline → Causilo + GBM → ensemble → holdout → interval → submission |
| `full_foundation.ipynb` | EDA tabular + pipeline ini |
| `build_pipeline_foundation.py` | builder |

## Kode tambahan

| Bagian | Arti |
|---|---|
| `CausiloModel` | wrapper `CausiloClassifier` / `CausiloRegressor` dengan antarmuka yang sama seperti model lain (fit/predict per fold) |
| Context-Size Learning Curve | skor vs jumlah baris konteks, Causilo vs GBM → `context_curve.csv` |
| Prediction Intervals | kuantil Causilo untuk regresi → `intervals.json`, `test_intervals.csv` |
| Foundation Model Importance | permutation importance → `causilo_permutation_importance.csv` |

## Settings tambahan

| Setting | Default | Arti |
|---|---|---|
| `MODELS` | `['causilo', 'lightgbm']` | model yang dilatih |
| `CAUSILO_ESTIMATORS` | `'auto'` | anggota ensemble (8 GPU, 4 CPU) |
| `CAUSILO_DEVICE` | `'auto'` | `'cuda'`, `'mps'`, `'cpu'` |
| `MAX_CONTEXT_ROWS` | `50_000` | baris konteks maksimum (data besar di-subsample) |
| `CONTEXT_CURVE`, `PERM_IMPORTANCE`, `INTERVAL` | `True`, `True`, `0.8` | analisis tambahan |

**Troubleshooting:** error saat fit pertama → butuh internet untuk unduh bobot `nums-ai/causilo`; lambat → pakai GPU atau turunkan `MAX_CONTEXT_ROWS`; cek lisensi bobot (non-komersial) di aturan lomba.
