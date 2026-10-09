# Multimodal: Forecasting + Teks

Pipeline forecasting (lihat [forecasting/README.md](../../forecasting/README.md)), ditambah fitur dari teks bertanggal.

| File | Fungsi |
|---|---|
| `eda_forecast_text.ipynb` | EDA forecasting + section 16 **Text Signal** (korelasi teks di t−lag vs target, lag terbaik ≥ horizon) |
| `pipeline_forecast_text.ipynb` | forecasting + fitur teks ter-lag + Text Modality Ablation |
| `full_forecast_text.ipynb`, `build_*.py` | gabungan, builder |

## Kode tambahan

| Fungsi | Arti |
|---|---|
| `find_text_file` | cari file teks (nama mengandung `news`, `text`, `review`, `tweet`, `post`, `berita`, `ulasan`) |
| `load_documents` | baca dokumen: tanggal, teks, ID series (opsional) |
| `to_bucket` | bulatkan tanggal dokumen ke periode forecast (`FREQ`) |
| `word_mean` | rata-rata jumlah kata (deteksi kolom teks) |

Fitur teks = jumlah dokumen, panjang rata-rata, komponen embedding per (series, periode), lalu di-lag `TEXT_LAG` (selalu ≥ horizon, jadi tidak bocor) dan di-rolling `TEXT_WINDOWS`.

## Settings tambahan

| Setting | Default | Arti |
|---|---|---|
| `TEXT_*_PATH` | `None` | file teks (kalau tidak ketemu otomatis) |
| `TEXT_COL`, `TEXT_DATE_COL`, `TEXT_ID_COLS` | `'auto'` | kolom teks, tanggal, ID series di file teks |
| `TEXT_FEATURES` | `'auto'` | `'embed'` atau `'tfidf'` |
| `TEXT_DIM` | `8` | komponen per dokumen |
| `TEXT_LAG`, `TEXT_WINDOWS` | `'auto'` | lag dan rolling dari EDA |
| `EMBED_MODEL` | `intfloat/multilingual-e5-base` | encoder |
| `TEXT_ABLATION` | `True` | bandingkan dengan vs tanpa teks → `text_ablation.csv` |

**Troubleshooting:** teks tidak ditemukan → isi `TEXT_*_PATH`; `text_useful = False` di EDA → teks tidak membantu, wajar kalau ablation tidak naik.
