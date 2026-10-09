# Multimodal: Tabular + Teks

Pipeline tabular (lihat [tabular/README.md](../../tabular/README.md) untuk semua section dan setting dasarnya), ditambah fitur dari kolom teks.

| File | Fungsi |
|---|---|
| `eda_multimodal.ipynb` | EDA tabular + section 15 **Modality Signal Check** (skor teks saja, tabular saja, gabungan) → `text_mode` |
| `pipeline_multimodal.ipynb` | pipeline tabular + fitur teks + ablation per modalitas |
| `full_multimodal.ipynb`, `build_*.py` | gabungan, builder |

## Yang berbeda dari tabular

| Bagian kode | Arti |
|---|---|
| `auto_text_mode` | pilih `TEXT_MODE` dari EDA (`text_mode`), kalau tidak ada: `'both'` saat ada GPU / data kecil |
| `load_encoder`, `encode`, `embed_column` | embedding transformer (mean pooling) → PCA `EMBED_DIM`, di-cache di `pipeline-output/cache/` |
| `embed_lookup` | ambil embedding dari cache per teks unik |
| fitur `nlp_*` | prediksi OOF pipeline NLP (stacking) dari `NLP_OOF_DIR` |
| Modality Ablation (section 8) | latih ulang dengan tabular saja / + tiap keluarga teks → `modality_ablation.csv` |

## Settings tambahan

| Setting | Default | Arti |
|---|---|---|
| `TEXT_MODE` | `'auto'` | `'tfidf'`, `'embed'`, `'both'`, `'none'` |
| `EMBED_MODEL` | `intfloat/multilingual-e5-base` | encoder Hugging Face |
| `EMBED_DIM`, `EMBED_MAX_LENGTH`, `EMBED_BATCH` | `32`, ..., ... | dimensi PCA, panjang token, batch |
| `NLP_OOF_DIR` | `None` | folder `pipeline-output/final` dari NLP (latih NLP dengan `TRAIN_FOLDS = [0,1,2,3,4]`) |
| `MODALITY_ABLATION` | `True` | jalankan ablation |

**Troubleshooting:** embedding lambat di CPU → `TEXT_MODE = 'tfidf'`; model embedding gagal diunduh → nyalakan internet atau ganti `EMBED_MODEL` ke path lokal; peringatan stacking bocor → semua fold NLP harus dilatih.
