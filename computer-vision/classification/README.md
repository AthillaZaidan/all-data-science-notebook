# Klasifikasi Gambar (DINOv3 ViT-L)

| File | Fungsi |
|---|---|
| `pipeline_cv_classification.ipynb` | linear probe → fine-tuning → blend → holdout → evidence map → submission |
| `build_pipeline_cv_classification.py` | builder |

**Data:** `train.csv` (nama file/ID gambar + label) + folder gambar, atau `train/<kelas>/*.jpg`. Task: binary, multiclass, multi-label, regresi.

## Alur section

| # | Section | Isi kode |
|---|---|---|
| 2 | Initialization | `find_root`, `ImageIndex` (cari gambar dari path/nama/ID), `detect_image_col`, `attach_paths` |
| 3 | Toolkit | `decide`, metrik (`score`, `to_labels`) |
| 4 | Data Overview and Decisions | `resolve_label`, `guess_task`, ukuran gambar (`image_info`), contoh gambar, duplikat (`file_hash`), resolusi `IMG_SIZE`, `FINETUNE`, dll. |
| 5 | Data Preparation | holdout + fold (`grouped_split`, duplikat digrup), `ImageDataset`, `build_transforms` (augmentasi) |
| 6 | DINOv3 Backbone | `hf_token`, `load_backbone` (fallback otomatis), `encode` (CLS + patch), peta PCA patch |
| 7 | Linear Probe | `extract_features` [CLS ; mean patch] + flip TTA (di-cache), Logistic/Ridge per fold, t-SNE |
| 8 | Fine-Tuning | `DinoClassifier`, `param_groups` (blok terakhir + LLRD), `train_fold`, `predict` |
| 9 | Ensemble | blend probe + fine-tune, `tune_decision` |
| 10–12 | Holdout, Interpretation, Error Analysis | skor, confusion matrix, `evidence_weights` (kontribusi patch), gambar paling salah |
| 13 | Export | `follow_sample`, `write_submission`, `report.json` |

## Settings penting

| Setting | Default | Arti |
|---|---|---|
| `IMAGE_COL`, `LABEL_COL`, `ID_COL`, `GROUP_COL` | `'auto'`/`None` | kolom gambar, label, ID, grup |
| `TASK`, `MULTILABEL_SEP` | `'auto'`, `None` | jenis task |
| `HF_TOKEN`, `BACKBONE`, `BACKBONE_FALLBACK` | `None`, `dinov3-vitl16`, `dinov2-large` | backbone |
| `IMG_SIZE` | `'auto'` | resolusi (224/336/448) |
| `PROBE`, `PROBE_C_GRID` | `True` | linear probe dan grid regularisasi |
| `FINETUNE`, `UNFREEZE_LAST_N` | `'auto'` | fine-tune (CUDA saja) dan jumlah blok terakhir yang dilatih |
| `EPOCHS`, `BATCH_SIZE`, `LR`, `HEAD_LR`, `LLRD` | `10`, `16`, `2e-5`, `1e-3`, `0.8` | budget training |
| `AUGMENT`, `VFLIP`, `TTA` | `'medium'`, `False`, `True` | augmentasi & test-time augmentation |
| `TRAIN_FOLDS` | `[0]` | fold yang di-fine-tune |
| `SUBMISSION_FORMAT` | `'auto'` | label / probabilitas |

**Output:** `submission.csv`, `oof/`, `holdout_scores.csv`, `test_predictions.csv`, `decisions.csv`, `report.json`, cache fitur di `pipeline-output/cache/`.

**Troubleshooting:** `GatedRepoError` → set `HF_TOKEN` (atau biarkan fallback); OOM → turunkan `BATCH_SIZE`/`IMG_SIZE`/`UNFREEZE_LAST_N`, `GRAD_CHECKPOINT = True`; kolom gambar tidak ketemu → isi `IMAGE_COL`.
