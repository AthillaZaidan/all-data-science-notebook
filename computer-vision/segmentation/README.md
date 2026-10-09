# Segmentasi (DINOv3 ViT-L)

| File | Fungsi |
|---|---|
| `pipeline_cv_segmentation.ipynb` | linear head → decoder multi-layer → blend → post-processing → holdout → mask + RLE submission |
| `build_pipeline_cv_segmentation.py` | builder |

**Data:** file mask (`masks/`, `_mask`) atau tabel RLE (`ImageId`, [`ClassId`], `EncodedPixels`). Binary atau multiclass.

## Alur section

| # | Section | Isi kode |
|---|---|---|
| 2 | Initialization | `is_mask`, `strip_suffix` (pasangkan gambar-mask), `detect_rle_col`, `ImageIndex` |
| 3 | Toolkit | `rle_decode` / `rle_encode`, `remove_small`, `to_mask`, `SegScore` (Dice, IoU, mIoU) |
| 4 | Data Overview and Decisions | `read_raw_mask`, `map_mask` (nilai/warna → kelas), deteksi `RLE_ORDER`, statistik mask, overlay, duplikat, setting `'auto'` |
| 5 | Data Preparation | `strat_label`, holdout + fold, `SegDataset` + augmentasi bersama gambar & mask |
| 6 | Backbone | `load_backbone`, `patch_maps` (fitur patch multi-layer), PCA |
| 7 | Baseline | `SegModel('linear')`, `seg_loss` (CE/BCE + Dice), `predict_probs`, `train_fold`, `run_stage` |
| 8 | Decoder | `SegModel('conv')`: 4 layer + cabang CNN detail, blok terakhir di-fine-tune |
| 9 | Ensemble & Post-Processing | blend, tuning threshold + ukuran objek minimum |
| 10–12 | Holdout, Interpretation, Error | skor resolusi asli (`full_scores`), IoU per kelas, overlay error, gambar terburuk |
| 13 | Export | `write_submission` (RLE ikut sample), `masks/*.png`, `report.json` |

## Settings penting

| Setting | Default | Arti |
|---|---|---|
| `MASK_DIR`, `MASK_SUFFIX` | `'auto'` | folder & suffix mask |
| `RLE_COL`, `IMAGE_COL`, `CLASS_COL`, `RLE_ORDER` | `'auto'` | tabel RLE; `'F'` kolom / `'C'` baris |
| `MASK_VALUES`, `IGNORE_INDEX`, `CLASS_NAMES` | `'auto'`, `'auto'`, `None` | pemetaan nilai mask → kelas |
| `IMG_SIZE` | `'auto'` | resolusi (224–512) |
| `BASELINE`, `BASELINE_EPOCHS` | `True`, `6` | linear head |
| `FINETUNE`, `UNFREEZE_LAST_N`, `DECODER_LAYERS` | `'auto'` | decoder stage |
| `EPOCHS`, `BATCH_SIZE`, `LR`, `HEAD_LR` | `15`, `8`, `2e-5`, `5e-4` | training |
| `DICE_WEIGHT`, `CLASS_WEIGHTS` | `'auto'` | bobot loss |
| `TUNE_POSTPROCESS`, `SCORE_MAX_SIDE` | `True`, `768` | tuning threshold, resolusi scoring |
| `EMPTY_RLE`, `SAVE_MASKS` | `''`, `True` | format mask kosong, simpan PNG |

**Output:** `submission.csv`, `masks/*.png`, `holdout_scores.csv`, `holdout_per_image.csv`, `training_history.csv`, `decisions.csv`, `report.json`.

**Troubleshooting:** gambar tidak terpasang dengan mask → isi `MASK_DIR` / `MASK_SUFFIX`; mask terlihat bergaris di overlay → `RLE_ORDER` kebalikan; kelas terlalu banyak → isi `MASK_VALUES`; OOM → turunkan `IMG_SIZE`/`BATCH_SIZE`, `GRAD_CHECKPOINT = True`.
