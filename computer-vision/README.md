# Computer Vision (DINOv3 ViT-L)

| Folder | Untuk | Dokumentasi |
|---|---|---|
| `classification/` | klasifikasi / regresi gambar | [README](classification/README.md) |
| `segmentation/` | segmentasi semantik (mask per piksel) | [README](segmentation/README.md) |

Keduanya: tanpa notebook EDA terpisah (section 4 = analisis data + keputusan `'auto'`), backbone `facebook/dinov3-vitl16-pretrain-lvd1689m` (gated, butuh `HF_TOKEN`, fallback `dinov2-large`), fine-tuning hanya otomatis di GPU CUDA.
