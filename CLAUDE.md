# Knee OA KL-Grade Prediction — Project Context

BE major project: predict Kellgren-Lawrence (KL) grade (0-4) for knee osteoarthritis
severity from X-rays, with explainability (XAI) and causal inference layered on top.
Causal inference is PARKED for later. Current focus: the KL-grade classifier.

## Hardware
- Local Windows laptop, RTX 3050 (6GB VRAM). Keep training memory-optimized:
  mixed precision (torch.amp), modest batch sizes, save the best checkpoint.
- Classifier: batch size 32 at 224x224 (drop to 16 on CUDA out-of-memory).
- On Windows, if DataLoader errors mention workers/pickling/paging file, use
  `--workers 0`.

## Folder layout (run everything from the project root)
- `dataset/{train,val,test}/{0..4}/*.png` — OAI images (Kaggle split). The scripts
  read these folders directly.
- `data/` — `image_manifest.csv`, `clinical_features.csv` (see below).
- `unified_oai_dataset.csv` — raw merged OAI clinical + image table.
- `segmentation_data/` — CGMH knee segmentation dataset (400 image/mask pairs).
- `notebooks/` — Colab segmentation notebook, kept as a record only. Do NOT rerun
  or convert it; segmentation is finished.
- `outputs/checkpoints/` — `unet_best.pt`, `unetpp_best.pt` (trained in Colab,
  copied from Google Drive) plus locally trained `kl_*_best.pt` classifiers.
- `outputs/results/` — training logs, test reports, confusion matrices,
  `kl_comparison.csv`. `outputs/predictions/` — single-image prediction figures.
- `scripts/prepare_dataset.py` — builds the two CSVs in `data/`.
- `scripts/kl_common.py` — shared dataset, transforms, models, eval helper.
- `scripts/train_classifier.py`, `scripts/evaluate_classifier.py`, `scripts/predict.py`.

## Datasets — decisions already made, don't redo this analysis
Source: OAI (Osteoarthritis Initiative), originally a Kaggle-derived export.

- **Images**: `dataset/train`, `dataset/val`, `dataset/test` (grade 0-4 subfolders).
  **`auto_test` is dropped** — verified near-duplicate of `test` (811 of 828 test
  patients also appear there under different filenames), not new data.
- **Clinical data**: originally 3 files (`ALLClinical.csv`, `final_project_master.csv`,
  `unified_oai_dataset.csv`). **Only `unified_oai_dataset.csv` is used** — strict
  superset of the other two.
- **Verified**: train/val/test have ZERO patient overlap (checked by patient ID).
- **Class imbalance**: real and consistent across splits (stratified). Train
  distribution: 0=39.6%, 1=18.1%, 2=26.2%, 3=13.1%, 4=3.0%. Handled with
  **class-weighted cross-entropy** (inverse frequency). No weighted sampler unless
  KL3/KL4 recall stays weak after weighted loss alone. No SMOTE (tabular only).

### Generated files in `data/` (via `scripts/prepare_dataset.py`, 2026-09-22)
- `image_manifest.csv` — ID, SIDE_CODE, SPLIT, IMAGE_NAME, IMAGE_PATH, KL_GRADE.
  8,260 rows (train 5,778 / val 826 / test 1,656). One file
  (`dataset/test/4/9215922R.png`) is missing from disk — pre-existing gap in the
  source data; harmless (1 of 8,260). The classifier scripts read folders directly,
  so the test set they use has 1,655 images.
- `clinical_features.csv` — filtered clinical columns for the causal phase (parked).
  Currently holds age (`V00AGE`), BMI (`P01BMI`), WOMAC pain/function R/L,
  family history (`P01FAMHR`/`P01FAMKR`), lateral JSN R/L, osteophyte R/L.
  **Correction:** the original selection was 14 clinical columns, including BOTH
  lateral and medial JSN (`P01SVRKJSL`, `P01SVRKJSM`, `P01SVLKJSL`, `P01SVLKJSM`);
  the "12 columns" figure in the old notes was wrong. Add the two medial JSN columns
  back when the causal phase resumes (medial JSN is clinically the more important one).

## Segmentation — DONE (trained in Colab)
- Ground truth: **CGMH Knee Segmentation Database**
  (github.com/yaufan/Knee_Segmentation_Database) — 400 AP knee X-rays with
  femur+tibia masks (BoneFinder), MIT licensed. Different cohort from OAI.
- Setup (all models): 320 train / 80 val, seed 42, 256x256, flip/rotate ±10°/
  brightness-contrast augmentation, T4 GPU, mixed precision. U-Net, U-Net++ and
  DeepLabV3+ used a ResNet34 ImageNet encoder, Dice loss, Adam 1e-4, batch 8.
  Mask R-CNN: ResNet50-FPN COCO-pretrained, SGD 0.005.
- Pixel-level results on the 80 val images:
  - U-Net (40 epochs):      P 0.9727, R 0.9728, F1 0.9727, IoU 0.9469
  - U-Net++ (40 epochs):    P 0.9756, R 0.9720, F1 0.9738, IoU 0.9490  <- selected
  - DeepLabV3+ (20 epochs): P 0.9743, R 0.9500, F1 0.9620, IoU 0.9268
  - Mask R-CNN (20 epochs): P 0.9571, R 0.9644, F1 0.9607, IoU 0.9244
- **Carrying forward U-Net and U-Net++ only.** U-Net++ leads U-Net by only ~0.1% F1.
- Known issue: on OAI images the masks are patchy. CGMH shows the full leg with the
  knee small in the middle; OAI images are tight joint close-ups (scale mismatch).
  A retrain on OAI-style crops of CGMH was planned but NOT done.
- Because OAI images are already cropped to the joint, mask-based cropping is not
  needed before classification. Masks are used as an optional 4th input channel
  (see below) and later for Grad-CAM validation.

## Classifier — three runs DONE (2026-09-25), reviewing results
- EfficientNet-B0 (ImageNet pretrained), 224x224, class-weighted cross-entropy,
  AdamW lr 3e-4 + cosine schedule, early stopping on val QWK (patience 6,
  max 25 epochs). Augmentation: horizontal flip, rotation ±10°, brightness/contrast.
  Trained locally with `--workers 0` (no DataLoader issues hit).
- Three runs compared (same setup otherwise):
  - `baseline` — image only (3 channels)
  - `unet` — image + frozen U-Net bone-probability map as a 4th channel
  - `unetpp` — image + frozen U-Net++ bone-probability map as a 4th channel
- **Headline metric: Quadratic Weighted Kappa (QWK).** Also accuracy, macro F1,
  per-grade precision/recall/F1 (esp. KL3/KL4 recall), 5x5 confusion matrix.
  Never rely on raw accuracy alone (always-predict-KL0 gets ~40%).
- Test split used only for final evaluation, never for model selection.
- Do not change model, loss, split or hyperparameters between the three runs
  without asking — they must stay comparable.
- **Important caveat on the unet/unetpp runs:** they use the original CGMH-trained
  U-Net/U-Net++ checkpoints (full-leg images, knee small in the frame — see the
  Segmentation section's known scale-mismatch issue). On OAI's tight joint
  close-ups the resulting bone-probability masks are patchy, so the 4th channel
  is a noisier signal than it would be from masks retrained on OAI-scale crops.
  That retrain was never done — keep this in mind when interpreting why the mask
  channel didn't clearly beat the baseline below.

### Test-set results (1,655 images), 2026-09-25
| Run | Accuracy | QWK | Macro F1 | Recall KL3 | Recall KL4 |
|---|---|---|---|---|---|
| baseline | 0.6284 | **0.8246** | 0.6746 | 0.7713 | 0.90 |
| unet | 0.6544 | 0.8130 | 0.6714 | 0.7309 | 0.86 |
| unetpp | 0.6508 | 0.8144 | 0.6542 | 0.7623 | 0.90 |

**Baseline (image-only) has the best QWK.** Adding either mask channel gave a
small accuracy bump but a lower QWK and (for unetpp) lower macro F1 — consistent
with the patchy-mask caveat above. KL1 ("doubtful OA") is the weak class across
all three runs (precision 0.31-0.36) — this boundary is subjective even between
radiologists, not obviously a model bug. Full per-grade reports:
`outputs/results/kl_{baseline,unet,unetpp}_test_report.csv`. Per-image predictions
(path, true/predicted grade, all 5 class probabilities):
`outputs/results/kl_{baseline,unet,unetpp}_test_predictions.csv`. Confusion
matrices: `outputs/results/kl_{baseline,unet,unetpp}_confusion_matrix.png`.
Comparison table: `outputs/results/kl_comparison.csv`.

## XAI (after the classifier)
- Grad-CAM on the best classifier; check attention overlaps the joint region using
  the U-Net++ masks.

## Causal inference — PARKED, do not build yet
- Use `clinical_features.csv` (age, BMI, WOMAC, family history) plus imaging
  features. OAI already includes radiologist-graded JSN/osteophyte scores, usable
  directly or to validate mask-derived measurements.
- Needs ONE scoped causal question (e.g. "does BMI causally affect KL grade,
  independent of age and family history"). Tools: DoWhy or CausalNex.

## Environment setup
- venv at `venv/` uses **Python 3.12**, not the system default 3.14 — PyTorch CUDA
  wheels aren't built for 3.14. Created with `py -3.12 -m venv venv`.
- Installed: `torch==2.14.0+cu126` (CUDA confirmed on the RTX 3050), numpy, pandas,
  pillow, scikit-learn, matplotlib. **Do not reinstall torch.**
- `requirements.txt` installed (segmentation-models-pytorch, albumentations,
  opencv-python, scipy, plus pandas/scikit-learn/matplotlib already present) —
  confirmed torch stayed the CUDA build afterwards.
- Git installed via `winget install --id Git.Git`.
- Network is flaky: large downloads can stall at 0 B/s. If there's no byte progress
  for a while, kill and retry. `git config http.postBuffer 524288000` plus a
  shallow `--depth 1` clone fixed the dataset clone.

## Commands
python scripts/train_classifier.py --mask-source none|unet|unetpp
python scripts/evaluate_classifier.py --mask-source none|unet|unetpp
python scripts/predict.py path/to/xray.png --run baseline|unet|unetpp [--show]

## Current status
Done: dataset cleanup (`prepare_dataset.py`, CSVs verified); Python 3.12 venv with
PyTorch+CUDA; CGMH dataset cloned; segmentation trained and evaluated in Colab
(4 models); U-Net and U-Net++ selected; classifier scripts written; `requirements.txt`
installed; all three classifier runs (baseline/unet/unetpp) trained and evaluated
on the test set (see results table in the Classifier section) — baseline has the
best QWK (0.8246).

Next: decide which classifier to carry forward (baseline looks like the pick given
the mask-quality caveat) and move on to Grad-CAM XAI on it, validated against the
U-Net++ masks. Retraining segmentation on OAI-scale crops (to fix the patchy-mask
issue) is an open option if the mask channel is revisited later, but not decided yet.
