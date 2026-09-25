# Knee OA — KL-Grade Prediction & Causal Inference

Predicts Kellgren-Lawrence (KL) grade (0-4) for knee osteoarthritis severity from
X-rays, with explainability (XAI) and causal inference layered on top. See
`CLAUDE.md` for the full project history, decisions, and current status.

## What's in this repo vs. what you need to get separately

Committed here: all code, the trained classifier + segmentation checkpoints
(via Git LFS for the two large ones), training/evaluation results, and the
small derived CSVs (`data/image_manifest.csv`, `data/clinical_features.csv`).

**Not committed** (too large and/or restricted-use OAI data — ask the team for
these): `dataset/` (OAI X-ray images), `unified_oai_dataset.csv` (raw OAI
clinical export), `segmentation_data/` (CGMH mask dataset — only needed if
you're retraining segmentation, not for inference).

## Setup

1. Python 3.12 specifically — PyTorch's CUDA wheels aren't built for 3.14 yet.
   ```
   py -3.12 -m venv venv
   venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu126
   venv\Scripts\pip install -r requirements.txt
   ```
2. Verify CUDA is working: `venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"`
3. Get `dataset/` and `unified_oai_dataset.csv` from the team and place them at
   the project root (same layout as this repo). Only needed for training/
   evaluation, not for single-image prediction (step below).

## Predict the KL grade of an X-ray

Checkpoints are already included, so this works right after setup — no dataset needed:

```
venv\Scripts\python scripts\predict.py
```

Opens a file browser to pick an image, then shows the predicted grade with
confidence and a probability chart. Repeats until you cancel. Options:
```
venv\Scripts\python scripts\predict.py path\to\xray.png --show   # skip the file browser
venv\Scripts\python scripts\predict.py --run unet                # or unetpp / baseline (default)
```

## Train / evaluate (needs `dataset/`)

```
venv\Scripts\python scripts\train_classifier.py --mask-source none|unet|unetpp --workers 0
venv\Scripts\python scripts\evaluate_classifier.py --mask-source none|unet|unetpp --workers 0
```
`--workers 0` avoids known Windows DataLoader issues. Results land in
`outputs/results/` (per-run training logs, test reports, confusion matrices,
per-image predictions CSV, and `kl_comparison.csv` across all three runs).
Current results: baseline has the best QWK (0.8246) — see `CLAUDE.md` for the
full comparison and why the mask channel didn't help.

## Causal inference (open — this is the next phase)

Not started yet. `data/clinical_features.csv` has age, BMI, WOMAC pain/function,
family history, and JSN/osteophyte grades already joined to each image's KL
grade and split. See the "Causal inference" section of `CLAUDE.md` for the
parked plan (needs one scoped causal question, e.g. does BMI causally affect
KL grade independent of age/family history; suggested tools: DoWhy or CausalNex).

## Project context

`CLAUDE.md` has the full history: dataset decisions, segmentation results (4
models compared in Colab), classifier setup and results, environment quirks,
and current status. Read it before making changes — it also documents *why*
certain choices were made, not just what was decided.
