"""Improved, re-validated joint space width (JSW) measurement. New file - does not edit or
import logic from scripts/measure_jsw.py (that script, its CLI and data/jsw_features.csv are
left exactly as they were). Explainable image processing only, no learned model.

What changed vs. the original method (see outputs/jsw_v2/jsw_log.md for the full trail):
  1. The joint-line row is found on a SMOOTHED row-intensity profile (a single noisy dark row
     no longer wins by itself), searched only in a narrower central band of the image.
  2. Per-column femur/tibia edges come from the SMOOTHED VERTICAL GRADIENT's strongest peak in
     each direction (the true bone/soft-tissue boundary), not the first row crossing a single
     global intensity threshold.
  3. Edge ROWS are outlier-filtered ACROSS COLUMNS (robust median/MAD) before gaps are formed,
     so a handful of columns with a spurious gradient peak can't corrupt the compartment's
     summary the way a single bad column could before.
  4. A compartment that ends up with no valid columns is reported as a measurement FAILURE
     (None), never silently coerced to a 0.0 gap.
  5. Medial/lateral assignment, the central tibial-spine exclusion (and its matching outer
     margin), and normalization by joint width are kept from the original method - Phase 2
     already diagnosed and fixed those, and this task doesn't ask to revisit them.

All hyperparameters below were chosen on the VALIDATION split only (--mode search), using the
Spearman correlation metric this task also uses for the pass/fail test. The test split is
touched exactly once, by --mode eval --split test.

CLI:
    python scripts/measure_jsw_v2.py --mode search   # validation-only hyperparameter search
    python scripts/measure_jsw_v2.py --mode eval --split val    # full val run, final config
    python scripts/measure_jsw_v2.py --mode eval --split test   # full test run, ONCE
    python scripts/measure_jsw_v2.py --mode orientation-check   # medial/lateral swap check
    python scripts/measure_jsw_v2.py --mode examples            # 2 overlay images per KL grade
"""
import os
import re
import json
import argparse

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kl_common import apply_clahe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs", "jsw_v2")
os.makedirs(OUT_DIR, exist_ok=True)

MANIFEST_PATH = os.path.join(ROOT, "data", "image_manifest.csv")
UNIFIED_PATH = os.path.join(ROOT, "unified_oai_dataset.csv")

# ---------------------------------------------------------------------------
# Settings chosen on the validation split only (outputs/jsw_v2/jsw_log.md has the search
# table). SCAN_FRAC and LOCAL_SEARCH_FRAC are kept at v1's already-diagnosed values (Phase 2
# found these the hard way; this task doesn't ask to revisit them).
# ---------------------------------------------------------------------------
JOINT_SEARCH_LO_FRAC = 0.35
JOINT_SEARCH_HI_FRAC = 0.65
# ROW_SMOOTH_SIGMA/COL_SMOOTH_SIGMA/OUTLIER_MAD_K: chosen by a two-round validation-only grid
# search (outputs/jsw_v2/val_search_results_round1.csv, val_search_results.csv) followed by a
# full-val confirmation of the top ~15 candidates (val_full_confirmation.csv) - the 250-image
# search subsample disagreed with the full val split about which config was best, so the final
# pick is the one with the best FULL-val medial_min_jsw-vs-medial-JSN rho (-0.2758), not the
# subsample's top pick. See jsw_log.md for the full trail.
ROW_SMOOTH_SIGMA = 1.5
COL_SMOOTH_SIGMA = 3.0
OUTLIER_MAD_K = 2.0
CENTRAL_SKIP_FRAC = 0.20
SCAN_FRAC = 0.18
LOCAL_SEARCH_FRAC = 0.10
MIN_VALID_FRAC = 0.15
PASS_THRESHOLD = -0.3


def find_joint_row_v2(gray, search_lo=JOINT_SEARCH_LO_FRAC, search_hi=JOINT_SEARCH_HI_FRAC,
                       row_smooth_sigma=ROW_SMOOTH_SIGMA):
    h, w = gray.shape
    lo, hi = int(h * search_lo), int(h * search_hi)
    row_mean = gray[lo:hi, :].astype(np.float32).mean(axis=1)
    smoothed = gaussian_filter1d(row_mean, sigma=row_smooth_sigma)
    return int(np.argmin(smoothed)) + lo


def joint_extent(gray, joint_row, threshold, band_half=10):
    """Horizontal bone extent (left_x, right_x) in a band around the joint row, via Otsu
    threshold - unchanged logic from v1 (this task doesn't ask to revisit bone-extent finding,
    only the joint-row/edge-detection/outlier steps)."""
    h, w = gray.shape
    r0, r1 = max(0, joint_row - band_half), min(h, joint_row + band_half)
    band = gray[r0:r1, :]
    cols = np.where((band > threshold).any(axis=0))[0]
    if len(cols) == 0:
        return 0, w - 1
    return int(cols[0]), int(cols[-1])


def column_edge_rows(gray, col, joint_row, max_scan, search_radius, col_smooth_sigma=COL_SMOOTH_SIGMA):
    """Finds this column's own local joint line within search_radius of the global joint_row
    (tolerates normal joint-line tilt, same fix Phase 2 made), then returns the row of the
    strongest SMOOTHED-GRADIENT edge above it (femur) and below it (tibia) within max_scan.
    Always returns a row for each direction (possibly a weak/spurious one) - genuine outliers
    are identified afterward by comparing across all columns in the compartment, not by a
    per-column magnitude threshold."""
    h = gray.shape[0]
    r0, r1 = max(0, joint_row - search_radius), min(h, joint_row + search_radius)
    local_row = int(np.argmin(gray[r0:r1, col].astype(np.float32))) + r0

    up0, up1 = max(0, local_row - max_scan), local_row + 1
    up_strip = gray[up0:up1, col].astype(np.float32)
    if len(up_strip) < 4:
        femur_row = local_row
    else:
        grad = gaussian_filter1d(np.gradient(up_strip), sigma=col_smooth_sigma)
        # Walking from the joint line upward into the femur, row index DECREASES while
        # intensity RISES (dark joint space -> bright bone), so the strongest rising edge in
        # this direction is the most negative gradient value (np.gradient is w.r.t. increasing
        # row index).
        femur_row = up0 + int(np.argmin(grad))

    down0, down1 = local_row, min(h, local_row + max_scan + 1)
    down_strip = gray[down0:down1, col].astype(np.float32)
    if len(down_strip) < 4:
        tibia_row = local_row
    else:
        grad = gaussian_filter1d(np.gradient(down_strip), sigma=col_smooth_sigma)
        tibia_row = down0 + int(np.argmax(grad))

    return femur_row, tibia_row


def _robust_outlier_mask(values, k=OUTLIER_MAD_K):
    """True where `values` is within k robust-MADs of the median. All-equal input (MAD=0)
    is treated as no outliers."""
    values = np.asarray(values, dtype=np.float32)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad < 1e-6:
        return np.ones_like(values, dtype=bool)
    return np.abs(values - med) <= k * 1.4826 * mad


def measure_jsw_array_v2(rgb):
    """Core measurement on an already-loaded RGB uint8 array. Returns a dict with per-
    compartment (image-left/image-right) normalized min/mean JSW, valid-column fractions, a
    per-image failure flag per compartment, and the raw per-column arrays needed to draw an
    overlay. Medial/lateral assignment is left to assign_compartments_v2 (caller knows side)."""
    clahe_rgb = apply_clahe(rgb)
    gray = cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    joint_row = find_joint_row_v2(gray, JOINT_SEARCH_LO_FRAC, JOINT_SEARCH_HI_FRAC, ROW_SMOOTH_SIGMA)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    left_x, right_x = joint_extent(gray, joint_row, threshold)
    joint_width = max(right_x - left_x, 1)
    max_scan = max(int(h * SCAN_FRAC), 5)
    search_radius = max(int(h * LOCAL_SEARCH_FRAC), 3)

    skip_half = CENTRAL_SKIP_FRAC * joint_width / 2
    outer_margin = CENTRAL_SKIP_FRAC * joint_width
    center_x = (left_x + right_x) / 2
    left_cols = [c for c in range(left_x, right_x + 1) if left_x + outer_margin < c < center_x - skip_half]
    right_cols = [c for c in range(left_x, right_x + 1) if center_x + skip_half < c < right_x - outer_margin]

    def summarize(cols):
        if not cols:
            return {"min": None, "mean": None, "valid_frac": 0.0, "failed": True,
                    "cols": [], "femur_rows": np.array([]), "tibia_rows": np.array([]),
                    "gaps": np.array([]), "valid_mask": np.array([], dtype=bool)}
        femur_rows, tibia_rows = [], []
        for c in cols:
            fr, tr = column_edge_rows(gray, c, joint_row, max_scan, search_radius, COL_SMOOTH_SIGMA)
            femur_rows.append(fr)
            tibia_rows.append(tr)
        femur_rows, tibia_rows = np.array(femur_rows, dtype=np.float32), np.array(tibia_rows, dtype=np.float32)

        femur_ok = _robust_outlier_mask(femur_rows, OUTLIER_MAD_K)
        tibia_ok = _robust_outlier_mask(tibia_rows, OUTLIER_MAD_K)
        gaps = tibia_rows - femur_rows
        valid_mask = femur_ok & tibia_ok & (gaps > 0)

        valid_frac = float(valid_mask.mean())
        if valid_frac < MIN_VALID_FRAC or valid_mask.sum() == 0:
            return {"min": None, "mean": None, "valid_frac": valid_frac, "failed": True,
                    "cols": cols, "femur_rows": femur_rows, "tibia_rows": tibia_rows,
                    "gaps": gaps, "valid_mask": valid_mask}
        valid_gaps = gaps[valid_mask]
        return {"min": float(valid_gaps.min()), "mean": float(valid_gaps.mean()),
                "valid_frac": valid_frac, "failed": False, "cols": cols,
                "femur_rows": femur_rows, "tibia_rows": tibia_rows, "gaps": gaps,
                "valid_mask": valid_mask}

    left = summarize(left_cols)
    right = summarize(right_cols)

    def norm(v):
        return None if v is None else v / joint_width

    return {
        "joint_row": joint_row, "left_x": left_x, "right_x": right_x, "joint_width_px": joint_width,
        "left_min_jsw": norm(left["min"]), "left_mean_jsw": norm(left["mean"]),
        "right_min_jsw": norm(right["min"]), "right_mean_jsw": norm(right["mean"]),
        "left_valid_frac": left["valid_frac"], "right_valid_frac": right["valid_frac"],
        "left_failed": left["failed"], "right_failed": right["failed"],
        "left_detail": left, "right_detail": right,
        "gray": gray,
    }


def measure_jsw_v2(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return measure_jsw_array_v2(rgb)


def assign_compartments_v2(side_code, measured, swap=False):
    """Same AP-view convention as scripts/measure_jsw.py (right knee: medial = image-right
    half; left knee: medial = image-left half) - Phase 2 already verified this convention by
    the same kind of swap check `--mode orientation-check` repeats here for the new method.
    swap=True deliberately inverts it, for that check only."""
    if side_code == "R":
        medial, lateral = ("left", "right") if swap else ("right", "left")
    elif side_code == "L":
        medial, lateral = ("right", "left") if swap else ("left", "right")
    else:
        return {k: None for k in ("medial_min_jsw", "medial_mean_jsw", "lateral_min_jsw", "lateral_mean_jsw")}
    return {
        "medial_min_jsw": measured[f"{medial}_min_jsw"], "medial_mean_jsw": measured[f"{medial}_mean_jsw"],
        "lateral_min_jsw": measured[f"{lateral}_min_jsw"], "lateral_mean_jsw": measured[f"{lateral}_mean_jsw"],
        "medial_failed": measured[f"{medial}_failed"], "lateral_failed": measured[f"{lateral}_failed"],
    }


# ---------------------------------------------------------------------------
# Overlay drawing (shared by the Part 1 example figures and the Part 2 UI panel)
# ---------------------------------------------------------------------------
FEMUR_COLOR = (255, 90, 90)     # red-ish
TIBIA_COLOR = (90, 160, 255)    # blue-ish
GAP_COLOR = (255, 220, 80)      # yellow-ish, min-gap marker
JOINT_LINE_COLOR = (120, 255, 150)


def draw_overlay(rgb, measurement):
    """Draws the joint line, per-column femur/tibia edge points (valid columns only), and the
    minimum-gap point in each compartment, on a copy of `rgb` (uint8, same size as measured).
    `measurement` is the dict returned by measure_jsw_array_v2."""
    out = rgb.copy()
    h, w = out.shape[:2]
    jr = measurement["joint_row"]
    cv2.line(out, (measurement["left_x"], jr), (measurement["right_x"], jr), JOINT_LINE_COLOR, 1)

    for side in ("left_detail", "right_detail"):
        d = measurement[side]
        cols, femur_rows, tibia_rows, valid_mask = d["cols"], d["femur_rows"], d["tibia_rows"], d["valid_mask"]
        for i, c in enumerate(cols):
            if not valid_mask[i]:
                continue
            cv2.circle(out, (c, int(femur_rows[i])), 1, FEMUR_COLOR, -1)
            cv2.circle(out, (c, int(tibia_rows[i])), 1, TIBIA_COLOR, -1)
        if not d["failed"] and valid_mask.any():
            gaps = d["gaps"]
            valid_idx = np.where(valid_mask)[0]
            min_i = valid_idx[np.argmin(gaps[valid_idx])]
            c = cols[min_i]
            cv2.circle(out, (c, int(femur_rows[min_i])), 4, GAP_COLOR, 2)
            cv2.circle(out, (c, int(tibia_rows[min_i])), 4, GAP_COLOR, 2)
            cv2.line(out, (c, int(femur_rows[min_i])), (c, int(tibia_rows[min_i])), GAP_COLOR, 1)
    return out


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
_CODE_RE = re.compile(r"^\s*(-?\d+)")


def _parse_code(value):
    if pd.isna(value):
        return np.nan
    m = _CODE_RE.match(str(value))
    return float(m.group(1)) if m else np.nan


def load_jsn_labels():
    """ID-keyed table of medial/lateral JSN numeric codes for both knee sides, parsed from
    unified_oai_dataset.csv's categorical strings (e.g. '1: 1 (narrowed, OARSI grade 1-2)')."""
    df = pd.read_csv(UNIFIED_PATH, usecols=["ID", "P01SVRKJSM", "P01SVLKJSM", "P01SVRKJSL", "P01SVLKJSL"],
                      dtype={"ID": str})
    # unified_oai_dataset.csv has duplicate rows per patient ID (repeat-visit rows); these JSN
    # columns are static per patient, so the duplicates carry identical values - verified
    # before dropping them (see jsw_log.md) rather than assumed.
    df = df.drop_duplicates(subset="ID", keep="first")
    for col in ["P01SVRKJSM", "P01SVLKJSM", "P01SVRKJSL", "P01SVLKJSL"]:
        df[col] = df[col].map(_parse_code)
    return df.set_index("ID")


def jsn_for_row(jsn_table, patient_id, side_code):
    if patient_id not in jsn_table.index:
        return np.nan, np.nan
    row = jsn_table.loc[patient_id]
    if side_code == "R":
        return row["P01SVRKJSM"], row["P01SVRKJSL"]
    elif side_code == "L":
        return row["P01SVLKJSM"], row["P01SVLKJSL"]
    return np.nan, np.nan


def load_manifest_split(split):
    manifest = pd.read_csv(MANIFEST_PATH, dtype={"ID": str})
    sub = manifest[manifest["SPLIT"] == split].copy()
    sub = sub[sub["IMAGE_PATH"].map(lambda p: os.path.isfile(os.path.join(ROOT, p)))]
    return sub.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Measurement over a dataframe of manifest rows
# ---------------------------------------------------------------------------
def run_on_rows(rows, swap=False, **overrides):
    """Applies the module-level settings (optionally overridden, for the search mode) to every
    row and returns a results dataframe with medial/lateral JSW + failure flags."""
    global JOINT_SEARCH_LO_FRAC, JOINT_SEARCH_HI_FRAC, ROW_SMOOTH_SIGMA, COL_SMOOTH_SIGMA, \
        CENTRAL_SKIP_FRAC, SCAN_FRAC, LOCAL_SEARCH_FRAC, OUTLIER_MAD_K, MIN_VALID_FRAC
    saved = dict(JOINT_SEARCH_LO_FRAC=JOINT_SEARCH_LO_FRAC, JOINT_SEARCH_HI_FRAC=JOINT_SEARCH_HI_FRAC,
                 ROW_SMOOTH_SIGMA=ROW_SMOOTH_SIGMA, COL_SMOOTH_SIGMA=COL_SMOOTH_SIGMA,
                 CENTRAL_SKIP_FRAC=CENTRAL_SKIP_FRAC, SCAN_FRAC=SCAN_FRAC,
                 LOCAL_SEARCH_FRAC=LOCAL_SEARCH_FRAC, OUTLIER_MAD_K=OUTLIER_MAD_K,
                 MIN_VALID_FRAC=MIN_VALID_FRAC)
    for k, v in overrides.items():
        globals()[k] = v

    jsn_table = load_jsn_labels()
    out_rows = []
    for _, r in rows.iterrows():
        try:
            m = measure_jsw_v2(os.path.join(ROOT, r["IMAGE_PATH"]))
        except Exception:
            continue
        comp = assign_compartments_v2(r["SIDE_CODE"], m, swap=swap)
        jsn_m, jsn_l = jsn_for_row(jsn_table, r["ID"], r["SIDE_CODE"])
        out_rows.append({
            "ID": r["ID"], "SIDE_CODE": r["SIDE_CODE"], "KL_GRADE": r["KL_GRADE"],
            "medial_min_jsw": comp["medial_min_jsw"], "medial_mean_jsw": comp["medial_mean_jsw"],
            "lateral_min_jsw": comp["lateral_min_jsw"], "lateral_mean_jsw": comp["lateral_mean_jsw"],
            "medial_failed": comp["medial_failed"], "lateral_failed": comp["lateral_failed"],
            "jsn_medial": jsn_m, "jsn_lateral": jsn_l,
        })

    for k, v in saved.items():
        globals()[k] = v
    return pd.DataFrame(out_rows)


def compute_metrics(df):
    def rho(a, b):
        mask = a.notna() & b.notna()
        if mask.sum() < 10:
            return None, int(mask.sum())
        a_vals = a[mask].to_numpy(dtype=np.float64)
        b_vals = b[mask].to_numpy(dtype=np.float64)
        r, p = spearmanr(a_vals, b_vals)
        return float(r), int(mask.sum())

    medial_vs_jsn, n1 = rho(df["medial_min_jsw"], df["jsn_medial"])
    medial_vs_kl, n2 = rho(df["medial_min_jsw"], df["KL_GRADE"])
    lateral_vs_jsn, n3 = rho(df["lateral_min_jsw"], df["jsn_lateral"])
    lateral_vs_kl, n4 = rho(df["lateral_min_jsw"], df["KL_GRADE"])
    return {
        "n_images": len(df),
        "medial_min_jsw_vs_jsn_medial": {"rho": medial_vs_jsn, "n": n1},
        "medial_min_jsw_vs_kl_grade": {"rho": medial_vs_kl, "n": n2},
        "lateral_min_jsw_vs_jsn_lateral": {"rho": lateral_vs_jsn, "n": n3},
        "lateral_min_jsw_vs_kl_grade": {"rho": lateral_vs_kl, "n": n4},
        "medial_fail_rate": float(df["medial_failed"].mean()),
        "lateral_fail_rate": float(df["lateral_failed"].mean()),
        "pass": bool(medial_vs_jsn is not None and medial_vs_jsn <= PASS_THRESHOLD),
    }


# ---------------------------------------------------------------------------
# CLI modes
# ---------------------------------------------------------------------------
def mode_search(sample_n=250, seed=42):
    val_rows = load_manifest_split("val")
    rng = np.random.RandomState(seed)
    sample_idx = rng.choice(len(val_rows), size=min(sample_n, len(val_rows)), replace=False)
    sample = val_rows.iloc[sample_idx].reset_index(drop=True)
    print(f"Validation-only grid search on a {len(sample)}-image random subsample of val (seed {seed}).")

    # CENTRAL_SKIP_FRAC (tibial-spine exclusion) is kept fixed at v1's already-validated 0.20
    # (Phase 2 diagnosed this one the hard way; this task's new elements are the smoothing and
    # the cross-column outlier removal, so the search focuses there instead of re-litigating a
    # setting that wasn't broken).
    #
    # Round 1 (sigma up to 2.5) showed a clear, monotonic trend: more column-gradient
    # smoothing kept improving the correlation, with the best result landing right at the top
    # of the tested range (COL_SMOOTH_SIGMA=2.5). Round 2 (this grid) extends the range to see
    # where that trend actually levels off, instead of just taking the edge of an arbitrary
    # first guess - still validation-only, still before the test split is touched.
    grid = []
    for row_sigma in (1.5, 2.0, 2.5, 3.0):
        for col_sigma in (2.5, 3.0, 3.5, 4.0):
            for mad_k in (2.0, 2.5, 3.0):
                grid.append(dict(ROW_SMOOTH_SIGMA=row_sigma, COL_SMOOTH_SIGMA=col_sigma,
                                  OUTLIER_MAD_K=mad_k, CENTRAL_SKIP_FRAC=CENTRAL_SKIP_FRAC))

    results = []
    for i, cfg in enumerate(grid):
        df = run_on_rows(sample, **cfg)
        metrics = compute_metrics(df)
        row = {**cfg, **{k: v for k, v in metrics.items() if not isinstance(v, dict)}}
        row["medial_vs_jsn_rho"] = metrics["medial_min_jsw_vs_jsn_medial"]["rho"]
        results.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(grid)} configs done")

    results_df = pd.DataFrame(results).sort_values("medial_vs_jsn_rho")
    results_df.to_csv(os.path.join(OUT_DIR, "val_search_results.csv"), index=False)
    best = results_df.iloc[0].to_dict()
    chosen = {k: best[k] for k in ("ROW_SMOOTH_SIGMA", "COL_SMOOTH_SIGMA", "OUTLIER_MAD_K", "CENTRAL_SKIP_FRAC")}
    chosen["medial_vs_jsn_rho_on_search_subsample"] = best["medial_vs_jsn_rho"]
    with open(os.path.join(OUT_DIR, "chosen_config.json"), "w") as f:
        json.dump(chosen, f, indent=2)
    print("Best config (by medial_min_jsw vs medial JSN rho, most negative):")
    print(json.dumps(chosen, indent=2))
    return chosen


def mode_confirm():
    """Round 1 and round 2 of --mode search ranked configs by rho on a 250-image val
    SUBSAMPLE (for speed) and disagreed on which config generalizes best once checked on the
    full val split - a sign the subsample is too noisy to trust as the final word. This mode
    re-evaluates the top candidates from both rounds directly on the FULL validation split
    (still validation-only, still before the test split is touched) and picks the real winner
    by that more reliable number."""
    candidates = [
        dict(ROW_SMOOTH_SIGMA=1.5, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=3.0, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=1.5, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=2.0, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=2.0, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=3.0, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=2.5, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=2.0, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=2.0, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=2.0, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=2.5, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=3.0, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=1.5, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=2.5, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=2.0, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=2.5, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=3.0, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=2.0, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=2.0, COL_SMOOTH_SIGMA=2.5, OUTLIER_MAD_K=3.0, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=1.0, COL_SMOOTH_SIGMA=2.5, OUTLIER_MAD_K=2.5, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=2.0, COL_SMOOTH_SIGMA=2.5, OUTLIER_MAD_K=2.5, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=3.0, COL_SMOOTH_SIGMA=2.5, OUTLIER_MAD_K=2.5, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=3.0, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=2.5, CENTRAL_SKIP_FRAC=0.2),
        dict(ROW_SMOOTH_SIGMA=1.5, COL_SMOOTH_SIGMA=2.5, OUTLIER_MAD_K=3.0, CENTRAL_SKIP_FRAC=0.2),
    ]
    val_rows = load_manifest_split("val")
    print(f"Confirming {len(candidates)} top subsample-search candidates on the FULL val split "
          f"({len(val_rows)} images).")
    results = []
    for i, cfg in enumerate(candidates):
        df = run_on_rows(val_rows, **cfg)
        m = compute_metrics(df)
        row = {**cfg, "full_val_medial_rho": m["medial_min_jsw_vs_jsn_medial"]["rho"],
               "full_val_pass": m["pass"], "medial_fail_rate": m["medial_fail_rate"]}
        results.append(row)
        print(f"  {i + 1}/{len(candidates)}: {cfg} -> full_val_rho={row['full_val_medial_rho']:.4f} "
              f"pass={row['full_val_pass']}")

    results_df = pd.DataFrame(results).sort_values("full_val_medial_rho")
    results_df.to_csv(os.path.join(OUT_DIR, "val_full_confirmation.csv"), index=False)
    best = results_df.iloc[0].to_dict()
    chosen = {k: best[k] for k in ("ROW_SMOOTH_SIGMA", "COL_SMOOTH_SIGMA", "OUTLIER_MAD_K", "CENTRAL_SKIP_FRAC")}
    chosen["full_val_medial_rho"] = best["full_val_medial_rho"]
    chosen["full_val_pass"] = bool(best["full_val_pass"])
    with open(os.path.join(OUT_DIR, "chosen_config.json"), "w") as f:
        json.dump(chosen, f, indent=2)
    print("\nFINAL chosen config (best on FULL val, not the subsample proxy):")
    print(json.dumps(chosen, indent=2))
    return chosen


def _apply_chosen_config():
    path = os.path.join(OUT_DIR, "chosen_config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        chosen = json.load(f)
    return {k: v for k, v in chosen.items() if k in
            ("ROW_SMOOTH_SIGMA", "COL_SMOOTH_SIGMA", "OUTLIER_MAD_K", "CENTRAL_SKIP_FRAC")}


def mode_eval(split):
    cfg = _apply_chosen_config()
    rows = load_manifest_split(split)
    print(f"Running the chosen config on the FULL {split} split ({len(rows)} images)"
          + (" - this split is touched exactly once." if split == "test" else "."))
    df = run_on_rows(rows, **cfg)
    df.to_csv(os.path.join(OUT_DIR, f"{split}_results.csv"), index=False)
    metrics = compute_metrics(df)
    with open(os.path.join(OUT_DIR, f"{split}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def mode_orientation_check():
    cfg = _apply_chosen_config()
    rows = load_manifest_split("val")
    df_normal = run_on_rows(rows, swap=False, **cfg)
    df_swapped = run_on_rows(rows, swap=True, **cfg)
    m_normal = compute_metrics(df_normal)["medial_min_jsw_vs_jsn_medial"]
    m_swapped = compute_metrics(df_swapped)["medial_min_jsw_vs_jsn_medial"]
    result = {"as_labeled_rho": m_normal["rho"], "swapped_rho": m_swapped["rho"], "n": m_normal["n"]}
    with open(os.path.join(OUT_DIR, "orientation_check.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    return result


def mode_examples():
    cfg = _apply_chosen_config()
    for k, v in cfg.items():
        globals()[k] = v
    test_rows = load_manifest_split("test")
    ex_dir = os.path.join(OUT_DIR, "examples")
    os.makedirs(ex_dir, exist_ok=True)
    jsn_table = load_jsn_labels()

    for grade in range(5):
        grade_rows = test_rows[test_rows["KL_GRADE"] == grade].reset_index(drop=True)
        saved = 0
        # First pass: try to grab one normal + one failure example per grade.
        got_failure = False
        for _, r in grade_rows.iterrows():
            if saved >= 2:
                break
            img_path = os.path.join(ROOT, r["IMAGE_PATH"])
            rgb = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            m = measure_jsw_array_v2(rgb)
            is_failure = m["left_failed"] or m["right_failed"]
            if is_failure and got_failure:
                continue
            if not is_failure and saved == 1 and not got_failure:
                # prefer a failure as the second example if one hasn't been found yet; keep
                # scanning a bit before giving up and accepting a second normal example.
                pass
            overlay = draw_overlay(apply_clahe(rgb), m)
            comp = assign_compartments_v2(r["SIDE_CODE"], m)
            out_path = os.path.join(ex_dir, f"KL{grade}_{r['IMAGE_NAME']}")
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            saved += 1
            if is_failure:
                got_failure = True
            print(f"KL{grade}: saved {out_path} (medial_min={comp['medial_min_jsw']}, "
                  f"lateral_min={comp['lateral_min_jsw']}, failure={is_failure})")
    print(f"Examples saved to {ex_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                     choices=["search", "confirm", "eval", "orientation-check", "examples"])
    ap.add_argument("--split", default="test", choices=["val", "test"])
    args = ap.parse_args()

    if args.mode == "search":
        mode_search()
    elif args.mode == "confirm":
        mode_confirm()
    elif args.mode == "eval":
        mode_eval(args.split)
    elif args.mode == "orientation-check":
        mode_orientation_check()
    elif args.mode == "examples":
        mode_examples()


if __name__ == "__main__":
    main()
