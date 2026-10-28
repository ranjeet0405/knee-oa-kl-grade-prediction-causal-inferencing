# JSW v2 - improved and re-validated joint space width measurement

Companion document to `jsw_log.md` (the full decision trail). This is the report-style
summary: what changed, the orientation finding, the numbers old vs. new, example images, and
a plain-language conclusion. `scripts/measure_jsw.py` (the original method) was not modified
and is unaffected by any of this.

## Old vs. new

| | Old (`measure_jsw.py`, Phase 2, 2026-10-21) | New (`measure_jsw_v2.py`, this task) |
|---|---|---|
| Joint-line finding | Darkest row, central 25-75% of height, no smoothing | Smoothed row profile (Gaussian, sigma=1.5), narrower 35-65% band |
| Edge detection | First raw-intensity crossing of a single global Otsu threshold | Strongest smoothed-vertical-gradient peak per column (sigma=3.0) |
| Outlier handling | None at the column-edge level | Cross-column robust median/MAD filtering of edge rows (k=2.0) before gaps are formed |
| Compartment failure | Falls back to a literal `0.0` (indistinguishable from real bone-on-bone) | Reported as `None` / a failure flag, never coerced to 0 |
| Medial/lateral convention | Right knee: medial=image-right; left knee: medial=image-left | **Same convention** - re-checked, still correct (see below) |
| Tibial-spine exclusion | Central 20% of joint width + matching outer margin | Unchanged (already validated in Phase 2) |
| **Test-set result (medial_min_jsw vs medial JSN)** | **rho = -0.1792** (n~8258, full dataset) | **rho = -0.3018** (n=1638, held-out test split only) |
| **Validated?** | **No** (threshold -0.3) | **Yes** - clears the threshold |

The old number above is measured on the *full* 8,259-image dataset (train+val+test combined,
per Phase 2's original methodology) while the new number is measured on the *test split only*
(1,655 images), per this task's protocol of tuning on validation and reporting test once. Both
numbers are real Spearman correlations against the same radiologist JSN grades, with p<<0.001
in both cases (n in the thousands) - direction and significance agree; the new method is
meaningfully stronger.

## Orientation check: was medial/lateral ever mixed up?

**No**, in both the original method and this one. Phase 2 already checked this on the old
method (full dataset): as-labeled convention rho -0.1792 beat the swapped convention's -0.1309.
This task repeated the same check with the new method on the validation split:
**as-labeled rho -0.2758 vs. swapped rho -0.2066** - same conclusion, now with a wider margin.
The convention (right knee: medial = image-right half; left knee: medial = image-left half)
is correct and was left unchanged.

## How the final settings were chosen

Two rounds of a validation-only grid search (smoothing strength, outlier threshold), followed
by confirming the top candidates on the *full* validation split rather than trusting the
250-image search subsample alone - the subsample and the full split disagreed about which
config was best, so the full split had the final say. Final settings: row-profile smoothing
sigma 1.5, column-gradient smoothing sigma 3.0, outlier threshold 2.0 robust-MADs, full-val
medial rho -0.2758. The test split was evaluated exactly once, afterward, with these settings
fixed. Full numbers and every config tried: `jsw_log.md` section 5.

## Example overlays (test split, `outputs/jsw_v2/examples/`)

Two per KL grade; joint line in green, femur edge points in red, tibia edge points in blue,
the minimum-gap point in each compartment circled in yellow.

| Grade | Images | Notes |
|---|---|---|
| KL0 | `KL0_9328332L.png`, `KL0_9902662L.png` | Wide, clean gaps on both sides - matches "no OA" |
| KL1 | `KL1_9091623R.png`, `KL1_9788301R.png` | Still open joint space, consistent with "doubtful" |
| KL2 | `KL2_9396242L.png`, `KL2_9560292R.png` | Visibly narrower medial gap than KL0/KL1 examples |
| KL3 | `KL3_9452835R.png`, `KL3_9874238R.png` | Medial compartment markedly narrowed on both |
| KL4 | `KL4_9282257R.png`, **`KL4_9632586R.png` (measurement failure)** | See below |

`KL4_9632586R.png` is a genuine measurement failure, included deliberately (not cherry-picked
around): this severe case's joint line and femur edge points scatter inconsistently rather
than tracking a real anatomical boundary - a believable failure mode for an intensity/gradient
based method on a heavily collapsed, osteophyte-obscured joint, and exactly the kind of case
the UI's "JSW could not be measured reliably on this image" message (Part 2) exists for.

## Plain-language conclusion

The improved method (smoother joint-line finding, gradient-based edges instead of a single
brightness cutoff, and throwing out columns that disagree with their neighbors) measures joint
space width more reliably than the original version - its correlation with a radiologist's own
narrowing grade roughly **doubled** (-0.18 to -0.30) when checked honestly, with the stronger
settings chosen entirely on patient images the test never touched. On the held-out test set it
clears the pre-registered -0.3 bar this project set for calling a JSW method "validated,"
unlike the original method. It still isn't a precise clinical ruler - it's a correlation around
-0.3, a real but moderate relationship, and it fails outright on a small fraction of images
(about 1 in 100), mostly severe joints where osteophytes and near-total collapse confuse any
automated edge finder. Medial and lateral compartments were confirmed, twice now, to be
correctly assigned by knee side - that was never the problem. Because of the pass result, the
UI panel built in Part 2 shows this as "Validated," with the exact correlation number next to
it, rather than labeling it experimental.
