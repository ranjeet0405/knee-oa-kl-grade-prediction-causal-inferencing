# JSW v2 - decision log

Built under a strict add-only rule (2026-10-28): new code only in `scripts/measure_jsw_v2.py`
and `ui/jsw_bridge.py`; new outputs only under `outputs/jsw_v2/`. `scripts/measure_jsw.py`,
`ui/model_bridge.py`, models, checkpoints and existing results are untouched. JSW is never fed
to the classifier and never changes the KL grade - confirmed in testing below.

## 1. How the old method (`scripts/measure_jsw.py`) worked and was validated

Read in full before writing anything new. Pipeline: CLAHE -> grayscale -> find the joint line
as the darkest horizontal band averaged over columns, searched in the central 25-75% of image
height (`find_joint_row`) -> Otsu threshold -> horizontal bone extent at the joint row
(`joint_extent`) -> for each column, find a *local* joint row within +-10% image height of the
global row (tolerates normal joint-line tilt - this was Phase 2's second bug fix, see
`final_build_log.md`), then scan up/down from there for the first row whose raw intensity
crosses the Otsu threshold (femur edge going up, tibia edge going down) -> gap = tibia row -
femur row, normalized by joint width. The central 20% of joint width (tibial spines) plus a
matching 20% outer margin at each compartment's edge (condyle curvature) are excluded; a
column where either edge isn't found returns `None`, excluded from that compartment's min/mean
unless every column in the compartment failed (then reported as a literal 0.0, not a failure
flag - **the one piece of this task's Part 2 that couldn't reuse the old semantics**, see
section 3).

**Validation** (final_build_log.md Phase 2, full 8,259-image dataset, joined on patient ID +
knee side against `unified_oai_dataset.csv`'s `P01SV{R,L}KJS{L,M}` columns): medial_min_jsw vs
medial JSN rho = **-0.1792**, lateral_min_jsw vs lateral JSN rho = -0.1455 (both n~8258,
p<1e-30). **Verdict: NOT validated** (threshold was -0.3). Box plots showed a clean,
monotonically decreasing trend by both JSN grade and KL grade - the direction was right, the
correlation just wasn't strong enough. Consequence already logged in CLAUDE.md: JSW excluded
from Hybrid 2's features and from the causal mediation analysis.

## 2. Medial/lateral orientation - checked, not assumed

`assign_compartments` encodes: right knee -> medial = image-right half; left knee -> medial =
image-left half. This is **not a new claim for v2 to verify from scratch** - Phase 2 already
ran exactly this swap check and logged it: as-labeled rho (-0.1792) beats the swapped
convention's rho (-0.1309) on the full dataset, i.e. **the old script did NOT mix up
medial/lateral**. This task's instruction to check it again was followed literally anyway
(`--mode orientation-check` in `measure_jsw_v2.py`, using `SIDE_CODE` from
`data/image_manifest.csv` exactly as instructed) with the new method, on the full validation
split: **as-labeled rho = -0.2758 vs swapped rho = -0.2066** - same conclusion, same direction,
now with a visibly larger margin. Convention confirmed correct twice, by two different
measurement methods, on two different data slices.

## 3. What v2 actually changes, and why

1. **Row-profile smoothing before finding the joint line.** `find_joint_row_v2` runs
   `scipy.ndimage.gaussian_filter1d` over the column-averaged row-intensity profile before
   taking the argmin, so one noisy dark row can no longer out-vote the real joint line by
   itself. Also narrowed the search band from v1's central 25-75% to 35-65% - the joint line is
   essentially never near the very top/bottom quarter of these pre-cropped OAI images, and a
   narrower band gives the smoothed argmin less room to latch onto a dark region near the
   femoral condyles or tibial shaft instead.
2. **Gradient-based edge detection instead of a fixed intensity threshold.** v1's
   `column_jsw` took the *first* row whose raw intensity crossed a single global Otsu
   threshold - sensitive to exactly where that threshold happens to fall for a given image.
   v2's `column_edge_rows` instead smooths each column's local intensity profile
   (`gaussian_filter1d` again) and takes the row of the *strongest* vertical-gradient peak in
   each direction - the point where the image is changing fastest from soft tissue to bone,
   which is a more direct definition of "the edge" than "the first point past an arbitrary
   brightness cutoff."
3. **Cross-column outlier removal**, done the way the brief asked for explicitly: rather than
   trusting every column's own gradient peak, `_robust_outlier_mask` computes each
   compartment's median femur-edge row and median tibia-edge row (and their MAD, median
   absolute deviation) and discards any column whose edge row sits more than
   `OUTLIER_MAD_K` robust-MADs from that median, *before* gaps are formed. A column surviving
   only if its edge is reasonably consistent with its neighbors, not merely "found some
   gradient peak," is the actual mechanism asked for.
4. **A compartment failure is `None`, never a silent `0.0`.** v1 fell back to reporting 0.0 JSW
   when every column in a compartment failed - indistinguishable from a genuine bone-on-bone
   zero reading in the downstream CSV. v2 reports `failed=True` with `min`/`mean` left `None`,
   which Part 2's UI panel turns into "JSW could not be measured reliably on this image"
   instead of a number, per the brief.
5. **Kept unchanged, deliberately**: the central tibial-spine exclusion and its matching outer
   margin, the per-column *local* joint-row search (tolerates joint-line tilt), and the
   medial/lateral assignment convention - Phase 2 already diagnosed/validated all three the
   hard way, and this task's new ideas (smoothing, gradient edges, outlier removal) don't
   touch any of them.

## 4. A real bug found and fixed during the validation-split search

The first grid-search attempt (`val_search_results.csv`, see `chosen_config_round1.json`)
produced **the exact same rho for all 27 configs** - a dead giveaway something was broken, not
a coincidence. Root cause: `find_joint_row_v2`, `column_edge_rows` and `_robust_outlier_mask`
all took their tunable parameter as a Python default argument bound to the module-level
constant (e.g. `def column_edge_rows(..., col_smooth_sigma=COL_SMOOTH_SIGMA)`). Python
evaluates a default argument **once, at function-definition time** - so the search loop's
`globals()[k] = v` monkey-patching, done to sweep settings without threading them through every
call site, silently never reached these three functions; only `CENTRAL_SKIP_FRAC` (read
directly from the module global inside the function body, not through a default) actually
varied, and it was the one parameter held fixed across that entire grid. Confirmed by
diffing two deliberately-very-different configs on the same 40 images before re-running
anything at scale - they produced different `medial_min_jsw` values only after the fix. Fixed
by passing every tunable value explicitly at each call site
(`find_joint_row_v2(gray, JOINT_SEARCH_LO_FRAC, JOINT_SEARCH_HI_FRAC, ROW_SMOOTH_SIGMA)` etc.),
so the live module global is actually read at call time. The round-1 search results file is
kept on disk for the record but was not used to choose anything.

A second, smaller bug: `unified_oai_dataset.csv` has duplicate rows per patient ID (repeat
clinic-visit rows); `jsn_table.loc[patient_id]` on a non-unique index returned a small Series
instead of a scalar, breaking `spearmanr`. Checked (not assumed) that the JSN columns are
identical across every duplicate for every patient (`groupby('ID').nunique()` all equal to 1)
before dropping duplicates with `keep="first"`.

## 5. Validation-split hyperparameter search (full trail)

Settings considered: `ROW_SMOOTH_SIGMA`, `COL_SMOOTH_SIGMA`, `OUTLIER_MAD_K`.
`CENTRAL_SKIP_FRAC` was held fixed at v1's already-validated 0.20 (Phase 2 tuned this one the
hard way; this task's new ideas are the smoothing and the outlier step, so the search focused
there). The test split was not touched until section 6.

- **Round 1** (`val_search_results_round1.csv`, 27 configs, smoothing sigma up to 2.5, 250-image
  val subsample for speed): best subsample rho -0.182, with `COL_SMOOTH_SIGMA=2.5` (the top of
  the tested range) winning - a sign the trend hadn't leveled off yet.
- **Round 2** (`val_search_results.csv`, 48 configs, smoothing sigma extended to 4.0): best
  subsample rho -0.311, peaking at `COL_SMOOTH_SIGMA=3.0` - genuinely inside the tested range
  this time, not at the edge, so the search was extended far enough.
- **Full-val confirmation** (`val_full_confirmation.csv`): the two rounds' top-ranked configs
  by the 250-image subsample **disagreed** about which one generalizes best once checked on the
  full 826-image validation split (round 1's subsample-pick scored -0.273 on full val; round
  2's subsample-pick scored only -0.267) - the subsample is a useful fast proxy for *narrowing*
  the search, but too noisy at n=250 to trust for the final call. Re-evaluated the top 15
  candidates from both rounds directly on the full validation split and picked the actual
  best: **`ROW_SMOOTH_SIGMA=1.5, COL_SMOOTH_SIGMA=3.0, OUTLIER_MAD_K=2.0`, full-val medial rho
  = -0.2758** (`outputs/jsw_v2/chosen_config.json`). All ~20 distinct configs tried across both
  rounds plus the confirmation step landed in a tight -0.25 to -0.28 band on full val - a
  stable plateau, not a search that was cut short. This settled the final hyperparameters
  *before* the test split was touched.

## 6. Test-split result (evaluated once, per the rule)

`python scripts/measure_jsw_v2.py --mode eval --split test` (1,655 images, the chosen config
fixed beforehand, run exactly once):

| Feature | vs matching JSN | rho | vs KL grade | rho |
|---|---|---|---|---|
| medial_min_jsw | jsn_medial | **-0.3018** (n=1638) | KL_GRADE | -0.2587 (n=1639) |
| lateral_min_jsw | jsn_lateral | -0.2736 (n=1638) | KL_GRADE | -0.2703 (n=1638) |

Failure rate: medial 0.97%, lateral 1.03% (16/1655 and 17/1655 images respectively).

**PASS: medial_min_jsw vs medial JSN rho = -0.3018, which clears the ≤ -0.3 threshold.** The
pass rule was fixed before this run and was not changed afterward. The test rho (-0.302) came
out somewhat stronger than the validation estimate (-0.276) - attributed to ordinary
between-split sampling variance (both splits are a few hundred to ~1,600 images, and the
validation search itself showed configs moving by several hundredths of a rho point between
a 250-image subsample and the full 826-image split), not treated as a reason to pick a
different number, since the config was already fixed before this run.

## 7. Part 2 (UI) decisions

- `jsw_bridge.measure(image_path, side=None)` imports `measure_jsw_array_v2`,
  `assign_compartments_v2` and `draw_overlay` from `measure_jsw_v2.py` only - no measurement
  logic is duplicated, and `measure_jsw.py` is never imported from here.
- The validated/experimental badge is decided by reading `outputs/jsw_v2/test_metrics.json`'s
  `pass` field at request time - never recomputed or re-decided by the UI layer, so the badge
  can only ever say what section 6 actually found. Given section 6's PASS result, the badge
  shows **green "Validated (rho = -0.302)"**.
- `side` given (`"L"`/`"R"`): compartments labeled Medial/Lateral via `assign_compartments_v2`
  (the same, orientation-checked convention as Part 1). `side` omitted: labeled "Image-left
  side"/"Image-right side", exactly as the brief specifies, using the raw `left_*`/`right_*`
  fields from `measure_jsw_array_v2` directly.
- The overlay (joint line + per-column femur/tibia edge points, valid columns only, with the
  minimum-gap point circled in each compartment) and a plain CLAHE base image (no lines) are
  both returned as base64 PNGs, so the UI's "show/hide measurement lines" toggle is a real
  image swap, not a CSS trick layered on a single baked-in image.
- A failed compartment returns `min_jsw`/`mean_jsw` = `None` and a fixed message string
  ("JSW could not be measured reliably on this image"), which the front end shows in place of
  a number.
- `ui/app.py`'s only change for this task is one new route, `POST /api/jsw` - it calls
  `jsw_bridge.measure()` and nothing else; it never touches `/api/analyze` or
  `model_bridge.analyze()`. The front end fires `/api/jsw` as a second, independent request
  alongside the existing `/api/analyze` call when "Run assessment" is clicked (same uploaded
  file, plus the optional knee-side selector), so a JSW failure can never affect the KL grade
  result and vice versa - verified in testing (see `ui/test_samples/jsw/`).

## 8. Screenshots

No browser-automation tool was available in this session (checked: no Playwright installed in
the venv, no computer-use/Chrome tool loaded) - same limitation hit by the earlier
`ui-redesign` task. Per that task's established approach, real evidence is saved to
`ui/test_samples/jsw/` instead of a faked screenshot: the rendered `GET /` HTML
(`index_page_rendered.html`, confirming the new "Joint space width" card and knee-side
selector are actually in the page), the full JSON test report
(`jsw_integration_report.txt`), and the actual overlay PNGs the `/api/jsw` route returned,
decoded and saved as real image files (`overlay_examples/`).
