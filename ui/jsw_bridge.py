"""Single entry point the UI's new JSW panel calls: measure(image_path, side=None). Imports
scripts/measure_jsw_v2.py's own functions - no measurement logic is copied or reimplemented
here, and scripts/measure_jsw.py (the original method) is never touched or imported. This
module never calls the classifier and never influences the KL grade in any way.

Usage:
    from ui.jsw_bridge import measure
    result = measure("path/to/xray.png", side="R")   # side: "L" / "R" / None
"""
import os
import sys
import json
import base64

import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
for _p in (ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from measure_jsw_v2 import measure_jsw_array_v2, assign_compartments_v2, draw_overlay, apply_clahe

JSW_V2_DIR = os.path.join(ROOT, "outputs", "jsw_v2")
NOTE_UNITS = "Relative to joint width; no pixel spacing available, so not in mm."
NOTE_NOT_KL = "Not used to compute the KL grade."


def _png_base64(rgb_uint8):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else None


def _validation_status():
    """Reads Part 1's one-time test-split result (written by
    `measure_jsw_v2.py --mode eval --split test`) to decide the UI's validated/experimental
    badge. Never recomputes or re-decides the pass/fail call made in Part 1."""
    path = os.path.join(JSW_V2_DIR, "test_metrics.json")
    if not os.path.exists(path):
        return {"status": "experimental", "rho": None}
    with open(path) as f:
        metrics = json.load(f)
    rho = metrics["medial_min_jsw_vs_jsn_medial"]["rho"]
    return {"status": "validated" if metrics.get("pass") else "experimental", "rho": rho}


def measure(image_path, side=None):
    """side: "L", "R", or None/anything else (treated as not given).

    Returns a JSON-serializable dict:
      overlay_png_base64 / base_png_base64 - CLAHE image with/without the measurement lines
        drawn on it (for the UI's show/hide-lines toggle).
      compartments - a 2-item list, each {key, label, min_jsw, mean_jsw, failed, message}.
        Labeled Medial/Lateral when `side` is given, Image-left/Image-right side otherwise.
      status / status_rho - "validated" or "experimental", from Part 1's one-time test result.
      note_units / note_not_kl - the two fixed disclosure strings the UI always shows.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    m = measure_jsw_array_v2(rgb)
    clahe_rgb = apply_clahe(rgb)
    overlay = draw_overlay(clahe_rgb, m)

    def comp_entry(key, label, min_jsw, mean_jsw, failed):
        return {
            "key": key, "label": label,
            "min_jsw": None if failed else round(min_jsw, 4),
            "mean_jsw": None if failed else round(mean_jsw, 4),
            "failed": bool(failed),
            "message": "JSW could not be measured reliably on this image" if failed else None,
        }

    side_given = side in ("L", "R")
    if side_given:
        comp = assign_compartments_v2(side, m)
        compartments = [
            comp_entry("medial", "Medial", comp["medial_min_jsw"], comp["medial_mean_jsw"], comp["medial_failed"]),
            comp_entry("lateral", "Lateral", comp["lateral_min_jsw"], comp["lateral_mean_jsw"], comp["lateral_failed"]),
        ]
    else:
        compartments = [
            comp_entry("left", "Image-left side", m["left_min_jsw"], m["left_mean_jsw"], m["left_failed"]),
            comp_entry("right", "Image-right side", m["right_min_jsw"], m["right_mean_jsw"], m["right_failed"]),
        ]

    status = _validation_status()
    return {
        "overlay_png_base64": _png_base64(overlay),
        "base_png_base64": _png_base64(clahe_rgb),
        "compartments": compartments,
        "side_given": side_given,
        "status": status["status"],
        "status_rho": status["rho"],
        "note_units": NOTE_UNITS,
        "note_not_kl": NOTE_NOT_KL,
    }
