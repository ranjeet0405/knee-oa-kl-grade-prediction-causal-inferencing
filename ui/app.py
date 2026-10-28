"""Flask UI, adapted from the teammate's fork (webapp/app.py) to call this project's final
model through ui/model_bridge.py instead of loading checkpoints directly. No model logic
lives here - every prediction, explanation, similar-case lookup and causal estimate comes
from model_bridge.analyze().

Changes from the fork's original app.py (logged in detail in ui/integration_log.md):
  - the hardcoded Path.home()/"Desktop"/<fork's own folder name> project-path assumption is
    gone; this file locates the project root relative to itself, so it works on any machine.
  - no direct `from scripts.predict import ...` - everything model-related goes through
    model_bridge.analyze() instead.
  - the kl_final/Hybrid 2 model dropdown is gone - model_bridge always serves this project's
    final configuration (the validated ensemble, or kl_final alone if the ensemble didn't
    beat it - decided once, at startup, exactly as predict_extended.py decides it).
  - age/BMI/family history are no longer Hybrid-2-only fields - they feed the causal panel
    for every request now (never the classifier itself, same rule as predict_extended.py).

    python ui/app.py
"""
import os
import sys
import csv
import json
import base64
import tempfile
import traceback

from flask import Flask, render_template, request, jsonify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
for _p in (ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ui.model_bridge import analyze, DISCLAIMER
# Reused directly for the redesign's live BMI what-if slider - same existing causal module
# model_bridge itself imports from, not a reimplementation. model_bridge.py is unchanged.
from causal_model import load_or_fit_causal_models, out_of_range_messages, expected_kl, contribution_bmi
# New JSW panel - goes through ui/jsw_bridge.py only; measure_jsw.py and model_bridge.py are
# both untouched, and this never feeds into or changes the KL grade.
from ui.jsw_bridge import measure as jsw_measure

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024
CAUSAL_DIR = os.path.join(ROOT, "outputs", "results", "causal")
EXTENDED_CAUSAL_DIR = os.path.join(ROOT, "outputs", "extended", "causal")
FINAL_TABLES_DIR = os.path.join(ROOT, "final_output", "tables")
DATASET_DIR = os.path.realpath(os.path.join(ROOT, "dataset"))

_causal_cache = {}


def _causal_models():
    if "models" not in _causal_cache:
        _causal_cache["models"] = load_or_fit_causal_models()
    return _causal_cache["models"]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES


def allowed_file(filename):
    return bool(filename) and "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_optional_float(raw):
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _attach_case_thumbnails(result):
    """UI-only decoration: reads the small training-case thumbnails analyze() already
    pointed at (by path) and attaches a base64 PNG for the redesign's case cards. Pure
    file I/O - no model logic. Any single bad path is skipped, not fatal."""
    import cv2

    for case in result.get("similar_cases", []):
        case["thumbnail_png_base64"] = None
        rel_path = case.get("path")
        if not rel_path:
            continue
        abs_path = os.path.realpath(os.path.join(ROOT, rel_path))
        if not abs_path.startswith(DATASET_DIR) or not os.path.isfile(abs_path):
            continue
        try:
            img = cv2.imread(abs_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (160, 160))
            ok, buf = cv2.imencode(".png", img)
            if ok:
                case["thumbnail_png_base64"] = base64.b64encode(buf.tobytes()).decode("utf-8")
        except Exception:
            continue


@app.post("/api/jsw")
def run_jsw():
    """Joint space width panel - entirely separate from /api/analyze. Calls
    ui/jsw_bridge.measure() only; never touches the classifier and never changes the KL
    grade. side: "L"/"R"/omitted - an optional hint for medial/lateral labeling only."""
    if "xray" not in request.files or not request.files["xray"].filename:
        return jsonify({"ok": False, "message": "Select a knee X-ray first."}), 400
    file = request.files["xray"]
    if not allowed_file(file.filename):
        return jsonify({"ok": False, "message": "Only PNG, JPG and JPEG images are supported."}), 400

    side = request.form.get("side")
    side = side if side in ("L", "R") else None

    tmp_path = None
    try:
        suffix = "." + file.filename.rsplit(".", 1)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        result = jsw_measure(tmp_path, side=side)
        result["ok"] = True
        return jsonify(result)
    except Exception:
        app.logger.error("JSW measurement failed:\n%s", traceback.format_exc())
        return jsonify({"ok": False, "message": "JSW measurement failed. Check the Flask terminal for details."}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/")
def index():
    return render_template("index.html", disclaimer=DISCLAIMER)


@app.get("/api/health")
def health():
    import torch
    return jsonify({"ok": True, "service": "Knee OA KL-grade research prototype",
                    "device_available": "cuda" if torch.cuda.is_available() else "cpu"})


@app.get("/api/causal")
def causal_results():
    """Serves the project's existing causal-analysis output files, unmodified and read-only."""
    import csv
    import json as json_mod

    def read_csv(name):
        path = os.path.join(CAUSAL_DIR, name)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    estimates = read_csv("causal_estimates.csv")
    query_results = read_csv("causal_query_results.csv")
    models_path = os.path.join(CAUSAL_DIR, "causal_query_models.json")
    models = None
    if os.path.isfile(models_path):
        with open(models_path, "r", encoding="utf-8") as fh:
            models = json_mod.load(fh)

    if estimates is None and query_results is None and models is None:
        return jsonify({"ok": False, "error": "No causal output files were found."}), 404

    return jsonify({"ok": True, "causal_estimates": estimates or [], "query_results": query_results or [],
                    "models": models or {}})


@app.post("/api/whatif")
def whatif():
    """Live BMI what-if slider for the redesigned causal panel. Calls causal_model.py's own
    expected_kl()/out_of_range_messages() directly - the same functions model_bridge.analyze()
    uses - so a slider drag never computes anything the rest of the project doesn't already
    stand behind. Out-of-range age/BMI is refused, exactly as causal_query.py refuses it."""
    data = request.get_json(silent=True) or {}
    age = parse_optional_float(data.get("age"))
    bmi = parse_optional_float(data.get("bmi"))
    family_history = data.get("family_history") if data.get("family_history") in ("yes", "no") else None

    if age is None or bmi is None:
        return jsonify({"ok": False, "message": "Age and BMI are both required for the what-if slider."}), 400

    models = _causal_models()
    refusal = out_of_range_messages(models, age, bmi)
    if refusal:
        return jsonify({"ok": False, "refused": True, "messages": refusal})

    famkr = {"yes": 1, "no": 0}.get(family_history, models["most_common_famkr"])
    clipped, was_clipped, raw = expected_kl(models, age, bmi, famkr)
    contribution = contribution_bmi(models, bmi)
    return jsonify({"ok": True, "expected_kl": {"value": clipped, "clipped": bool(was_clipped)},
                    "bmi_contribution": contribution})


@app.get("/api/causal_insights")
def causal_insights():
    """Assembles the Causal insights page's numbers straight from this project's existing
    result files (never invented, never recomputed) - same discipline as
    scripts/make_final_figures.py. Read-only."""

    def read_csv_rows(path):
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    estimates = read_csv_rows(os.path.join(CAUSAL_DIR, "causal_estimates.csv"))
    query_results = read_csv_rows(os.path.join(CAUSAL_DIR, "causal_query_results.csv"))
    refutations = read_csv_rows(os.path.join(CAUSAL_DIR, "refutation_results.csv"))
    injury = read_csv_rows(os.path.join(EXTENDED_CAUSAL_DIR, "injury_estimates.csv"))
    injury_refutations = read_csv_rows(os.path.join(EXTENDED_CAUSAL_DIR, "injury_refutations.csv"))
    t7 = read_csv_rows(os.path.join(FINAL_TABLES_DIR, "T7_causal.csv"))

    methods = []
    for row in estimates:
        if row.get("Analysis") == "primary_one_knee" and row.get("Outcome") == "KL_grade (0-4)":
            label = {"propensity_score_matching": "Propensity score matching",
                     "linear_regression": "Regression adjustment"}.get(row["Method"], row["Method"])
            methods.append({"label": label, "ate": float(row["ATE"]), "n": row.get("N")})
    forest_row = next((r for r in t7 if r["Finding"].startswith("Obesity effect - personalized mean")), None)
    if forest_row:
        import re
        m = re.search(r"[-+]?\d*\.\d+|\d+", forest_row["Value"])
        if m:
            methods.append({"label": "Causal forest (average)", "ate": float(m.group()), "n": None})

    age_group_ci = []
    for row in query_results:
        if row["Effect"].startswith("Obesity, age"):
            age_group_ci.append({"label": row["Effect"], "estimate": float(row["coef"]),
                                  "ci_low": float(row["ci_low"]), "ci_high": float(row["ci_high"])})

    evalue = {}
    for row in t7:
        if "E-value (point estimate)" in row["Finding"]:
            evalue["point"] = float(row["Value"])
        elif "E-value (CI bound)" in row["Finding"]:
            evalue["ci_bound"] = float(row["Value"])

    refuters = [{"name": r["Refuter"].replace("_", " "), "new_effect": float(r["New_effect"]),
                 "original_effect": float(r["Original_effect"]), "p_value": float(r["p_value"])}
                for r in refutations]

    injury_out = {}
    if injury:
        injury_out = {r["Method"]: float(r["ATE"]) for r in injury}
    injury_out["refuters"] = [{"name": r["refuter"].replace("_", " "), "new_effect": float(r["new_effect"]),
                                "original_effect": float(r["original_effect"])}
                               for r in injury_refutations]

    return jsonify({"ok": True, "methods": methods, "age_group_ci": age_group_ci, "evalue": evalue,
                    "refuters": refuters, "injury": injury_out})


@app.post("/api/patient")
def save_patient():
    """Validates patient details for this prototype; nothing is persisted."""
    data = request.get_json(silent=True) or {}
    required = ["patient_id", "age", "side"]
    missing = [k for k in required if not str(data.get(k, "")).strip()]
    if missing:
        return jsonify({"ok": False, "error": "Missing required fields: " + ", ".join(missing)}), 400
    try:
        age = float(data["age"])
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Age must be a valid number."}), 400
    if not 18 <= age <= 120:
        return jsonify({"ok": False, "error": "Age must be between 18 and 120."}), 400
    side = str(data["side"]).strip().upper()
    if side not in {"L", "R", "LEFT", "RIGHT"}:
        return jsonify({"ok": False, "error": "Side must be left or right."}), 400
    return jsonify({"ok": True, "message": "Patient details validated. No data was stored.",
                    "patient": {"patient_id": str(data["patient_id"]).strip(), "age": age, "side": side}})


@app.post("/api/analyze")
def run_analyze():
    if "xray" not in request.files or not request.files["xray"].filename:
        return jsonify({"ok": False, "message": "Select a knee X-ray first."}), 400
    file = request.files["xray"]
    if not allowed_file(file.filename):
        return jsonify({"ok": False, "message": "Only PNG, JPG and JPEG images are supported."}), 400

    age = parse_optional_float(request.form.get("age"))
    bmi = parse_optional_float(request.form.get("bmi"))
    family_history = request.form.get("family_history") or None
    if family_history not in ("yes", "no"):
        family_history = None

    tmp_path = None
    try:
        suffix = "." + file.filename.rsplit(".", 1)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        result = analyze(tmp_path, age=age, bmi=bmi, family_history=family_history)
        result["ok"] = True
        result["disclaimer"] = DISCLAIMER
        _attach_case_thumbnails(result)
        return jsonify(result)
    except Exception:
        app.logger.error("Inference failed:\n%s", traceback.format_exc())
        return jsonify({"ok": False, "message": "Inference failed. Check the Flask terminal for details."}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"ok": False, "message": "Image exceeds the 12 MB upload limit."}), 413


if __name__ == "__main__":
    print("=" * 55)
    print("Knee OA KL-grade research webapp (ui/)")
    print(f"Project root: {ROOT}")
    print(f"Disclaimer: {DISCLAIMER}")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5050, debug=False)
