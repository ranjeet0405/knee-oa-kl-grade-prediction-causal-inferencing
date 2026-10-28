import json
import uuid
import urllib.request

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from ui.jsw_bridge import measure as jsw_measure_direct

# (path, side, expected_grade). expected_grade is the model's own PREDICTION on this image,
# from the earlier ui-redesign task's test_redesign_match.py run (direct model_bridge.analyze()
# call) on these exact same 5 images - NOT the true KL folder grade, which the model doesn't
# always match (e.g. the "KL1 sample" is predicted 0, a known, pre-existing model imperfection
# unrelated to this task). Used here as the "the KL grade didn't change" reference.
IMAGES = [
    ("dataset/test/0/9003175L.png", "L", 0),
    ("dataset/test/1/9001400L.png", "L", 0),
    ("dataset/test/2/9003316R.png", "R", 0),
    ("dataset/test/3/9011053L.png", "L", 3),
    ("dataset/test/4/9012867R.png", "R", 4),
]


def post_multipart(url, path, fields):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as fh:
        file_bytes = fh.read()
    parts = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"xray\"; filename=\"{path.split('/')[-1]}\"\r\n"
        f"Content-Type: image/png\r\n\r\n".encode() + file_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


report = []
all_ok = True
for path, side, expected_grade in IMAGES:
    analyze_resp = post_multipart("http://127.0.0.1:5050/api/analyze", path, {})
    grade_ok = analyze_resp["grade"] == expected_grade

    jsw_http = post_multipart("http://127.0.0.1:5050/api/jsw", path, {"side": side})
    jsw_direct = jsw_measure_direct(path, side=side)

    comp_match = True
    for ch, cd in zip(jsw_http["compartments"], jsw_direct["compartments"]):
        if ch["failed"] != cd["failed"]:
            comp_match = False
        elif not ch["failed"] and abs(ch["min_jsw"] - cd["min_jsw"]) > 1e-6:
            comp_match = False

    ok = grade_ok and comp_match and jsw_http["ok"]
    all_ok = all_ok and ok
    line = (f"{path}: grade={analyze_resp['grade']} (expected {expected_grade}, ok={grade_ok}) | "
            f"jsw_status={jsw_http['status']} rho={jsw_http['status_rho']} | "
            f"compartments_match_direct_call={comp_match} | "
            f"http_compartments={jsw_http['compartments']}")
    print(line)
    report.append(line)

print("\nALL_OK:", all_ok)
with open("ui/test_samples/jsw/jsw_integration_report.txt", "w") as f:
    f.write("\n\n".join(report) + f"\n\nALL_OK: {all_ok}\n")
