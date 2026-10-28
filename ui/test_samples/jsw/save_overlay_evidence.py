import os
import base64
import uuid
import urllib.request
import json

IMAGES = [
    ("dataset/test/0/9003175L.png", "L"),
    ("dataset/test/3/9011053L.png", "L"),
    ("dataset/test/4/9012867R.png", "R"),
]
OUT_DIR = "ui/test_samples/jsw/overlay_examples"
os.makedirs(OUT_DIR, exist_ok=True)


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


for path, side in IMAGES:
    r = post_multipart("http://127.0.0.1:5050/api/jsw", path, {"side": side})
    name = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(OUT_DIR, f"{name}_overlay.png")
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(r["overlay_png_base64"]))
    print(f"saved {out_path} | status={r['status']} compartments={r['compartments']}")
