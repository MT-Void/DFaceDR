import json
import os
from datetime import datetime

import cv2
import numpy as np

from enroll import build_model  # uses your existing build_model()

EMPLOYEES_JSON = "employees.json"

def norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def best_face(app, img_bgr):
    faces = app.get(img_bgr)
    if not faces:
        return None
    # pick largest face
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

def main():
    app = build_model("cpu")

    with open(EMPLOYEES_JSON, "r", encoding="utf-8") as f:
        employees = json.load(f)

    ok = 0
    fail = 0

    for emp_id, emp in employees.items():
        ingest = (emp.get("ingest_status") or {}).get("hr_photo") or {}
        path = ingest.get("path")

        if not path:
            continue

        # Skip if already ingested successfully
        if ingest.get("ok") is True and (emp.get("embeddings") or {}).get("seed_hr"):
            ok += 1
            continue

        path_fs = path.replace("/", "\\")
        if not os.path.exists(path_fs):
            print(f"[FAIL] {emp_id}: HR path not found: {path_fs}")
            emp.setdefault("ingest_status", {}).setdefault("hr_photo", {})["ok"] = False
            fail += 1
            continue

        img = cv2.imread(path_fs)
        if img is None:
            print(f"[FAIL] {emp_id}: cv2.imread failed: {path_fs}")
            emp.setdefault("ingest_status", {}).setdefault("hr_photo", {})["ok"] = False
            fail += 1
            continue

        face = best_face(app, img)
        if face is None:
            print(f"[FAIL] {emp_id}: no face found in HR photo")
            emp.setdefault("ingest_status", {}).setdefault("hr_photo", {})["ok"] = False
            fail += 1
            continue

        emb = norm(face.embedding)

        emp.setdefault("embeddings", {})
        emp["embeddings"].setdefault("seed_hr", [])
        emp["embeddings"]["seed_hr"] = [{
            "vector": emb.tolist(),
            "source": "hr_photo",
            "quality": "low",
            "ts": datetime.now().isoformat(timespec="seconds")
        }]

        emp.setdefault("ingest_status", {}).setdefault("hr_photo", {})
        emp["ingest_status"]["hr_photo"]["ok"] = True
        emp["ingest_status"]["hr_photo"]["path"] = path_fs

        print(f"[OK] {emp_id}: embedded from {path_fs}")
        ok += 1

    with open(EMPLOYEES_JSON, "w", encoding="utf-8") as f:
        json.dump(employees, f, indent=2)

    print(f"[DONE] HR ingestion complete. OK={ok}, FAIL={fail}")

if __name__ == "__main__":
    main()
