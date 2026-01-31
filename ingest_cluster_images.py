import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np

from enroll import build_model  # uses your existing InsightFace setup

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Keep these modest; matcher.py already has strict gates
MIN_DET_SCORE = 0.35
MIN_FACE_H = 60

def norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def list_images(folder: str):
    out = []
    for name in os.listdir(folder):
        p = os.path.join(folder, name)
        if os.path.isfile(p) and os.path.splitext(name.lower())[1] in IMG_EXTS:
            out.append(p)
    return sorted(out)

def pad_and_upscale(img, target=640):
    h, w = img.shape[:2]
    
    pad = int(0.25 * max(h, w))
    img2 = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0,0,0))
    return cv2.resize(img2, (target, target), interpolation=cv2.INTER_CUBIC)

def pick_best_face(app, img_bgr):
    faces = app.get(img_bgr)
    if not faces:
        return None
    # pick largest face
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

def ingest_employee_folder(app, emp_id: str, folder: str, context: str, max_per_person: int):
    embeddings = []
    for path in list_images(folder):
        img = cv2.imread(path)
        if img is None:
            continue

        face = pick_best_face(app, img)
        if face is None:
            img2 = pad_and_upscale(img, target= 640)
            face = pick_best_face(app, img2)

        if face is None:
            print("[NOFACE]", path)
            continue
            
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_h = int(y2 - y1)
        det = float(face.det_score)

        if det < MIN_DET_SCORE:
            print("[LOWDET]", path, det)
            continue

        if face_h < MIN_FACE_H:
            print("[SMALL]", path, face_h)
            continue

        emb = norm(face.embedding)
        embeddings.append((emb, det, face_h, path))

    # Prefer higher det + bigger face
    embeddings.sort(key=lambda t: (t[1], t[2]), reverse=True)
    embeddings = embeddings[:max_per_person]

    items = []
    for emb, det, face_h, path in embeddings:
        items.append({
            "vector": emb.tolist(),
            "source": "cluster_ingest",
            "quality": "high",
            "ts": datetime.now().isoformat(timespec="seconds"),
            "meta": {
                "image": path.replace("\\", "/"),
                "det_score": round(det, 4),
                "face_h": face_h,
            }
        })

    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--employees_json", default="employees.json")
    ap.add_argument("--clusters_dir", required=True, help="Folder containing subfolders named by employee_id")
    ap.add_argument("--context", default="office", choices=["office", "factory_ppe"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "gpu"])
    ap.add_argument("--max_per_person", type=int, default=5)
    args = ap.parse_args()

    app = build_model(args.device)

    with open(args.employees_json, "r", encoding="utf-8") as f:
        employees = json.load(f)

    clusters_dir = args.clusters_dir
    subfolders = [d for d in os.listdir(clusters_dir) if os.path.isdir(os.path.join(clusters_dir, d))]
    subfolders.sort()

    ok = 0
    skip = 0
    fail = 0

    for emp_id in subfolders:
        folder = os.path.join(clusters_dir, emp_id)
        if emp_id not in employees:
            print(f"[SKIP] {emp_id}: not in employees.json (add entry first)")
            skip += 1
            continue

        items = ingest_employee_folder(app, emp_id, folder, args.context, args.max_per_person)
        if not items:
            print(f"[FAIL] {emp_id}: no usable faces found in {folder}")
            fail += 1
            continue

        employees[emp_id].setdefault("embeddings", {})
        employees[emp_id]["embeddings"].setdefault(args.context, [])

        before = len(employees[emp_id]["embeddings"][args.context])
        employees[emp_id]["embeddings"][args.context].extend(items)
        after = len(employees[emp_id]["embeddings"][args.context])

        print(f"[OK] {emp_id}: added {after - before} embeddings to embeddings.{args.context}")
        ok += 1

    with open(args.employees_json, "w", encoding="utf-8") as f:
        json.dump(employees, f, indent=2)

    print(f"[DONE] cluster ingestion complete. OK={ok}, SKIP={skip}, FAIL={fail}")

if __name__ == "__main__":
    main()
