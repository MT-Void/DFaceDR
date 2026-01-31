import os
import json
import cv2
import numpy as np

from enroll import build_model

SESSIONS_DIR = "sessions"
EMPLOYEES_JSON = "employees.json"

def norm(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

def best_face(app, img):
    faces = app.get(img)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

def embed_image(app, path):
    img = cv2.imread(path)
    if img is None:
        return None
    f = best_face(app, img)
    if f is None:
        return None
    return norm(f.embedding)

def cosine(a, b):
    return float(np.dot(a, b))

def main():
    app = build_model("cpu")

    with open(EMPLOYEES_JSON, "r", encoding="utf-8") as f:
        employees = json.load(f)

    emp_id = input("Employee ID (HR-seeded:) ").strip()
    emp = employees.get(emp_id)
    if not emp:
        print("No such employee_id in employees.json")
        return
    
    seed = (emp.get("embeddings") or {}).get("seed_hr") or []
    if not seed or not seed[0].get("vector"):
        print("This employee has no seed_hr vector. Run HR ingestion first.")
        return
    
    hr_emb = norm(np.array(seed[0]["vector"], dtype=np.float32))

    # scan session folders
    scores = []
    for d in sorted(os.listdir(SESSIONS_DIR)):
        folder = os.path.join(SESSIONS_DIR, d)
        if not os.path.isdir(folder):
            continue

        # pick a few frames from the folder
        imgs = [os.path.join(folder, x) for x in sorted(os.listdir(folder)) if x.lower().endswith((".jpg",".jpeg",".png"))]
        if not imgs:
            continue

        best = None
        for p in imgs[:10]: # try up to 10 images
            emb = embed_image(app, p)
            if emb is None:
                continue
            s = cosine(hr_emb, emb)
            if best is None or s > best: 
                best = s

        if best is not None:
            scores.append((d, best))

    scores.sort(key=lambda x: x[1], reverse=True)

    print("\nTop matches:")
    for folder, s in scores[:10]:
        print(f"    {folder}    sim={s:.3f}")
    
if __name__ == "__main__":
    main()