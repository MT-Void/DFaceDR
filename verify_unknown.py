import os
import json
import argparse
from datetime import datetime

import cv2
import numpy as np
from enroll import build_model


PENDING_DIR = "pending_verification"
VERIFIED_DIR = "verified_crops"
EMPLOYEES_JSON = "employees.json"
VERIFIED_LOG = "logs/verified.jsonl"


def log_verified(row: dict):
    os.makedirs(os.path.dirname(VERIFIED_LOG), exist_ok=True)
    with open(VERIFIED_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_latest_pending():
    if not os.path.exists(PENDING_DIR):
        return None
    files = [
        os.path.join(PENDING_DIR, f)
        for f in os.listdir(PENDING_DIR)
        if f.lower().endswith((".jpg",".jpeg",".png",".webp"))
    ]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]



def extract_embedding(app, img_bgr):
    faces = app.get(img_bgr)
    if not faces:
        return None, None
    
    # pick largest face
    faces.sort(key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    f0 = faces[0]

    emb = f0.embedding.astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb, float(f0.det_score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="Path to a pending crop. If omitted, uses latest.")
    ap.add_argument("--employee_id", default=None, help="Employee ID to assign. If omitted, will ask.")
    ap.add_argument("--context", choices=["office", "factory_ppe"], default="office")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = ap.parse_args()

    crop_path = args.path or pick_latest_pending()
    if not crop_path or not os.path.exist(crop_path):
        raise SystemExit("No pending crops found.")
    
    img = cv2.imread(crop_path)
    if img is None:
        raise SystemExit(f"Failed to read image: {crop_path}")
    
    # Show crop
    cv2.imshow("Verify crop (press any key)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    employee_id = args.employee_id or input ("Enter employee_id for this crop (or blank to skip): ").strip()
    if not employee_id:
        print("Skipped.")
        return
    
    # Load employees.json
    if not os.path.exists(EMPLOYEES_JSON):
        raise SystemExit(f"Missing {EMPLOYEES_JSON}")
    
    with open(EMPLOYEES_JSON, "r", encoding="utf-8") as f:
        employees = json.load(f)

    if employee_id not in employees:
        raise SystemExit(f"employee_id '{employee_id}' not found in employees.json")
    
    # Build model and embed
    app = build_model(args.device)
    emb, det = extract_embedding(app, img)
    if emb is None:
        raise SystemExit("No face detected in crop. Try a different crop.")
    
    # Append embedding to chose bucket
    emp = employees[employee_id]
    emp.setdefault("embeddings", {})
    emp["embeddings"].setdefault(args.context, [])

    emp["embeddings"][args.context].append({
        "vector": emb.tolist(),
        "source": "manual_verification_crop",
        "quality": "high",
        "det_score": round(det, 4),
        "ts": datetime.now().isoformat(timespec="seconds"),
        "path": crop_path.replace("\\", "/"),
    })

    # Save employees.json
    with open(EMPLOYEES_JSON, "w", encoding="utf-8") as f:
        json.dump(employees, f, ensure_ascii=False, indent=2)

    # Move crop to verified folder so you don't verify it twice
    os.makedirs(VERIFIED_DIR, exist_ok=True)
    emp_dir = os.path.join(VERIFIED_DIR, employee_id)
    os.makedirs(emp_dir, exist_ok=True)

    base = os.path.basename(crop_path)
    new_path = os.path.join(emp_dir, base)
    os.replace(crop_path, new_path)

    # Log action
    log_verified({
        "type": "verified",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "employee_id": employee_id,
        "context": args.context,
        "crop_src": crop_path.replace("\\", "/"),
        "crop_dst": new_path.replace("\\", "/"),
    })


    print(f"[OK] Added embeddings to {employee_id} -> embeddings.{args.context}")
    print(f"[OK] Moved crop to: {new_path}")


if __name__ == "__main__":
    main()