import cv2
import json
import time
import numpy as np
from enroll import build_model

EMPLOYEES_JSON = "employees.json"
OUT_DIR = "enrollment_crops"
MIN_FACE_SIZE = 160      # pixels
REQUIRED_SAMPLES = 5

def is_good_face(face, frame_shape):
    x1, y1, x2, y2 = face.bbox.astype(int)
    w = x2 - x1
    h = y2 - y1
    if min(w, h) < MIN_FACE_SIZE:
        return False
    if face.det_score < 0.7:
        return False
    return True

def main():
    employee_id = input("Enter employee ID: ").strip()
    if not employee_id:
        print("No ID. Exiting.")
        return

    with open(EMPLOYEES_JSON, "r", encoding="utf-8") as f:
        employees = json.load(f)

    if employee_id not in employees:
        print("Employee not found in employees.json")
        return

    app = build_model("cpu")
    cap = cv2.VideoCapture(0)

    collected = 0
    embeddings = []

    while collected < REQUIRED_SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        if not faces:
            cv2.imshow("Enroll", frame)
            cv2.waitKey(1)
            continue

        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        if not is_good_face(face, frame.shape):
            cv2.putText(frame, "MOVE CLOSER / FACE CAMERA", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow("Enroll", frame)
            cv2.waitKey(1)
            continue

        emb = face.embedding.astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        embeddings.append(emb)

        collected += 1
        print(f"Captured sample {collected}/{REQUIRED_SAMPLES}")
        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()

    employees[employee_id].setdefault("embeddings", {})
    employees[employee_id]["embeddings"].setdefault("office", [])

    for emb in embeddings:
        employees[employee_id]["embeddings"]["office"].append({
            "vector": emb.tolist(),
            "source": "enroll_station",
            "quality": "high",
            "ts": time.time()
        })

    with open(EMPLOYEES_JSON, "w", encoding="utf-8") as f:
        json.dump(employees, f, indent=2)

    print("Enrollment complete.")

if __name__ == "__main__":
    main()
