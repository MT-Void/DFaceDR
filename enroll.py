import os
import time
import numpy as np
import cv2
import argparse

from insightface.app import FaceAnalysis

DATA_DIR = "data/authorized_faces"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def load_db(base_dir: str = DATA_DIR):
    """
    Loads centroid embeddings for each enrolled person.
    Returns dict: person_id -> centroid_embedding
    """
    db = {}
    if not os.path.exists(base_dir):
        return db
    for person_id in os.listdir(base_dir):
        cpath = os.path.join(base_dir, person_id, "centroid.npy")
        if os.path.exists(cpath):
            db[person_id] = np.load(cpath).astype(np.float32)
    return db


def pick_main_face(faces):
    """Pick the largest detected face"""
    if not faces:
        return None
    faces_sorted = sorted(
        faces, 
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        reverse=True
    )
    return faces_sorted[0]


def build_model(device: str = "cpu"):
    ctx_id = -1 if device.lower() == "cpu" else 0
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def enroll(person_id: str, cam_index: int, samples: int, device: str):
    """
    Press SPACE to capture a sample, ESC to quit.
    Saves multiple embeddings anda centroid.
    """
    ensure_dir(DATA_DIR)
    person_dir = os.path.join(DATA_DIR, person_id)
    ensure_dir(person_dir)

    app = build_model(device)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webacm. Try a different --cam index")
    
    captured = []
    print(f"[ENROLL] Person: {person_id}")
    print("Press SPACE to capture a sample. Press ESC to exit")


    while True:
        ok, frame = cap.read()
        if not ok:
            break

        faces = app.get(frame)
        f = pick_main_face(faces)

        label = "No face"
        if f is not None:
            x1, y1, x2, y2 = f.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Face score: {float(f.det_score):.2f}"

        cv2.putText(frame, f"{label} | Captured: {len(captured)}/{samples}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Enroll", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        if key == 32:   # SPACE
            if f is None:
                print("No face detected. Try again")
                continue
            if float(f.det_score) < 0.60:
                print("Face detected but confidence is low. Move closer / improve lighting")
                continue

            emb = f.embedding.astype(np.float32)
            captured.append(emb)
            print(f"Captured sample {len(captured)}/{samples}")

            # Small pause so you can pose between samples
            time.sleep(0.25)

            if len(captured) >= samples:
                break

    cap.release()
    cv2.destroyAllWindows()

    if len(captured) < max(5, samples // 2):
        raise RuntimeError("Not enough good samples captured. Try again with better lighting / closer camara")
    
    # save embeddings
    for i, emb in enumerate(captured):
        np.save(os.path.join(person_dir, f"emb_{i:03d}.npy"), emb)

    # save centroid (normalized)
    centroid = np.mean(np.stack(captured), axis=0)
    centroid = centroid/ (np.linalg.norm(centroid) + 1e-9)
    np.save(os.path.join(person_dir, "centroid.npy"), centroid)

    print(f"[DONE] Enrolled {person_id}. Saved {len(captured)} samples to {person_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()

    enroll(args.person, args.cam, args.samples, args.device)

if __name__ == "__main__":
    main()