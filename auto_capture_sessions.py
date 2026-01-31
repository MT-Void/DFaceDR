import os
import time
from datetime import datetime

import cv2
import numpy as np

from enroll import build_model


_SOUND_CACHE = {}
_SOUND_INIT = False

SOUND_FILES = {
    "start": os.path.join("sounds", "starttakingpic.mp3"),
    "done": os.path.join("sounds", "donetakingpic.mp3"),
    "ready": os.path.join("sounds", "readytakingpic.mp3"),
}

OUT_ROOT = "sessions"
os.makedirs(OUT_ROOT, exist_ok=True)

# Trigger (tune for camera/webcam)
MIN_DET_SCORE = 0.75
MIN_FACE_H = 140

# Session behavior
TARGET_COUNT = 8
SHOT_INTERVAL = 0.6          # seconds between shots
START_HOLD_SECONDS = 0.6     # face must be "good" this long to start
COOLDOWN_SECONDS = 10.0

# Crop margin (IMPORTANT: gives better downstream embedding)
CROP_MARGIN = 0.35

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def expand_bbox(x1, y1, x2, y2, w, h, margin=0.35):
    bw = x2 - x1
    bh = y2 - y1
    mx = int(margin * bw)
    my = int(margin * bh)
    X1 = clamp(x1 - mx, 0, w - 1)
    Y1 = clamp(y1 - my, 0, h - 1)
    X2 = clamp(x2 + mx, 0, w - 1)
    Y2 = clamp(y2 + my, 0, h - 1)
    return X1, Y1, X2, Y2

def init_sound():
    global _SOUND_INIT
    if _SOUND_INIT:
        return
    try:
        import pygame
        pygame.mixer.init() # uses default audio device
        _SOUND_INIT = True
    except Exception as e:
        print("[SOUND] Disabled (pygame init failed):", e)
        _SOUND_INIT = False

def play_sound(kind: str):
    init_sound()
    if not _SOUND_INIT:
        return

    path = SOUND_FILES.get(kind)
    if not path:
        return

    path = os.path.abspath(path)
    if not os.path.exists(path):
        print("[SOUND] Missing file:", path)
        return

    try:
        import pygame
        snd = _SOUND_CACHE.get(path)
        if snd is None:
            snd = pygame.mixer.Sound(path)
            _SOUND_CACHE[path] = snd\

        # Stop any currently playing sound so prompts don't overlap
        pygame.mixer.stop()
        snd.play()
    except Exception as e:
        print("[SOUND] play failed:", e) 


def main():
    app = build_model("cpu")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    cv2.namedWindow("Auto Session Capture", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Auto Session Capture", 1280, 720)

    state = "IDLE"  # IDLE, CAPTURING, COOLDOWN
    good_since = None
    cooldown_until = 0.0

    session_dir = None
    captured = 0
    last_shot = 0.0

    print("[INFO] Auto mode. ESC to quit.")
    print(f"[INFO] Starts when face_h>={MIN_FACE_H} and det>={MIN_DET_SCORE} for {START_HOLD_SECONDS}s")
    print(f"[INFO] Captures {TARGET_COUNT} images then cooldown {COOLDOWN_SECONDS}s")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        now = time.time()
        h, w = frame.shape[:2]

        faces = app.get(frame)
        face = None
        if faces:
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

        # UI overlay
        overlay = frame.copy()

        # Cooldown logic
        if state == "COOLDOWN":
            remaining = max(0.0, cooldown_until - now)
            cv2.putText(overlay, f"COOLDOWN: {remaining:.1f}s",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            if now >= cooldown_until:
                state = "IDLE"
                good_since = None
                play_sound("ready")

        if face is not None:
            x1, y1, x2, y2 = face.bbox.astype(int)
            det = float(face.det_score)
            face_h = int(y2 - y1)

            good = (det >= MIN_DET_SCORE) and (face_h >= MIN_FACE_H) and (state != "COOLDOWN")

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255,255,255), 2)
            cv2.putText(overlay, f"det={det:.2f} h={face_h}",
                        (x1, max(20, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            if state == "IDLE":
                if good:
                    if good_since is None:
                        good_since = now
                    held = now - good_since
                    cv2.putText(overlay, f"HOLD... {held:.1f}/{START_HOLD_SECONDS:.1f}s",
                                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

                    if held >= START_HOLD_SECONDS:
                        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        session_dir = os.path.join(OUT_ROOT, f"{ts}_cam0")
                        os.makedirs(session_dir, exist_ok=True)
                        state = "CAPTURING"
                        captured = 0
                        last_shot = 0.0
                        play_sound("start")
                        print("[START] Session:", session_dir)
                else:
                    good_since = None

            elif state == "CAPTURING":
                cv2.putText(overlay, f"CAPTURING: {captured}/{TARGET_COUNT}",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

                if good and (now - last_shot >= SHOT_INTERVAL) and captured < TARGET_COUNT:
                    X1, Y1, X2, Y2 = expand_bbox(x1, y1, x2, y2, w, h, margin=CROP_MARGIN)
                    crop = frame[Y1:Y2, X1:X2].copy()
                    out_path = os.path.join(session_dir, f"frame_{captured+1:03d}.jpg")
                    cv2.imwrite(out_path, crop)
                    captured += 1
                    last_shot = now
                    print("[SAVE]", out_path)

                if captured >= TARGET_COUNT:
                    print("[DONE] Session complete:", session_dir)
                    play_sound("done")
                    state = "COOLDOWN"
                    cooldown_until = now + COOLDOWN_SECONDS
                    session_dir = None
                    good_since = None

        else:
            if state == "IDLE":
                good_since = None
            cv2.putText(overlay, "No face detected", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.putText(overlay, "ESC: Quit", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow("Auto Session Capture", overlay)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
