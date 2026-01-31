import argparse
import numpy as np
import cv2
import time
import os
import json
import pygame
from datetime import datetime

from enroll import build_model, load_db, DATA_DIR, cosine_sim
from matcher import no_guess_match


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMPLOYEES_JSON = os.path.join(BASE_DIR, "employees.json")

def save_snapshot(frame, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    cv2.imwrite(path, frame)
    print(f"[SNAPSHOT] Saved: {path}")
    return path


def save_unknown_snapshot(frame, cam_source: int, score: float, out_dir="logs/unknown"):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cam_tag = "rtsp" if isinstance(cam_source, str) else cam_source
    filename = f"{ts}_cam{cam_tag}_unknown_sim{score:.2f}.jpg"
    return save_snapshot(frame, out_dir, filename)


def save_unauth_snapshot(frame, cam_source: int, person_id: str, score: float, out_dir="logs/unauthorized"):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_id = person_id.replace(" ", "_")
    cam_tag = "rtsp" if isinstance(cam_source, str) else cam_source
    filename = f"{ts}_cam{cam_tag}_unauth_{safe_id}_sim{score:.2f}.jpg"
    return save_snapshot(frame, out_dir, filename)


def save_face_crop(frame, bbox, out_dir="pending_verification", filename=None, pad_ratio=0.25):
    """
    Saves a padded face crop from the frame.
    bbox = [x1, y1, x2, y2]
    """
    os.makedirs(out_dir, exist_ok=True)

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    crop = frame[cy1:cy2, cx1:cx2].copy()
    if crop.size == 0:
        return None
    
    if filename is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{ts}_crop.jpg"

    path = os.path.join(out_dir, filename)
    ok = cv2.imwrite(path, crop)
    if not ok:
        print(f"[CROP] FAILED to write: {path}")
        return None
    print(f"[CROP] Saved: {path}")
    return path

_audio_inited = False
_sound_cache = {}

def init_audio():
    global _audio_inited
    if _audio_inited:
        return
    pygame.mixer.init()
    _audio_inited = True


def play_alarm_sound(path: str):
    """
    Plays an MP3/MAV using pygame. Cache loaded sounds for low-latency playback
    """
    if not path or path == "silent":
        return
    if not os.path.exists(path):
        print(f"[ALARM] Sound not found: {path}")
        return
    if path not in _sound_cache:
        # pygame supports WAV/OGG well; MP3 works in most cases on Windows
        _sound_cache[path] = pygame.mixer.Sound(path)
    _sound_cache[path].play()


def letterbox_to(frame, target_w=1920, target_h=1080):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


def log_event(event: dict, log_path="logs/events.jsonl"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[LOG] {event['type']} -> {log_path}")


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0

    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    return float(interArea / (boxAArea + boxBArea - interArea + 1e-9))


def point_in_polygon(pt, polygon):
    x, y = pt
    poly = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def face_in_any_roi(bbox, roi_polygons):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    for poly in roi_polygons:
        if point_in_polygon((cx, cy), poly):
            return True
    return False


def load_zone_config(path: str, zone_name: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for z in cfg.get("zones", []):
        if z.get("name") == zone_name:
            return z
    raise RuntimeError(f"Zone '{zone_name}' not found in {path}")

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|fflags;nobuffer"

def open_capture(cam_source):
    if isinstance(cam_source, str):
        cap = cv2.VideoCapture(cam_source, cv2.CAP_FFMPEG)
    else:
        cap = cv2.VideoCapture(int(cam_source), cv2.CAP_DSHOW)
    if cap.isOpened():
        # Keep latency down
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return cap


def recognize_with_zone(cam_source, device, zone_cfg):
    app = build_model(device)

    alarm_until = 0.0
    alarm_text = ""

    allowed_ids = set(zone_cfg.get("allowed_ids", []))
    roi_polygons = zone_cfg.get("roi_polygons", [])
    zone_name = zone_cfg.get("name", "zone")
    alarm_sounds = zone_cfg.get("alarm_sounds") or {}
    alarm_policy = zone_cfg.get("alarm_policy") or {}

    print(f"[ZONE] {zone_name}")
    print(f"[ALLOWLIST] {len(allowed_ids)} allowed IDs")

    cv2.namedWindow("Recognize", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Recognize", 1280, 720)

    EMP_PATH = os.path.abspath("employees.json")
    print("[DEBUG] employees.json path:", EMP_PATH)

    with open(EMPLOYEES_JSON, "r", encoding="utf-8") as f:
        employees_db = json.load(f)

    print("[DEBUG] employees.json loaded keys:", len(employees_db))
    print("[DEBUG] sample 6666 office embeddings:",
        len(employees_db.get("6666", {}).get("embeddings", {}).get("office", [])))


    tracks = {}
    next_track_id = 1

    track_ttl_seconds = 1.0
    match_iou_threshold = 0.3

    confirm_seconds = 2.0
    cooldown_seconds = 10.0

    reconnect_delay = 2.0   # seconds
    
    while True:
        cap = open_capture(cam_source)
        if not cap.isOpened():
            print("[VIDEO] Failed to open stream. Retrying...")
            time.sleep(reconnect_delay)
            continue

        print("[VIDEO] Stream opened:", cam_source)
    
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"[VIDEO] Read failed. Reconeccting...")
                break # break inner loop, reopen stream

            now = time.time()
            h, w = frame.shape[:2]
            cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
            cv2.putText(frame, "RESTRICTED", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, "FREE ZONE", (w // 2 + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # draw ROI polygons for visibility
            for poly in roi_polygons:
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 255), thickness=2)

            faces = app.get(frame)

            for f in faces:
                x1, y1, x2, y2 = f.bbox.astype(int)
                det = float(f.det_score)
                emb = f.embedding.astype(np.float32)

                bbox = [int(x1), int(y1), int(x2), int(y2)]

                # Only process if inside ROI (if ROIs are configured)
                if roi_polygons and not face_in_any_roi(bbox, roi_polygons):
                    # still draw a box, but don’t evaluate alarm logic
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    cv2.putText(frame, "OUTSIDE ZONE", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    continue

                # Recognition
                print("[DEBUG] det_score", det, "face_h", (y2 - y1))
                employee_id, score = no_guess_match(
                    emb=emb,
                    employees=employees_db,
                    context="office",      # or "factory_ppe"
                    det_score=det,
                    face_h=(y2 - y1),
                )
                print("[DEBUG] match result:", employee_id, score)
                if employee_id is None and score is None:
                    print("[DEBUG] no_guess_match return (None,None) -> gated as unknown")

                is_unknown = (employee_id is None)
                is_unauthorized = (not is_unknown) and (employee_id not in allowed_ids)

                # for display
                if employee_id is None:
                    status = "UNKNOWN"
                    display_id = "unknown"
                else:
                    status = "UNAUTHORIZED" if is_unauthorized else "AUTHORIZED"
                    display_id = employee_id

                #   Score can be None when unknown
                score_for_text = -1.0 if score is None else float(score)

            
                text = f"{status}: {display_id} | sim={score_for_text:.2f}"
                cv2.putText(frame, text, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


                # Track association
                best_tid = None
                best_iou_score = 0.0
                for tid, t in tracks.items():
                    s = iou(bbox, t["bbox"])
                    if s > best_iou_score:
                        best_iou_score = s
                        best_tid = tid

                if best_tid is None or best_iou_score < match_iou_threshold:
                    tid = next_track_id
                    next_track_id += 1
                    tracks[tid] = {
                        "bbox": bbox,
                        "last_seen": now,
                        "unknown_start": None,
                        "unauth_start": None,
                        "last_confirm_unknown": 0.0,
                        "last_confirm_unauth": 0.0,
                    }
                else:
                    tid = best_tid
                    tracks[tid]["bbox"] = bbox
                    tracks[tid]["last_seen"] = now

                # Unknown confirmation
                if is_unknown:
                    if tracks[tid]["unknown_start"] is None:
                        tracks[tid]["unknown_start"] = now

                    unknown_duration = now - tracks[tid]["unknown_start"]
                    in_cd = (now - tracks[tid]["last_confirm_unknown"]) < cooldown_seconds

                    if (unknown_duration >= confirm_seconds) and (not in_cd):
                        tracks[tid]["last_confirm_unknown"] = now
                        score_for_event = -1.0 if score is None else float(score)
                        snap = save_unknown_snapshot(frame.copy(), cam_source, score_for_event)

                        # Save a face crop for later verification
                        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        def safe_filename(s: str) -> str:
                            return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in str(s))
                        
                        cam_tag = safe_filename(cam_source) if isinstance(cam_source, str) else str(cam_source)
                        zone_tag = safe_filename(zone_name)

                        crop_name = f"{ts}_zone{zone_tag}_cam{cam_tag}_tid{tid}.jpg"
                        crop_path = save_face_crop(frame, bbox, out_dir="pending_verification", filename=crop_name)

                        # Log a separate "verification required" event (queue time)
                        ver_event = {
                            "type": "verification_required",
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "zone": zone_name,
                            "cam": cam_source,
                            "track_id": tid,
                            "det_score": round(det, 4),
                            "bbox": bbox,
                            "crop": (crop_path.replace("\\", "/") if crop_path else None),
                            "snapshot": snap.replace("\\", "/"),
                        }
                        log_event(ver_event)


                        event = {
                            "type": "unknown_confirmed",
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "zone": zone_name,
                            "cam": cam_source,
                            "track_id": tid,
                            "similarity": round(score_for_event, 4),
                            "det_score": round(det, 4),
                            "bbox": bbox,
                            "snapshot": snap.replace("\\", "/"),
                        }
                        log_event(event)
                        if alarm_policy.get("unknown_confirmed", "sound") == "sound":
                            sound_path = alarm_sounds.get("unknown_confirmed")
                            if sound_path:
                                play_alarm_sound(sound_path)
                            alarm_until = now + 3.0
                            alarm_text = "ALARM: UNKNOWN PERSON IN RESTRICTED AREA"

                    # reset unauth timer
                    tracks[tid]["unauth_start"] = None

                # Unauthorized confirmation
                elif is_unauthorized:
                    if tracks[tid]["unauth_start"] is None:
                        tracks[tid]["unauth_start"] = now

                    unauth_duration = now - tracks[tid]["unauth_start"]
                    in_cd = (now - tracks[tid]["last_confirm_unauth"]) < cooldown_seconds

                    if (unauth_duration >= confirm_seconds) and (not in_cd):
                        tracks[tid]["last_confirm_unauth"] = now
                        score_for_event = -1.0 if score is None else float(score)
                        snap = save_unauth_snapshot(frame.copy(), cam_source, employee_id, score_for_event)

                        event = {
                            "type": "unauthorized_confirmed",
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "zone": zone_name,
                            "cam": cam_source,
                            "track_id": tid,
                            "person_id": employee_id,
                            "similarity": round(score_for_event, 4),
                            "det_score": round(det, 4),
                            "bbox": bbox,
                            "snapshot": snap.replace("\\", "/"),
                        }
                        log_event(event)
                        if alarm_policy.get("unauthorized_confirmed", "sound") == "sound":
                            sound_path = alarm_sounds.get("unauthorized_confirmed")
                            if sound_path:
                                play_alarm_sound(sound_path)
                            alarm_until = now + 3.0
                            alarm_text = f"ALARM: UNAUTHORIZED ({employee_id}) IN RESTRICTED AREA"
                    
                    # reset unknown timer
                    tracks[tid]["unknown_start"] = None

                else:
                    # authorized (known and in allowlist)
                    tracks[tid]["unknown_start"] = None
                    tracks[tid]["unauth_start"] = None

            # Cleanup tracks
            for tid in list(tracks.keys()):
                if (now - tracks[tid]["last_seen"]) > track_ttl_seconds:
                    del tracks[tid]

            if now < alarm_until:
                h, w = frame.shape[:2]
                overlay = frame.copy()

                cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 255), -1)
                cv2.putText(
                    overlay,
                    alarm_text,
                    (20, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (255, 255, 255),
                    3,
                )
                alpha = 0.85
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

            display = letterbox_to(frame, 1920, 1080)
            cv2.imshow("Recognize", display)
            if cv2.waitKey(1) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                return
        cap.release()
        time.sleep(reconnect_delay)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run")
    p_run.add_argument("--zone_config", default="zones.json")
    p_run.add_argument("--zone_name", required=True)
    p_run.add_argument("--device", choices=["cpu", "gpu"], default="cpu")

    args = parser.parse_args()

    zone_cfg = load_zone_config(args.zone_config, args.zone_name)
    cam_source = zone_cfg.get("cam_source", 0)
    if isinstance(cam_source, str) and cam_source.strip().isdigit():
        cam_source = int(cam_source.strip())

    recognize_with_zone(cam_source, args.device, zone_cfg)


if __name__ == "__main__":
    main()
