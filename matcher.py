import numpy as np

# Conservative but usable for office testing
HIGH_THRESHOLD = 0.55
MARGIN = 0.10

MIN_FACE_HEIGHT = 90
MIN_DET_SCORE = 0.60


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def cosine_sim(a, b) -> float:
    a = _norm(a)
    b = _norm(b)
    return float(np.dot(a, b))


def no_guess_match(
    emb: np.ndarray,
    employees: dict,
    context: str,
    det_score: float,
    face_h: int,
):
    # Quality gates (don’t even try if garbage)
    if det_score is not None and det_score < MIN_DET_SCORE:
        return None, None
    if face_h is not None and face_h < MIN_FACE_HEIGHT:
        return None, None

    emb = _norm(emb)

    # Compute BEST similarity per employee_id (not per vector!)
    best_per_emp = {}

    for emp_id, emp in employees.items():
        buckets = (emp.get("embeddings") or {})
        vectors = buckets.get(context, []) or []

        best = None
        for item in vectors:
            vec = item.get("vector")
            if not vec:
                continue
            v = _norm(np.array(vec, dtype=np.float32))
            s = float(np.dot(emb, v))
            if (best is None) or (s > best):
                best = s

        if best is not None:
            best_per_emp[emp_id] = best

    if not best_per_emp:
        return None, None

    ranked = sorted(best_per_emp.items(), key=lambda x: x[1], reverse=True)
    best_id, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else -1.0

    # Must clear absolute threshold
    if s1 < HIGH_THRESHOLD:
        return None, None

    # If there is a runner-up, enforce margin
    if len(ranked) > 1 and (s1 - s2) < MARGIN:
        return None, None

    return best_id, s1
