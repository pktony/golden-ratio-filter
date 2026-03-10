import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import numpy as np
from config import GOLDEN_RATIO, TOLERANCE, RATIO_CHECKS


def euclidean_distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))


def _pt(landmarks: list[NormalizedLandmark], idx: int, frame_w: int, frame_h: int) -> tuple[int, int]:
    lm = landmarks[idx]
    return int(lm.x * frame_w), int(lm.y * frame_h)


def is_golden_ratio(
    landmarks: list[NormalizedLandmark],
    frame_w: int,
    frame_h: int,
) -> tuple[bool, dict[str, float]]:
    """
    Tasks API NormalizedLandmark 리스트로 황금 비율 여부 판별.
    Returns: (is_golden, ratios)
    """
    forehead    = _pt(landmarks, 10,  frame_w, frame_h)
    chin        = _pt(landmarks, 152, frame_w, frame_h)
    nose_tip    = _pt(landmarks, 4,   frame_w, frame_h)
    cheek_left  = _pt(landmarks, 234, frame_w, frame_h)
    cheek_right = _pt(landmarks, 454, frame_w, frame_h)
    nose_bridge = _pt(landmarks, 168, frame_w, frame_h)
    upper_lip   = _pt(landmarks, 0,   frame_w, frame_h)

    face_length   = euclidean_distance(forehead, chin)
    face_width    = euclidean_distance(cheek_left, cheek_right)
    upper_face    = euclidean_distance(forehead, nose_tip)
    lower_face    = euclidean_distance(nose_tip, chin)
    eye_to_nose   = euclidean_distance(nose_bridge, nose_tip)
    nose_to_mouth = euclidean_distance(nose_tip, upper_lip)

    ratios: dict[str, float] = {}
    checks: list[bool] = []

    if RATIO_CHECKS.get("face_length_width") and face_width > 0:
        r = face_length / face_width
        ratios["face_length_width"] = r
        checks.append(abs(r - GOLDEN_RATIO) <= GOLDEN_RATIO * TOLERANCE)

    if RATIO_CHECKS.get("upper_lower_face") and lower_face > 0:
        r = upper_face / lower_face
        ratios["upper_lower_face"] = r
        checks.append(abs(r - GOLDEN_RATIO) <= GOLDEN_RATIO * TOLERANCE)

    if RATIO_CHECKS.get("eye_nose_mouth") and nose_to_mouth > 0:
        r = eye_to_nose / nose_to_mouth
        ratios["eye_nose_mouth"] = r
        checks.append(abs(r - GOLDEN_RATIO) <= GOLDEN_RATIO * TOLERANCE)

    return bool(checks) and all(checks), ratios


def create_face_landmarker() -> vision.FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=10,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)
