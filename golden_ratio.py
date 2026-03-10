import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from config import (
    TOLERANCE_EYE, TOLERANCE_NOSE_CHIN,
    FACE_L, EYE_L_OUT, EYE_L_IN, EYE_R_IN, EYE_R_OUT, FACE_R,
    NOSE_BASE, MOUTH_TOP, MOUTH_BOT, CHIN, NOSE_MOUTH_TARGET,
)

# 트랙바 기본값 (config에서 로드)
DEFAULT_TOLERANCE_EYE       = TOLERANCE_EYE
DEFAULT_TOLERANCE_NOSE_CHIN = TOLERANCE_NOSE_CHIN


def _pt(landmarks: list[NormalizedLandmark], idx: int, frame_w: int, frame_h: int) -> tuple[float, float]:
    lm = landmarks[idx]
    return lm.x * frame_w, lm.y * frame_h


def _project(p: tuple[float, float], origin: tuple[float, float], unit: tuple[float, float]) -> float:
    """점 p를 origin 기준 unit 방향 직선에 정사영한 거리."""
    return (p[0] - origin[0]) * unit[0] + (p[1] - origin[1]) * unit[1]


def is_golden_ratio(
    landmarks: list[NormalizedLandmark],
    frame_w: int,
    frame_h: int,
    tol_eye: float = DEFAULT_TOLERANCE_EYE,
    tol_nose_chin: float = DEFAULT_TOLERANCE_NOSE_CHIN,
) -> tuple[bool, dict[str, float]]:
    """
    눈 수평 5등분 비율 검사 (face_l → face_r 직선 정사영 기반).
    얼굴(왼) : 왼눈 : 눈사이 : 오른눈 : 얼굴(오) = 1:1:1:1:1

    Returns: (is_golden, segments_ratio)
      segments_ratio 값은 각 구간 / 평균 → 1.0이 이상적
    """
    face_l    = _pt(landmarks, FACE_L,    frame_w, frame_h)
    eye_l_out = _pt(landmarks, EYE_L_OUT, frame_w, frame_h)
    eye_l_in  = _pt(landmarks, EYE_L_IN,  frame_w, frame_h)
    eye_r_in  = _pt(landmarks, EYE_R_IN,  frame_w, frame_h)
    eye_r_out = _pt(landmarks, EYE_R_OUT, frame_w, frame_h)
    face_r    = _pt(landmarks, FACE_R,    frame_w, frame_h)

    # face_l → face_r 방향 단위벡터
    dx, dy = face_r[0] - face_l[0], face_r[1] - face_l[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return False, {}
    unit: tuple[float, float] = (dx / length, dy / length)

    # 각 포인트를 직선에 정사영 → 1D 위치값
    proj = [_project(p, face_l, unit) for p in
            [face_l, eye_l_out, eye_l_in, eye_r_in, eye_r_out, face_r]]

    d1, d2, d3, d4, d5 = [proj[i+1] - proj[i] for i in range(5)]
    segments: list[float] = [d1, d2, d3, d4, d5]
    mean: float = sum(segments) / len(segments)

    if mean <= 0:
        return False, {}

    # 각 구간을 평균으로 나눔 → 1.0이 이상적
    ratios: dict[str, float] = {
        "face_L → eye_L": d1 / mean,
        "left eye":        d2 / mean,
        "between eyes":    d3 / mean,
        "right eye":       d4 / mean,
        "eye_R → face_R":  d5 / mean,
    }

    check1: bool = all(abs(r - 1.0) <= tol_eye for r in ratios.values())

    # ── 검사 2: 코밑 → 입중앙 → 턱끝 수직 비율 (0.33 : 0.67) ──────
    nose_base = _pt(landmarks, NOSE_BASE, frame_w, frame_h)
    mouth_mid = (
        (landmarks[MOUTH_TOP].x + landmarks[MOUTH_BOT].x) / 2 * frame_w,
        (landmarks[MOUTH_TOP].y + landmarks[MOUTH_BOT].y) / 2 * frame_h,
    )
    chin = _pt(landmarks, CHIN, frame_w, frame_h)

    # 코밑→턱끝 방향 단위벡터에 정사영
    vx, vy = chin[0] - nose_base[0], chin[1] - nose_base[1]
    vlen = math.sqrt(vx * vx + vy * vy)
    if vlen > 0:
        vunit: tuple[float, float] = (vx / vlen, vy / vlen)
        t_mouth = _project(mouth_mid, nose_base, vunit)
        t_chin  = _project(chin,      nose_base, vunit)
        if t_chin > 0:
            r_nose_mouth = t_mouth / t_chin          # 목표: 0.333
            ratios["nose→mouth"] = r_nose_mouth
            ratios["mouth→chin"] = 1.0 - r_nose_mouth  # 목표: 0.667
            check2 = abs(r_nose_mouth - NOSE_MOUTH_TARGET) <= tol_nose_chin * NOSE_MOUTH_TARGET
        else:
            check2 = False
    else:
        check2 = False

    is_golden: bool = check1 and check2
    return is_golden, ratios


def create_face_landmarker() -> vision.FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task")
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=10,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)
