import cv2
import math
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from config import (
    FACE_L, EYE_L_OUT, EYE_L_IN, EYE_R_IN, EYE_R_OUT, FACE_R,
    NOSE_BASE, MOUTH_TOP, MOUTH_BOT, CHIN,
    COLOR_EYE_SEG, COLOR_NOSE_MOUTH, COLOR_MOUTH_CHIN, COLOR_KEYPOINT,
)


def _projected_pts(
    landmarks: list[NormalizedLandmark],
    frame_w: int,
    frame_h: int,
) -> list[tuple[int, int]]:
    """6개 포인트를 face_l → face_r 방향 직선에 정사영.
    직선은 눈 4개 코너의 중심을 지나도록 위치."""
    indices = [FACE_L, EYE_L_OUT, EYE_L_IN, EYE_R_IN, EYE_R_OUT, FACE_R]
    pts = [(lm.x * frame_w, lm.y * frame_h) for lm in [landmarks[i] for i in indices]]

    dx, dy = pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return [(int(p[0]), int(p[1])) for p in pts]
    unit = (dx / length, dy / length)

    eye_indices = [1, 2, 3, 4]  # pts 내 인덱스: EYE_L_OUT, EYE_L_IN, EYE_R_IN, EYE_R_OUT
    origin = (
        sum(pts[i][0] for i in eye_indices) / 4,
        sum(pts[i][1] for i in eye_indices) / 4,
    )

    result: list[tuple[int, int]] = []
    for p in pts:
        t = (p[0] - origin[0]) * unit[0] + (p[1] - origin[1]) * unit[1]
        result.append((int(origin[0] + t * unit[0]), int(origin[1] + t * unit[1])))
    return result


def draw_eye_ratio_lines(
    frame: np.ndarray,
    landmarks: list[NormalizedLandmark],
    frame_w: int,
    frame_h: int,
    ratios: dict[str, float],
) -> None:
    """face_l → face_r 직선에 정사영한 눈 수평 5개 구간 표시."""
    if not ratios:
        return

    proj = _projected_pts(landmarks, frame_w, frame_h)
    keys = list(ratios.keys())

    for i in range(5):
        color = COLOR_EYE_SEG[i]
        p1, p2 = proj[i], proj[i + 1]

        cv2.line(frame, p1, p2, color, 2)
        cv2.circle(frame, p1, 4, color, -1)
        cv2.circle(frame, p2, 4, color, -1)

        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        ratio_val = ratios.get(keys[i], 0.0)
        cv2.putText(frame, f"{ratio_val:.2f}", (mid[0] - 16, mid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)


def draw_nose_chin_lines(
    frame: np.ndarray,
    landmarks: list[NormalizedLandmark],
    frame_w: int,
    frame_h: int,
    ratios: dict[str, float],
) -> None:
    """코밑→입중앙→턱끝 수직 비율선 표시."""
    if "nose→mouth" not in ratios:
        return

    nose  = (int(landmarks[NOSE_BASE].x * frame_w), int(landmarks[NOSE_BASE].y * frame_h))
    mouth = (
        int((landmarks[MOUTH_TOP].x + landmarks[MOUTH_BOT].x) / 2 * frame_w),
        int((landmarks[MOUTH_TOP].y + landmarks[MOUTH_BOT].y) / 2 * frame_h),
    )
    chin  = (int(landmarks[CHIN].x * frame_w), int(landmarks[CHIN].y * frame_h))

    cv2.line(frame, nose,  mouth, COLOR_NOSE_MOUTH, 2)
    cv2.line(frame, mouth, chin,  COLOR_MOUTH_CHIN, 2)
    for pt in (nose, mouth, chin):
        cv2.circle(frame, pt, 4, COLOR_KEYPOINT, -1)

    cv2.putText(frame, f"{ratios['nose→mouth']:.3f} (0.333)", (nose[0] + 6, (nose[1] + mouth[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_NOSE_MOUTH, 1)
    cv2.putText(frame, f"{ratios['mouth→chin']:.3f} (0.667)", (mouth[0] + 6, (mouth[1] + chin[1]) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_MOUTH_CHIN, 1)
