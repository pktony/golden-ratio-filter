import cv2
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from config import (
    COLOR_GOLDEN, COLOR_NOT_GOLDEN,
)


def face_bbox(
    landmarks: list[NormalizedLandmark],
    frame_w: int,
    frame_h: int,
    pad: int = 10,
) -> tuple[int, int, int, int]:
    xs = [int(lm.x * frame_w) for lm in landmarks]
    ys = [int(lm.y * frame_h) for lm in landmarks]
    x1 = max(0, min(xs) - pad)
    y1 = max(0, min(ys) - pad)
    x2 = min(frame_w, max(xs) + pad)
    y2 = min(frame_h, max(ys) + pad)
    return x1, y1, x2, y2


def draw_bbox(
    frame: np.ndarray,
    landmarks: list[NormalizedLandmark],
    frame_w: int,
    frame_h: int,
    golden: bool,
) -> None:
    x1, y1, x2, y2 = face_bbox(landmarks, frame_w, frame_h)
    color = COLOR_GOLDEN if golden else COLOR_NOT_GOLDEN
    label: str = "Golden Ratio" if golden else "Not Golden"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
