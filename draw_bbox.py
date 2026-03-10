import cv2
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

LABEL_MAP: dict[str, str] = {
    "face_length_width": "face H/W",
    "upper_lower_face":  "upper/lower",
    "eye_nose_mouth":    "eye-nose/nose-mouth",
}


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
    ratios: dict[str, float],
) -> None:
    x1, y1, x2, y2 = face_bbox(landmarks, frame_w, frame_h)
    color: tuple[int, int, int] = (0, 255, 0) if golden else (0, 0, 255)
    label: str = "Golden Ratio" if golden else "Not Golden"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    for i, (key, val) in enumerate(ratios.items()):
        text: str = f"{LABEL_MAP.get(key, key)}: {val:.3f}"
        cv2.putText(frame, text, (x1, y2 + 28 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.65,
                    color=color,
                    thickness=2)
