import cv2
import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from config import MOSAIC_BLOCK_SIZE
from draw_bbox import face_bbox


def apply_mosaic(
    frame: np.ndarray,
    landmarks: list[NormalizedLandmark],
    frame_w: int,
    frame_h: int,
    block_size: int = MOSAIC_BLOCK_SIZE,
) -> None:
    x1, y1, x2, y2 = face_bbox(landmarks, frame_w, frame_h)
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return
    small = cv2.resize(roi,
                       (max(1, w // block_size), max(1, h // block_size)),
                       interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
