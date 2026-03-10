import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from config import MOSAIC_BLOCK_SIZE


def apply_mosaic(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    block_size: int = MOSAIC_BLOCK_SIZE,
) -> np.ndarray:
    """bbox=(x, y, w, h) 영역에 모자이크 적용."""
    x, y, w, h = bbox
    h_f, w_f = frame.shape[:2]

    x  = max(0, x)
    y  = max(0, y)
    x2 = min(w_f, x + w)
    y2 = min(h_f, y + h)

    if x2 <= x or y2 <= y:
        return frame

    roi = frame[y:y2, x:x2]
    roi_h, roi_w = roi.shape[:2]

    if roi_h < block_size or roi_w < block_size:
        return frame

    small  = cv2.resize(roi, (roi_w // block_size, roi_h // block_size),
                        interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
    frame[y:y2, x:x2] = mosaic
    return frame


def create_face_detector() -> vision.FaceDetector:
    base_options = python.BaseOptions(model_asset_path="models/blaze_face_short_range.tflite")
    options = vision.FaceDetectorOptions(base_options=base_options)
    return vision.FaceDetector.create_from_options(options)
