import cv2
import numpy as np
from collections.abc import Iterable
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python.vision import FaceLandmarksConnections
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from config import COLOR_LANDMARK, KEY_LANDMARKS


def draw_mesh(frame: np.ndarray, landmarks: list[NormalizedLandmark]) -> None:
    drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
    )
    drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=landmarks,
        connections=FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
    )




def draw_landmark_indices(
    frame: np.ndarray,
    landmarks: list[NormalizedLandmark],
    frame_w: int,
    frame_h: int,
    indices: Iterable[int] | None = None,
) -> None:
    """
    indices: 표시할 인덱스 목록. None이면 KEY_LANDMARKS 전체 표시.
    전체 468개를 보려면 indices=range(468) 전달.
    """
    targets: Iterable[int] = indices if indices is not None else KEY_LANDMARKS.keys()

    for idx in targets:
        lm = landmarks[idx]
        x, y = int(lm.x * frame_w), int(lm.y * frame_h)
        label: str = f"{idx}" if indices is not None else f"{idx}:{KEY_LANDMARKS[idx]}"
        cv2.circle(frame, (x, y), 3, COLOR_LANDMARK, -1)
        cv2.putText(frame, label, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_LANDMARK, 1)
