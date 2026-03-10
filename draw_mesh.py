import cv2
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python.vision import FaceLandmarksConnections


def draw_mesh(frame, landmarks):
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
        connections=FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
    )


# 황금 비율 계산에 사용하는 주요 랜드마크
KEY_LANDMARKS = {
    10:  "forehead",
    152: "chin",
    4:   "nose_tip",
    234: "cheek_L",
    454: "cheek_R",
    168: "nose_bridge",
    0:   "upper_lip",
}


def draw_landmark_indices(frame, landmarks, frame_w, frame_h, indices=None):
    """
    indices: 표시할 인덱스 목록. None이면 KEY_LANDMARKS 전체 표시.
    전체 468개를 보려면 indices=range(468) 전달.
    """
    targets = indices if indices is not None else KEY_LANDMARKS.keys()

    for idx in targets:
        lm = landmarks[idx]
        x, y = int(lm.x * frame_w), int(lm.y * frame_h)
        label = f"{idx}" if indices is not None else f"{idx}:{KEY_LANDMARKS[idx]}"
        cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
        cv2.putText(frame, label, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
