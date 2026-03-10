import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarkerResult
from mediapipe.tasks.python.vision import FaceLandmarker
from golden_ratio import create_face_landmarker, is_golden_ratio
from draw_mesh import draw_mesh, draw_landmark_indices
from draw_bbox import draw_bbox
from mosaic import apply_mosaic
from draw_lines import draw_eye_ratio_lines, draw_nose_chin_lines
from controls import ControlPanel, ControlValues


def main() -> None:
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    landmarker: FaceLandmarker = create_face_landmarker()
    ctrl_panel = ControlPanel(
        toggle_labels   = ["Mesh", "Guidelines", "Indices", "BBox", "Mosaic"],
        toggle_defaults = [True,   True,         True,      True,   False],
    )
    ctrl_panel.create()
    print("황금 비율 필터 실행 중... (q 키로 종료)")

    while True:
        ret: bool
        frame: np.ndarray
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        frame_h: int
        frame_w: int
        frame_h, frame_w = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result: FaceLandmarkerResult = landmarker.detect(mp_image)
        ctrl: ControlValues = ctrl_panel.read()

        golden_count: int = 0
        if result.face_landmarks:
            face_lms: list[NormalizedLandmark]
            for face_lms in result.face_landmarks:
                if ctrl.show_mesh:
                    draw_mesh(frame, face_lms)
                if ctrl.show_indices:
                    draw_landmark_indices(frame, face_lms, frame_w, frame_h)
                golden: bool
                ratios: dict[str, float]
                golden, ratios = is_golden_ratio(face_lms, frame_w, frame_h,
                                                 tol_eye=ctrl.tol_eye,
                                                 tol_nose_chin=ctrl.tol_nose_chin)
                if ctrl.show_lines:
                    draw_eye_ratio_lines(frame, face_lms, frame_w, frame_h, ratios)
                    draw_nose_chin_lines(frame, face_lms, frame_w, frame_h, ratios)
                if ctrl.show_bbox:
                    draw_bbox(frame, face_lms, frame_w, frame_h, golden)
                if ctrl.show_mosaic and not golden:
                    apply_mosaic(frame, face_lms, frame_w, frame_h)
                if golden:
                    golden_count += 1

        total: int = len(result.face_landmarks) if result.face_landmarks else 0
        cv2.putText(frame, f"Faces: {total} | Golden: {golden_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"tol eye:{ctrl.tol_eye:.2f}  nose-chin:{ctrl.tol_nose_chin:.2f}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.imshow("Golden Ratio Filter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
