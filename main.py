import cv2
import mediapipe as mp
from golden_ratio import create_face_landmarker, is_golden_ratio
from draw_mesh import draw_mesh, draw_landmark_indices
from draw_bbox import draw_bbox


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    landmarker = create_face_landmarker()
    print("황금 비율 필터 실행 중... (q 키로 종료)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect(mp_image)

        golden_count = 0
        if result.face_landmarks:
            for face_lms in result.face_landmarks:
                draw_mesh(frame, face_lms)
                draw_landmark_indices(frame, face_lms, frame_w, frame_h)
                golden, ratios = is_golden_ratio(face_lms, frame_w, frame_h)
                draw_bbox(frame, face_lms, frame_w, frame_h, golden, ratios)
                if golden:
                    golden_count += 1

        total = len(result.face_landmarks) if result.face_landmarks else 0
        cv2.putText(frame, f"Faces: {total} | Golden: {golden_count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Golden Ratio Filter", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
