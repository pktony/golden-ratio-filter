GOLDEN_RATIO: float = 1.618
TOLERANCE: float = 0.05  # ±5% 허용 오차

# 사용할 비율 측정 항목 (True = 활성화)
RATIO_CHECKS: dict[str, bool] = {
    "face_length_width": True,    # 얼굴 길이 / 너비
    "upper_lower_face": True,     # 이마~코 / 코~턱
    "eye_nose_mouth": True,       # 눈~코 / 코~입
}

# 모자이크 블록 크기
MOSAIC_BLOCK_SIZE: int = 15
