TOLERANCE_EYE:        float = 0.1   # 눈 수평 5등분 ±10%
TOLERANCE_NOSE_CHIN:  float = 0.1   # 코밑-입-턱 비율 ±10%

# 모자이크 블록 크기
MOSAIC_BLOCK_SIZE: int = 30

# ── 검사 1: 눈 수평 5등분 (왼쪽 → 오른쪽) ──────────────────────────
FACE_L:    int = 127  # 얼굴 왼쪽 경계
EYE_L_OUT: int = 130  # 왼눈 외측
EYE_L_IN:  int = 133  # 왼눈 내측
EYE_R_IN:  int = 362  # 오른눈 내측
EYE_R_OUT: int = 446  # 오른눈 외측
FACE_R:    int = 356  # 얼굴 오른쪽 경계

# ── 검사 2: 코밑-입-턱 수직 비율 ────────────────────────────────────
NOSE_BASE: int = 2    # 코 밑
MOUTH_TOP: int = 13   # 윗입술 내측
MOUTH_BOT: int = 14   # 아랫입술 내측
CHIN:      int = 152  # 턱끝

NOSE_MOUTH_TARGET: float = 1 / 3  # 코밑→입 / 전체 = 0.333

# ── 색상 (BGR) ──────────────────────────────────────────────────────
Color = tuple[int, int, int]

# 눈 수평 5등분 구간
COLOR_EYE_SEG: list[Color] = [
    (255, 200,   0),  # 얼굴왼 → 왼눈
    (0,   200, 255),  # 왼눈 너비
    (0,   255, 100),  # 눈 사이
    (0,   200, 255),  # 오른눈 너비
    (255, 200,   0),  # 오른눈 → 얼굴오른
]

# 코밑-입-턱 수직 구간
COLOR_NOSE_MOUTH: Color = (180, 105, 255)  # 코→입
COLOR_MOUTH_CHIN: Color = (255, 105, 180)  # 입→턱
COLOR_KEYPOINT:   Color = (255, 255, 255)  # 키포인트 점

# bbox / 상태 라벨
COLOR_GOLDEN:     Color = (0, 255,   0)   # 황금 비율
COLOR_NOT_GOLDEN: Color = (0,   0, 255)   # 비황금

# 랜드마크 인덱스 표시
COLOR_LANDMARK:   Color = (0, 255, 255)   # 노란-초록

# ── 주요 랜드마크 레이블 ────────────────────────────────────────────
KEY_LANDMARKS: dict[int, str] = {
    FACE_L:    "face_L",
    EYE_L_OUT: "eye_L_out",
    EYE_L_IN:  "eye_L_in",
    EYE_R_IN:  "eye_R_in",
    EYE_R_OUT: "eye_R_out",
    FACE_R:    "face_R",
    NOSE_BASE: "nose_base",
    MOUTH_TOP: "mouth_top",
    MOUTH_BOT: "mouth_bot",
    CHIN:      "chin",
}
