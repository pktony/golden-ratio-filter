# 황금 비율 얼굴 정의

황금 비율 대신 **균등 분할 비율**을 기반으로 얼굴의 2가지 비율을 측정하여 판별합니다.

---

## 측정 항목

### 검사 1: 눈 수평 5등분 비율 (Eye Horizontal Quintile)

```
얼굴 왼쪽 경계 → 왼눈 외측 → 왼눈 내측 → 오른눈 내측 → 오른눈 외측 → 얼굴 오른쪽 경계
5개 구간의 길이가 모두 균등 = 1 : 1 : 1 : 1 : 1
```

- `face_L → face_R` 방향 단위벡터에 각 포인트를 **정사영(projection)** 하여 1D 위치로 변환
- 5개 구간을 평균으로 나눈 비율이 각각 `1.0 ± TOLERANCE_EYE(10%)` 이내이면 통과

### 검사 2: 코밑-입-턱 수직 비율 (Nose-Mouth-Chin Vertical Ratio)

```
코밑 → 입 중앙 : 코밑 → 턱끝 = 1 : 3  (= 0.333)
```

- `nose_base → chin` 방향 단위벡터에 입 중앙점을 **정사영** 하여 분할 비율 계산
- `(코밑→입) / (코밑→턱끝) ≈ 0.333 ± TOLERANCE_NOSE_CHIN(10%) × 0.333` 이내이면 통과
- 입 중앙 = `(MOUTH_TOP + MOUTH_BOT) / 2`

---

## MediaPipe Face Mesh 랜드마크 포인트

MediaPipe Face Mesh는 **478개** 랜드마크를 제공합니다.
사용 포인트는 다음과 같습니다.

| 상수명 | 인덱스 | 설명 |
|--------|--------|------|
| `FACE_L` | `#127` | 얼굴 왼쪽 경계 |
| `EYE_L_OUT` | `#130` | 왼눈 외측 끝 |
| `EYE_L_IN` | `#133` | 왼눈 내측 끝 |
| `EYE_R_IN` | `#362` | 오른눈 내측 끝 |
| `EYE_R_OUT` | `#446` | 오른눈 외측 끝 |
| `FACE_R` | `#356` | 얼굴 오른쪽 경계 |
| `NOSE_BASE` | `#2` | 코 밑 |
| `MOUTH_TOP` | `#13` | 윗입술 내측 |
| `MOUTH_BOT` | `#14` | 아랫입술 내측 |
| `CHIN` | `#152` | 턱끝 |

---

## 판별 기준

```
허용 오차 (눈 5등분)     TOLERANCE_EYE       = ±10%  (config.py)
허용 오차 (코밑-입-턱)   TOLERANCE_NOSE_CHIN = ±10%  (config.py)

코밑→입 목표 비율        NOSE_MOUTH_TARGET   = 1/3 ≈ 0.333
```

- **검사 1** + **검사 2** 모두 통과 → 황금 비율 얼굴 ✅ → 초록 bbox
- 하나라도 실패 → 비황금 비율 ❌ → 빨간 bbox + 모자이크 처리

---

## 계산 흐름 요약

```
입력: MediaPipe FaceLandmarker 결과 (NormalizedLandmark 목록)

[검사 1] 눈 수평 5등분
  픽셀 좌표 변환 (_pt)
  → face_L~face_R 단위벡터 산출
  → 6개 포인트 정사영 (_project)
  → 5개 구간 길이 계산
  → 각 구간 / 평균 비율이 [0.9, 1.1] 이내?

[검사 2] 코밑-입-턱 수직 비율
  픽셀 좌표 변환 (_pt, 입 중앙 계산)
  → nose_base→chin 단위벡터 산출
  → 입 중앙 정사영 (_project)
  → (코밑→입) / (코밑→턱끝) 이 [0.300, 0.367] 이내?

출력: (is_golden: bool, ratios: dict[str, float])
```

---

## 참고

- [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [MediaPipe Face Mesh 랜드마크 맵](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models)
