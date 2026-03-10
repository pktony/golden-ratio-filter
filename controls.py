import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from golden_ratio import DEFAULT_TOLERANCE_EYE, DEFAULT_TOLERANCE_NOSE_CHIN

WIN      = "Controls"
BTN_W    = 44
BTN_H    = 44
PADDING  = 10
LABEL_X  = 76
PANEL_W  = 460
SLIDER_W = 220
SLIDER_H = 24
SLIDER_X = 126
SLIDER_Y1 = 90
SLIDER_GAP = 74
TOP_H = 224
COLOR_ON  = (60, 200, 80)
COLOR_OFF = (70, 70, 70)
COLOR_BG  = (24, 24, 26)
COLOR_CARD = (34, 35, 39)
COLOR_CARD_BORDER = (58, 60, 66)
COLOR_MUTED = (165, 168, 176)
COLOR_ACCENT = (255, 186, 84)
COLOR_TRACK = (88, 91, 98)
COLOR_FILL = (72, 184, 255)
COLOR_TXT = (255, 255, 255)
TITLE_LABEL = "상처받을 용기"
EYE_TOL_LABEL = "눈 거리"
NOSE_CHIN_LABEL = "코-턱"
KOREAN_FONT_PATH = Path("/System/Library/Fonts/AppleSDGothicNeo.ttc")


@dataclass
class ControlValues:
    tol_eye:       float
    tol_nose_chin: float
    show_mesh:     bool
    show_lines:    bool
    show_indices:  bool
    show_bbox:     bool
    show_mosaic:   bool


class ControlPanel:
    def __init__(
        self,
        toggle_labels:   list[str],
        toggle_defaults: list[bool],
        win_name: str = WIN,
    ) -> None:
        self._labels = toggle_labels
        self._states = list(toggle_defaults)
        self._win    = win_name
        self._font_ko = self._load_font(24)
        self._font_ko_small = self._load_font(17)
        self._font_ko_meta = self._load_font(14)
        self._tol_eye = DEFAULT_TOLERANCE_EYE
        self._tol_nose_chin = DEFAULT_TOLERANCE_NOSE_CHIN
        self._dragging_slider: str | None = None

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        if KOREAN_FONT_PATH.exists():
            return ImageFont.truetype(str(KOREAN_FONT_PATH), size)
        return ImageFont.load_default()

    def create(self) -> None:
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self._win, self._on_click)
        panel_h = TOP_H + len(self._labels) * (BTN_H + PADDING) + PADDING
        cv2.resizeWindow(self._win, PANEL_W, panel_h)
        self._render()

    def _render(self) -> None:
        h   = TOP_H + len(self._labels) * (BTN_H + PADDING) + PADDING
        img = np.full((h, PANEL_W, 3), COLOR_BG, dtype=np.uint8)

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        self._draw_top_panel(draw)
        self._draw_slider_section(draw, TITLE_LABEL, EYE_TOL_LABEL, self._tol_eye, SLIDER_Y1)
        self._draw_slider_section(draw, None, NOSE_CHIN_LABEL, self._tol_nose_chin, SLIDER_Y1 + SLIDER_GAP)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        self._draw_slider_graphics(img, self._tol_eye, SLIDER_Y1)
        self._draw_slider_graphics(img, self._tol_nose_chin, SLIDER_Y1 + SLIDER_GAP)
        cv2.line(img, (PADDING + 6, TOP_H - 14), (PANEL_W - PADDING - 6, TOP_H - 14), COLOR_CARD_BORDER, 1)
        for i, label in enumerate(self._labels):
            y1    = TOP_H + i * (BTN_H + PADDING)
            y2    = y1 + BTN_H
            color = COLOR_ON if self._states[i] else COLOR_OFF
            cv2.rectangle(img, (PADDING, y1), (PADDING + BTN_W, y2), color, -1)
            cv2.rectangle(img, (PADDING, y1), (PADDING + BTN_W, y2), (100, 100, 100), 1)
            cv2.putText(img, label, (LABEL_X, y1 + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TXT, 2)
        cv2.imshow(self._win, img)

    def _draw_top_panel(self, draw: ImageDraw.ImageDraw) -> None:
        draw.rounded_rectangle(
            (PADDING, PADDING, PANEL_W - PADDING, TOP_H - 28),
            radius=20,
            fill=COLOR_CARD,
            outline=COLOR_CARD_BORDER,
            width=1,
        )

    def _draw_slider_section(
        self,
        draw: ImageDraw.ImageDraw,
        section_title: str | None,
        slider_label: str,
        value: float,
        y: int,
    ) -> None:
        if section_title is not None:
            draw.text((PADDING + 12, 20), section_title, font=self._font_ko, fill=COLOR_TXT)

        section_left = PADDING + 22
        section_right = PANEL_W - PADDING - 22
        label_y = y - 34
        draw.text((section_left, label_y), slider_label, font=self._font_ko_small, fill=COLOR_TXT)

        value_text = f"{int(value * 100)}%"
        value_box = (section_right - 56, label_y - 2, section_right, label_y + 22)
        draw.rounded_rectangle(value_box, radius=12, fill=(59, 48, 26))
        value_bbox = draw.textbbox((0, 0), value_text, font=self._font_ko_meta)
        value_w = value_bbox[2] - value_bbox[0]
        value_h = value_bbox[3] - value_bbox[1]
        value_x = value_box[0] + ((value_box[2] - value_box[0] - value_w) / 2)
        value_y = value_box[1] + ((value_box[3] - value_box[1] - value_h) / 2) - 1
        draw.text((value_x, value_y), value_text, font=self._font_ko_meta, fill=COLOR_ACCENT)

        left_text = "미켈란젤로"
        right_text = "인간"
        left_bbox = draw.textbbox((0, 0), left_text, font=self._font_ko_meta)
        right_bbox = draw.textbbox((0, 0), right_text, font=self._font_ko_meta)
        left_w = left_bbox[2] - left_bbox[0]
        right_w = right_bbox[2] - right_bbox[0]
        scale_y = y + 14
        draw.text((SLIDER_X - (left_w / 2), scale_y), left_text, font=self._font_ko_meta, fill=COLOR_MUTED)
        draw.text((SLIDER_X + SLIDER_W - (right_w / 2), scale_y), right_text,
                  font=self._font_ko_meta, fill=COLOR_MUTED)

    def _draw_slider_graphics(self, img: np.ndarray, value: float, y: int) -> None:
        y_mid = y
        cv2.line(img, (SLIDER_X, y_mid), (SLIDER_X + SLIDER_W, y_mid), COLOR_TRACK, 6, cv2.LINE_AA)
        knob_x = int(SLIDER_X + value * SLIDER_W / 0.5)
        cv2.line(img, (SLIDER_X, y_mid), (knob_x, y_mid), COLOR_FILL, 6, cv2.LINE_AA)
        cv2.circle(img, (knob_x, y_mid), 12, (242, 244, 248), -1, cv2.LINE_AA)
        cv2.circle(img, (knob_x, y_mid), 12, (118, 122, 130), 1, cv2.LINE_AA)

    def _slider_hit(self, x: int, y: int, slider_y: int) -> bool:
        return SLIDER_X - 16 <= x <= SLIDER_X + SLIDER_W + 16 and slider_y - 16 <= y <= slider_y + 16

    def _update_slider(self, slider_name: str, x: int) -> None:
        value = max(0.0, min(0.5, ((x - SLIDER_X) / SLIDER_W) * 0.5))
        if slider_name == "eye":
            self._tol_eye = value
        else:
            self._tol_nose_chin = value

    def _on_click(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self._slider_hit(x, y, SLIDER_Y1):
                self._dragging_slider = "eye"
                self._update_slider("eye", x)
                self._render()
                return
            if self._slider_hit(x, y, SLIDER_Y1 + SLIDER_GAP):
                self._dragging_slider = "nose_chin"
                self._update_slider("nose_chin", x)
                self._render()
                return
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging_slider is not None:
            self._update_slider(self._dragging_slider, x)
            self._render()
            return
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging_slider = None
            return
        else:
            return

        if y < TOP_H:
            return
        i = (y - TOP_H) // (BTN_H + PADDING)
        if 0 <= i < len(self._labels):
            self._states[i] = not self._states[i]
            self._render()

    def read(self) -> ControlValues:
        return ControlValues(
            tol_eye       = self._tol_eye,
            tol_nose_chin = self._tol_nose_chin,
            show_mesh     = self._states[0],
            show_lines    = self._states[1],
            show_indices  = self._states[2],
            show_bbox     = self._states[3],
            show_mosaic   = self._states[4],
        )
