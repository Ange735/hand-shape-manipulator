"""
✋ Hand Shape Manipulator
========================
Manipulez des formes géométriques avec vos doigts !

Gestes :
  • PINCE (pouce + index proches)   →  attraper et déplacer une forme
  • 2 MAINS pincées + écarter       →  zoomer / rétrécir
  • 2 MAINS pincées + tourner       →  rotation
  • Q                               →  quitter
  • D                               →  désélectionner tout

Dépendances :
  pip install opencv-python mediapipe numpy
"""

import cv2
import numpy as np
import math
import os
import urllib.request

# ── Téléchargement automatique du modèle MediaPipe ───────────────────────────
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print(f"Telechargement du modele MediaPipe...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modele telecharge.")

# ── Import MediaPipe Tasks ────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

BaseOptions    = mp_python.BaseOptions
HandLandmarker = mp_vision.HandLandmarker
HLOptions      = mp_vision.HandLandmarkerOptions
RunningMode    = mp_vision.RunningMode

PINCH_THRESHOLD = 0.055  # distance normalisée pouce-index

# ── Palette ──────────────────────────────────────────────────────────────────
PALETTE = [
    ( 80, 100, 255),
    ( 60, 220, 120),
    (255,  80,  80),
    (255, 180,  40),
    (200,  60, 255),
    ( 40, 220, 255),
]

# ── Classe Shape ─────────────────────────────────────────────────────────────
class Shape:
    def __init__(self, kind, cx, cy, size, color):
        self.kind     = kind
        self.cx       = float(cx)
        self.cy       = float(cy)
        self.size     = float(size)
        self.color    = color
        self.angle    = 0.0
        self.selected = False
        self.dragging = False
        self.drag_ox  = 0.0
        self.drag_oy  = 0.0

    def _poly(self, n, r=None, ao=-90):
        r = r or self.size
        return np.array([
            [int(self.cx + r * math.cos(math.radians(self.angle + ao + i*360/n))),
             int(self.cy + r * math.sin(math.radians(self.angle + ao + i*360/n)))]
            for i in range(n)
        ], np.int32)

    def _star(self):
        pts = []
        for i in range(10):
            r = self.size if i % 2 == 0 else self.size * 0.40
            a = math.radians(self.angle - 90 + i * 36)
            pts.append([int(self.cx + r*math.cos(a)), int(self.cy + r*math.sin(a))])
        return np.array(pts, np.int32)

    def _rect(self):
        w, h  = self.size, self.size * 0.65
        rad   = math.radians(self.angle)
        ca, sa = math.cos(rad), math.sin(rad)
        return np.array([
            [int(self.cx + (-w)*ca - (-h)*sa), int(self.cy + (-w)*sa + (-h)*ca)],
            [int(self.cx + ( w)*ca - (-h)*sa), int(self.cy + ( w)*sa + (-h)*ca)],
            [int(self.cx + ( w)*ca - ( h)*sa), int(self.cy + ( w)*sa + ( h)*ca)],
            [int(self.cx + (-w)*ca - ( h)*sa), int(self.cy + (-w)*sa + ( h)*ca)],
        ], np.int32)

    def hit(self, px, py):
        return math.hypot(px - self.cx, py - self.cy) < self.size * 1.3

    def draw(self, frame):
        if self.selected:
            ov = frame.copy()
            gc = tuple(min(255, int(c * 1.4)) for c in self.color)
            cv2.circle(ov, (int(self.cx), int(self.cy)), int(self.size*1.4), gc, 10)
            cv2.addWeighted(ov, 0.3, frame, 0.7, 0, frame)

        border = (255, 255, 255) if self.selected else (20, 20, 20)
        bt     = 3 if self.selected else 1

        # Remplissage
        if self.kind == 'circle':
            cv2.circle(frame, (int(self.cx), int(self.cy)), int(self.size), self.color, -1)
            cv2.circle(frame, (int(self.cx), int(self.cy)), int(self.size), border, bt)
        elif self.kind == 'rect':
            pts = self._rect()
            cv2.fillPoly(frame, [pts], self.color)
            cv2.polylines(frame, [pts], True, border, bt)
        elif self.kind == 'triangle':
            pts = self._poly(3)
            cv2.fillPoly(frame, [pts], self.color)
            cv2.polylines(frame, [pts], True, border, bt)
        elif self.kind == 'pentagon':
            pts = self._poly(5)
            cv2.fillPoly(frame, [pts], self.color)
            cv2.polylines(frame, [pts], True, border, bt)
        elif self.kind == 'hexagon':
            pts = self._poly(6, ao=0)
            cv2.fillPoly(frame, [pts], self.color)
            cv2.polylines(frame, [pts], True, border, bt)
        elif self.kind == 'star':
            pts = self._star()
            cv2.fillPoly(frame, [pts], self.color)
            cv2.polylines(frame, [pts], True, border, bt)

        # Nom centré
        (tw, th), _ = cv2.getTextSize(self.kind, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, self.kind,
                    (int(self.cx)-tw//2, int(self.cy)+th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1, cv2.LINE_AA)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_pinch_mid(lm, W, H):
    return ((lm[4].x + lm[8].x)/2 * W, (lm[4].y + lm[8].y)/2 * H)

def check_pinch(lm):
    return math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y) < PINCH_THRESHOLD

def draw_skeleton(frame, lm, W, H):
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17),
    ]
    pts = [(int(l.x*W), int(l.y*H)) for l in lm]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (60,60,60), 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (80,80,80), -1)

def draw_hud(frame, W, H, n_sel):
    cv2.rectangle(frame, (0, H-80), (W, H), (10,10,10), -1)
    lines = [
        "PINCE (pouce+index) sur forme  ->  attraper / deplacer",
        "2 MAINS pincees               ->  zoomer  |  tourner",
        f"  {n_sel} forme(s) selectionnee(s)       Q = quitter   D = tout deselectionner",
    ]
    for j, line in enumerate(lines):
        cv2.putText(frame, line, (14, H-60+j*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160,210,255), 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    options = HLOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    detector = HandLandmarker.create_from_options(options)

    # Formes initiales
    specs = [
        ('circle',   0.18, 0.38),
        ('rect',     0.36, 0.38),
        ('triangle', 0.54, 0.38),
        ('pentagon', 0.72, 0.38),
        ('hexagon',  0.27, 0.62),
        ('star',     0.63, 0.62),
    ]
    shapes = [Shape(k, cx*W, cy*H, 72, PALETTE[i])
              for i, (k, cx, cy) in enumerate(specs)]

    prev_dist  = None
    prev_angle = None

    print("Demarrage — appuie sur Q pour quitter.")
    cv2.namedWindow("Hand Shape Manipulator", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        # Détection (RunningMode.IMAGE = appel synchrone)
        rgb_mp = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(rgb_mp)

        # Fond assombri
        frame = cv2.convertScaleAbs(frame, alpha=0.50, beta=0)

        detected_hands = []

        for hlm in result.hand_landmarks:
            lm = hlm
            draw_skeleton(frame, lm, W, H)

            pinched = check_pinch(lm)
            pp      = get_pinch_mid(lm, W, H)
            tip_px  = (int(lm[8].x*W), int(lm[8].y*H))

            detected_hands.append((pp, pinched, lm))

            col = (255, 230, 60) if pinched else (200, 200, 200)
            cv2.circle(frame, tip_px, 9, col, -1)
            if pinched:
                cv2.circle(frame, (int(pp[0]), int(pp[1])), 14, (255,220,50), 2)

        # ── Drag ─────────────────────────────────────────────────────────────
        any_pinched = any(p for _, p, _ in detected_hands)

        for pp, pinched, lm in detected_hands:
            if pinched:
                moved = False
                for s in shapes:
                    if s.dragging:
                        s.cx = pp[0] + s.drag_ox
                        s.cy = pp[1] + s.drag_oy
                        moved = True
                if not moved:
                    for s in reversed(shapes):
                        if not s.dragging and s.hit(pp[0], pp[1]):
                            s.dragging = True
                            s.selected = True
                            s.drag_ox  = s.cx - pp[0]
                            s.drag_oy  = s.cy - pp[1]
                            break

        if not any_pinched:
            for s in shapes:
                s.dragging = False

        # ── Scale + Rotation (2 mains pincées) ───────────────────────────────
        pinched_hands = [(pp, lm) for pp, p, lm in detected_hands if p]

        if len(pinched_hands) >= 2:
            pp0, pp1 = pinched_hands[0][0], pinched_hands[1][0]
            cur_dist  = math.hypot(pp1[0]-pp0[0], pp1[1]-pp0[1])
            cur_angle = math.degrees(math.atan2(pp1[1]-pp0[1], pp1[0]-pp0[0]))

            if prev_dist and prev_dist > 5:
                scale_f = cur_dist / prev_dist
                dangle  = cur_angle - prev_angle
                if dangle >  90: dangle -= 180
                if dangle < -90: dangle += 180

                for s in shapes:
                    if s.selected:
                        s.size  = max(25, min(220, s.size * scale_f))
                        s.angle = (s.angle + dangle) % 360

            prev_dist  = cur_dist
            prev_angle = cur_angle

            cv2.line(frame,
                     (int(pp0[0]), int(pp0[1])), (int(pp1[0]), int(pp1[1])),
                     (100,180,255), 1, cv2.LINE_AA)
        else:
            prev_dist  = None
            prev_angle = None

        # ── Dessin formes ─────────────────────────────────────────────────────
        for s in shapes:
            s.draw(frame)

        draw_hud(frame, W, H, sum(1 for s in shapes if s.selected))

        cv2.imshow("Hand Shape Manipulator", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            for s in shapes:
                s.selected = False

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
