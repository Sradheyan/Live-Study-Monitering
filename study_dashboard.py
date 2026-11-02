"""
Study Focus Monitor - v3 (Streamlit dashboard)

Run:  streamlit run study_dashboard.py

Install (inside your venv):
      pip install streamlit pandas opencv-python mediapipe==0.10.14 pygame numpy
      pip install "protobuf<5"
      pip install ultralytics        # only needed for phone detection

New in v4: pen-grip rule (looking down without a pen in hand triggers the alarm),
  better sideways-phone detection (bigger model, higher resolution).

New in v3:
  * Notebook mode: looking down to write/read notes does not trigger the alarm
  * Phone detection (YOLO): alarm when you pick up your phone
  * A phone resting on the desk during calibration is ignored
"""

import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pygame
import streamlit as st

try:
    from ultralytics import YOLO

    HAS_YOLO = True
except Exception:
    HAS_YOLO = False

HISTORY_FILE = Path(__file__).with_name("study_sessions.csv")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Put your own alarm sound next to this file. First one found is used.
ALARM_FILES = ["Padhai Likhai Karo Ias Bano Meme.mp3", "alarm.wav", "alarm.ogg", "alarm.mp3"]

# MediaPipe Face Mesh landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP, FOREHEAD, CHIN = 1, 10, 152
FACE_LEFT, FACE_RIGHT = 234, 454

TYPE_COLS = ["phone", "no_pen", "no_face", "look_away", "eyes_closed"]
REASON_TEXT = {
    "phone": "Phone detected - put it down!",
    "no_pen": "Looking down with no pen - write or look up",
    "no_face": "You left the desk",
    "look_away": "Eyes on the screen!",
    "eyes_closed": "Eyes closed - wake up!",
}
PHONE_CLASS_ID = 67  # "cell phone" in the COCO dataset


# ----------------------------- alarm ---------------------------------------
@st.cache_resource
def get_alarm(freq=880):
    """Your alarm file if present, otherwise a built-in beep. Returns (sound, channel)."""
    if pygame.mixer.get_init() is None:
        pygame.mixer.init(frequency=44100, size=-16, channels=1)

    for name in ALARM_FILES:
        path = Path(__file__).with_name(name)
        if path.exists():
            try:
                sound = pygame.mixer.Sound(str(path))
                sound.set_volume(1.0)  # 0.0 to 1.0
                return sound, pygame.mixer.Channel(0)
            except pygame.error:
                pass  # unreadable file, fall back to the beep

    sr = 44100
    t = np.linspace(0, 0.35, int(sr * 0.35), False)
    beep = np.sin(2 * np.pi * freq * t) * 0.6
    wave = np.concatenate([beep, np.zeros(int(sr * 0.25))])
    wave = (wave * 32767).astype(np.int16)
    if pygame.mixer.get_init()[2] == 2:  # some devices force stereo
        wave = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(wave), pygame.mixer.Channel(0)


# ----------------------------- phone detection -----------------------------
@st.cache_resource(show_spinner="Loading phone detector (first run downloads the model)...")
def get_phone_model(weights="yolov8s.pt"):
    return YOLO(weights)


def detect_phones(model, frame_bgr, conf, imgsz=640):
    """Return a list of (x1, y1, x2, y2) boxes for phones in the frame."""
    res = model.predict(
        frame_bgr, imgsz=imgsz, conf=conf, classes=[PHONE_CLASS_ID],
        verbose=False, device="cpu",
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return []
    return [tuple(float(v) for v in b) for b in res.boxes.xyxy.cpu().numpy()]


def phone_shape_ok(b, w, h, min_area_frac, max_aspect):
    """A pen is small and very thin; a phone is bigger and closer to a rectangle."""
    bw, bh = b[2] - b[0], b[3] - b[1]
    if bw <= 0 or bh <= 0:
        return False
    big_enough = (bw * bh) / float(w * h) >= min_area_frac
    not_too_thin = max(bw, bh) / min(bw, bh) <= max_aspect
    return big_enough and not_too_thin


def box_center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


def matches_rest(box, rests, margin=0.35):
    """True if the box centre sits inside (a slightly enlarged) resting-phone box."""
    cx, cy = box_center(box)
    for r in rests:
        rw, rh = r[2] - r[0], r[3] - r[1]
        if (r[0] - margin * rw <= cx <= r[2] + margin * rw
                and r[1] - margin * rh <= cy <= r[3] + margin * rh):
            return True
    return False


def pen_grip(hand_lm, w, h, ratio):
    """True if thumb tip and index tip are pinched together (how you hold a pen).
    The gap is measured relative to palm length so it works at any distance."""
    p = lambda i: np.array([hand_lm[i].x * w, hand_lm[i].y * h])
    palm = np.linalg.norm(p(0) - p(9)) + 1e-6
    return np.linalg.norm(p(4) - p(8)) / palm < ratio


# ----------------------------- vision helpers ------------------------------
def pt(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h])


def eye_aspect_ratio(lm, idxs, w, h):
    p1, p2, p3, p4, p5, p6 = [pt(lm, i, w, h) for i in idxs]
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (
        2.0 * np.linalg.norm(p1 - p4) + 1e-6
    )


def head_ratios(lm, w, h):
    """(yaw_ratio, pitch_ratio, ear); about 0.5 / 0.5 when facing the camera."""
    nose = pt(lm, NOSE_TIP, w, h)
    left, right = pt(lm, FACE_LEFT, w, h), pt(lm, FACE_RIGHT, w, h)
    top, bottom = pt(lm, FOREHEAD, w, h), pt(lm, CHIN, w, h)
    yaw = (nose[0] - left[0]) / (right[0] - left[0] + 1e-6)
    pitch = (nose[1] - top[1]) / (bottom[1] - top[1] + 1e-6)
    ear = (
        eye_aspect_ratio(lm, LEFT_EYE, w, h) + eye_aspect_ratio(lm, RIGHT_EYE, w, h)
    ) / 2.0
    return yaw, pitch, ear


def decorate(img, color, label):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), color, 6)
    cv2.rectangle(img, (0, 0), (w, 38), (0, 0, 0), -1)
    cv2.putText(img, label, (12, 27), FONT, 0.75, color, 2)


def draw_box(img, b, color, label):
    x1, y1, x2, y2 = [int(v) for v in b]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(52, y1 - 6)), FONT, 0.55, color, 2)


# ----------------------------- monitor -------------------------------------
class FocusMonitor:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.started_at = datetime.now()
        self.focused = 0.0
        self.distracted = 0.0
        self.by_type = {k: 0 for k in TYPE_COLS}
        self.prev_active = set()
        self.timeline = []  # 1 = focused, 0 = distracted, sampled every second
        self.last_sample = 0.0
        self.last_t = None
        # phone detection state
        self.phone_boxes = []
        self.last_det = 0.0
        self.last_phone_seen = 0.0
        # pen-grip detection (MediaPipe Hands)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.last_pen_seen = 0.0
        # notebook-mode state
        self.last_pitch_delta = 0.0
        self.pitch_hist = []   # (time, pitch) for the last ~2 s while the face is visible
        self.pitch_peak = 0.0  # highest recent head-down pitch (frozen when the face is lost)
        self.diag = {}         # live values shown under the video
        self.last_face_t = 0.0
        self.start_calibration()

    # --- derived stats
    @property
    def monitored(self):
        return self.focused + self.distracted

    @property
    def events(self):
        return sum(self.by_type.values())

    @property
    def score(self):
        return 100.0 * self.focused / self.monitored if self.monitored > 0 else 100.0

    def start_calibration(self):
        self.calibrating = True
        self.calib_start = None
        self.calib_left = 0.0
        self.samples = []
        self.base = None
        self.smooth = None
        self.rest_boxes = []  # phones resting on the desk during calibration
        self.since = {k: None for k in TYPE_COLS}
        self.prev_active = set()

    def close(self):
        self.face_mesh.close()
        self.hands.close()

    def process(self, frame_bgr, cfg, phone_model=None):
        """Returns (rgb_frame, state, active_reasons).
        state is 'calibrating', 'focused' or 'distracted'."""
        now = time.time()
        dt = 0.0 if self.last_t is None else min(now - self.last_t, 0.5)
        self.last_t = now

        frame_bgr = cv2.flip(frame_bgr, 1)
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        face = res.multi_face_landmarks[0].landmark if res.multi_face_landmarks else None

        # ---- phone detection (throttled, runs ~3x per second)
        if phone_model is not None and cfg["phone_on"]:
            if now - self.last_det >= 0.35:
                self.last_det = now
                self.phone_boxes = detect_phones(
                    phone_model, frame_bgr, cfg["phone_conf"], cfg["phone_imgsz"]
                )
        else:
            self.phone_boxes = []

        # ---- calibration
        if self.calibrating:
            if self.calib_start is None:
                self.calib_start = now
            elapsed = now - self.calib_start
            self.calib_left = max(0.0, cfg["calib_s"] - elapsed)
            if face:
                self.samples.append(head_ratios(face, w, h))
            if cfg["phone_rest"]:
                for b in self.phone_boxes:
                    if (phone_shape_ok(b, w, h, cfg["phone_min_area"], cfg["phone_max_aspect"])
                            and not matches_rest(b, self.rest_boxes, margin=0.0)):
                        self.rest_boxes.append(b)
            for b in self.rest_boxes:
                draw_box(rgb, b, (150, 150, 150), "resting phone")
            if elapsed >= cfg["calib_s"]:
                if len(self.samples) >= 15:
                    self.base = np.median(np.array(self.samples), axis=0)
                    self.calibrating = False
                else:  # no face seen, try again
                    self.samples = []
                    self.calib_start = now
            decorate(rgb, (255, 190, 0), f"Calibrating {self.calib_left:.1f}s")
            return rgb, "calibrating", []

        # ---- phones that are NOT resting on the desk
        pen_before = self.last_pen_seen > 0 and (now - self.last_pen_seen) < cfg["pen_hold_s"]
        area_need = cfg["phone_min_area"] * (2.5 if pen_before else 1.0)  # stricter while gripping a pen
        candidates = [b for b in self.phone_boxes if not matches_rest(b, self.rest_boxes)]
        live_phones = [b for b in candidates
                       if phone_shape_ok(b, w, h, area_need, cfg["phone_max_aspect"])]
        rejected = [b for b in candidates if b not in live_phones]
        if live_phones:
            self.last_phone_seen = now

        # ---- detection
        flags = {k: False for k in TYPE_COLS}
        flags["phone"] = self.last_phone_seen > 0 and (now - self.last_phone_seen) < 0.8

        # ---- pen grip: is a hand pinching like it holds a pen?
        n_hands = 0
        if cfg["notebook_mode"] and cfg["pen_rule"]:
            hres = self.hands.process(rgb)
            n_hands = len(hres.multi_hand_landmarks or [])
            for hl in hres.multi_hand_landmarks or []:
                grip = pen_grip(hl.landmark, w, h, cfg["pen_grip"])
                if grip:
                    self.last_pen_seen = now
                mp.solutions.drawing_utils.draw_landmarks(
                    rgb, hl, mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(60, 220, 120) if grip else (150, 150, 150),
                        thickness=1, circle_radius=1),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(60, 220, 120) if grip else (150, 150, 150), thickness=1),
                )
                wx, wy = int(hl.landmark[0].x * w), int(hl.landmark[0].y * h)
                cv2.putText(rgb, "pen grip" if grip else "hand", (wx, min(h - 8, wy + 22)),
                            FONT, 0.5, (60, 220, 120) if grip else (200, 200, 200), 1)
        pen_recent = self.last_pen_seen > 0 and (now - self.last_pen_seen) < cfg["pen_hold_s"]
        head_down = False
        d_pitch = d_yaw = ear = None

        if face is None:
            # While writing, your face can drop out of view. Give a grace period if
            # the last pose was head-down.
            writing = (
                cfg["notebook_mode"]
                and (now - self.last_face_t) < cfg["notebook_grace_s"]
                and (self.pitch_peak > cfg["head_down"] * 0.6 or pen_recent)
            )
            if writing:
                head_down = True
            else:
                flags["no_face"] = True
        else:
            self.last_face_t = now
            cur = np.array(head_ratios(face, w, h))
            self.smooth = cur if self.smooth is None else 0.4 * cur + 0.6 * self.smooth
            d_yaw = self.smooth[0] - self.base[0]
            d_pitch = self.smooth[1] - self.base[1]  # positive = looking down
            ear = self.smooth[2]
            self.last_pitch_delta = d_pitch
            self.pitch_hist = [(t, p) for t, p in self.pitch_hist if now - t < 2.0] + [(now, d_pitch)]
            self.pitch_peak = max(p for _, p in self.pitch_hist)

            head_down = d_pitch > cfg["head_down"]
            # foreshortening distorts the yaw ratio when the head is bowed, so be lenient
            yaw_limit = cfg["yaw_tol"] * (1.6 if (cfg["notebook_mode"] and head_down) else 1.0)
            if abs(d_yaw) > yaw_limit:
                flags["look_away"] = True
            if d_pitch < -cfg["pitch_up"]:
                flags["look_away"] = True
            if d_pitch > cfg["pitch_down"] and not cfg["notebook_mode"]:
                flags["look_away"] = True

            # Looking down makes eyes look closed, so skip that check in notebook mode
            head_down = d_pitch > cfg["head_down"]
            # While writing, the eyes naturally look narrower, so skip this check then
            writing_now = cfg["notebook_mode"] and (
                head_down or d_pitch > cfg["head_down"] * 0.5 or pen_recent)
            if cfg["eyes_on"] and ear < self.base[2] * cfg["ear_factor"] and not writing_now:
                flags["eyes_closed"] = True

        # Looking down is fine while holding a pen; looking down without one is not
        if cfg["notebook_mode"] and cfg["pen_rule"] and head_down and not pen_recent:
            flags["no_pen"] = True

        self.diag = {
            "face": face is not None, "pitch": d_pitch, "head_down": head_down,
            "hands": n_hands, "pen": pen_recent, "phones": len(live_phones),
            "ear": ear, "ear_thr": float(self.base[2] * cfg["ear_factor"]),
        }

        limits = {
            "phone": cfg["phone_s"],
            "no_pen": cfg["nopen_s"],
            "no_face": cfg["no_face_s"],
            "look_away": cfg["look_away_s"],
            "eyes_closed": cfg["eyes_closed_s"],
        }
        active = []
        for name, on in flags.items():
            if on:
                if self.since[name] is None:
                    self.since[name] = now
                if now - self.since[name] >= limits[name]:
                    active.append(name)
            else:
                self.since[name] = None

        distracted = bool(active)
        for name in set(active) - self.prev_active:
            self.by_type[name] += 1
        self.prev_active = set(active)

        if distracted:
            self.distracted += dt
        else:
            self.focused += dt

        if now - self.last_sample >= 1.0:
            self.last_sample = now
            self.timeline.append(0 if distracted else 1)

        # ---- draw
        for b in self.rest_boxes:
            draw_box(rgb, b, (150, 150, 150), "resting phone")
        for b in rejected:
            draw_box(rgb, b, (255, 170, 0), "ignored: too small/thin")
        for b in live_phones:
            draw_box(rgb, b, (255, 70, 70), "phone")

        if distracted:
            decorate(rgb, (255, 70, 70), REASON_TEXT[active[0]])
            return rgb, "distracted", active
        decorate(rgb, (60, 220, 120), "Focused")
        return rgb, "focused", active


# ----------------------------- history -------------------------------------
def save_session(mon):
    if mon.monitored < 10:  # ignore accidental tiny sessions
        return False
    row = {
        "started": mon.started_at.strftime("%d %b %H:%M"),
        "date": mon.started_at.strftime("%Y-%m-%d"),
        "total_min": round(mon.monitored / 60, 2),
        "focused_min": round(mon.focused / 60, 2),
        "distracted_min": round(mon.distracted / 60, 2),
        "focus_score": round(mon.score, 1),
        "distractions": mon.events,
        **mon.by_type,
    }
    new = pd.DataFrame([row])
    if HISTORY_FILE.exists():  # merge with older files that lack the "phone" column
        try:
            new = pd.concat([pd.read_csv(HISTORY_FILE), new], ignore_index=True)
        except Exception:
            pass
    new.to_csv(HISTORY_FILE, index=False)
    return True


def load_history():
    if HISTORY_FILE.exists():
        try:
            df = pd.read_csv(HISTORY_FILE)
            for col in TYPE_COLS:
                if col not in df:
                    df[col] = 0
            df[TYPE_COLS] = df[TYPE_COLS].fillna(0)
            return df
        except Exception:
            pass
    return pd.DataFrame()


def fmt_time(seconds):
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


# ----------------------------- page setup ----------------------------------
st.set_page_config(page_title="Study Focus Monitor", page_icon="🎯", layout="wide")
st.markdown(
    """
<style>
.banner{padding:14px 18px;border-radius:12px;font-size:1.15rem;font-weight:600;margin:4px 0 12px}
.ok{background:#12372a;color:#4ade80;border:1px solid #1f6b4d}
.bad{background:#3b1518;color:#ff8a8a;border:1px solid #7a2b31}
.warn{background:#3a2d0b;color:#ffcf5c;border:1px solid #7a6020}
.idle{background:#1e2430;color:#b8c2d6;border:1px solid #39445a}
</style>
""",
    unsafe_allow_html=True,
)

ss = st.session_state
ss.setdefault("status", "idle")  # idle | running | paused
ss.setdefault("mon", None)
ss.setdefault("flash", None)


# ----------------------------- button callbacks ----------------------------
def on_start():
    if ss.mon is not None:
        ss.mon.close()
    ss.mon = FocusMonitor()
    ss.status = "running"


def on_pause():
    ss.status = "paused"


def on_resume():
    ss.mon.last_t = None
    ss.status = "running"


def on_recalibrate():
    ss.mon.start_calibration()
    ss.mon.last_t = None
    ss.status = "running"


def on_stop():
    saved = save_session(ss.mon)
    ss.mon.close()
    ss.mon = None
    ss.status = "idle"
    ss.flash = "Session saved to history." if saved else "Session too short to save."


# ----------------------------- sidebar -------------------------------------
with st.sidebar:
    st.title("🎯 Focus Monitor")
    idle, running, paused = (ss.status == k for k in ("idle", "running", "paused"))

    st.button("▶ Start session", on_click=on_start, disabled=not idle,
              type="primary", use_container_width=True)
    c1, c2 = st.columns(2)
    if paused:
        c1.button("⏵ Resume", on_click=on_resume, use_container_width=True)
    else:
        c1.button("⏸ Pause", on_click=on_pause, disabled=not running, use_container_width=True)
    c2.button("🎯 Recalibrate", on_click=on_recalibrate, disabled=idle, use_container_width=True)
    st.button("⏹ Stop & save", on_click=on_stop, disabled=idle, use_container_width=True)

    st.divider()
    st.subheader("Notebook mode")
    notebook_mode = st.toggle(
        "Allow writing / reading notes", True,
        help="Looking down at a notebook won't trigger the alarm, and eyes-closed "
             "detection is paused while your head is tilted down.",
    )
    pen_rule = st.toggle(
        "Alarm if I look down without a pen", True,
        help="Looking down is OK only while a hand is gripping a pen. Needs your writing "
             "hand to be visible to the camera (it's drawn green as 'pen grip').",
    )
    nopen_s = st.slider("Alarm after head down, no pen (s)", 2.0, 20.0, 5.0, 0.5)
    pen_grip_ratio = st.slider("Pen-grip sensitivity", 0.20, 0.90, 0.45, 0.05,
                               help="Higher = more hand shapes count as holding a pen.")
    head_down_thr = st.slider("Head-down threshold", 0.03, 0.25, 0.08, 0.01,
                              help="How far your head must tilt down to count as looking down.")

    st.subheader("Phone detection")
    if HAS_YOLO:
        phone_on = st.toggle("Alarm when I pick up my phone", True)
        phone_s = st.slider("Alarm after phone visible (s)", 0.5, 5.0, 1.5, 0.5)
        phone_quality = st.selectbox(
            "Phone detector", ["Accurate (slower)", "Fast"],
            help="Accurate finds phones held sideways or at an angle better, but uses more CPU.")
        phone_conf = st.slider("Detection confidence", 0.15, 0.80, 0.35, 0.05,
                               help="Lower = detects more phones but more false alarms.")
        phone_min_area = st.slider(
            "Minimum phone size (% of frame)", 0.5, 10.0, 1.5, 0.5,
            help="Detections smaller than this are ignored (pens, small objects). "
                 "They appear as orange 'ignored' boxes.") / 100.0
        phone_max_aspect = st.slider(
            "Max phone shape (long side / short side)", 1.5, 5.0, 2.8, 0.1,
            help="Very long, thin detections (like a pen) are ignored.")
        phone_rest = st.toggle(
            "Ignore phone resting on desk", True,
            help="Phones visible during the 5-second calibration are treated as resting "
                 "on the desk and ignored. Keep your phone still on the desk while calibrating.",
        )
    else:
        phone_on, phone_s, phone_conf, phone_rest, phone_quality = False, 1.5, 0.3, True, "Fast"
        phone_min_area, phone_max_aspect = 0.015, 2.8
        st.info("Phone detection is off. Install it with:\n\n`pip install ultralytics`")

    st.divider()
    st.subheader("Alarm timing")
    look_away_s = st.slider("Looking away (s)", 1.0, 10.0, 3.0, 0.5)
    eyes_closed_s = st.slider("Eyes closed (s)", 0.5, 6.0, 2.0, 0.5)
    no_face_s = st.slider("Left the desk (s)", 1.0, 15.0, 3.0, 0.5)

    st.subheader("Sensitivity")
    yaw_tol = st.slider("Head turn tolerance", 0.05, 0.40, 0.17, 0.01,
                        help="Lower = alarm triggers on smaller left/right turns. "
                             "Raise it if your notebook sits to the side.")
    pitch_up = st.slider("Look-up tolerance", 0.05, 0.40, 0.12, 0.01)
    pitch_down = st.slider("Look-down tolerance", 0.05, 0.50, 0.22, 0.01,
                           help="Only used when Notebook mode is off.")
    eyes_on = st.toggle("Detect closed eyes (drowsiness)", True,
                        help="Turn off if you get 'eyes closed' alarms while writing.")
    ear_factor = st.slider("Eye-closed threshold", 0.40, 0.90, 0.70, 0.01,
                           help="Eyes count as closed below this fraction of your open-eye baseline.")

    st.subheader("Other")
    sound = st.toggle("Alarm sound", True)
    show_debug = st.toggle("Show debug info under video", True)
    cam_idx = st.number_input("Camera index", 0, 5, 0)

cfg = {
    "yaw_tol": yaw_tol, "pitch_up": pitch_up, "pitch_down": pitch_down,
    "ear_factor": ear_factor, "look_away_s": look_away_s,
    "eyes_closed_s": eyes_closed_s, "no_face_s": no_face_s, "calib_s": 5.0,
    "notebook_mode": notebook_mode, "notebook_grace_s": 30.0,
    "phone_on": phone_on, "phone_s": phone_s, "phone_conf": phone_conf,
    "phone_rest": phone_rest,
    "phone_model": "yolov8s.pt" if phone_quality.startswith("Accurate") else "yolov8n.pt",
    "phone_imgsz": 640 if phone_quality.startswith("Accurate") else 416,
    "pen_rule": pen_rule, "nopen_s": nopen_s, "pen_grip": pen_grip_ratio,
    "pen_hold_s": 1.5, "head_down": head_down_thr,
    "phone_min_area": phone_min_area, "phone_max_aspect": phone_max_aspect,
    "eyes_on": eyes_on,
}

if ss.flash:
    st.toast(ss.flash)
    ss.flash = None

# ----------------------------- main layout ---------------------------------
st.title("Study Focus Monitor")
banner_ph = st.empty()
cols = st.columns(5)
m_ph = [c.empty() for c in cols]

tab_live, tab_hist = st.tabs(["Live", "History"])
with tab_live:
    left, right = st.columns([2, 1])
    frame_ph = left.empty()
    diag_ph = left.empty()
    with right:
        st.subheader("Focus over time")
        chart_ph = st.empty()
        st.caption("Rolling 30-second focus %. Dips show when you drifted.")

with tab_hist:
    df = load_history()
    if df.empty:
        st.info("No saved sessions yet. Finish a session with 'Stop & save' and it shows up here.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Sessions", len(df))
        b.metric("Total focused", f"{df['focused_min'].sum() / 60:.1f} h")
        c.metric("Average focus score", f"{df['focus_score'].mean():.0f}%")
        d.metric("Best session", f"{df['focus_score'].max():.0f}%")

        st.subheader("Focus score per session")
        st.bar_chart(df.set_index("started")["focus_score"])

        h1, h2 = st.columns(2)
        with h1:
            st.subheader("Minutes per day")
            daily = df.groupby("date")[["focused_min", "distracted_min"]].sum()
            st.bar_chart(daily)
        with h2:
            st.subheader("What distracts you most")
            types = df[TYPE_COLS].sum().rename(
                {"phone": "Phone", "no_pen": "Down, no pen", "no_face": "Left desk",
                 "look_away": "Looked away", "eyes_closed": "Eyes closed"}
            )
            st.bar_chart(types)

        st.subheader("All sessions")
        st.dataframe(df.iloc[::-1].reset_index(drop=True), hide_index=True)


# ----------------------------- rendering helpers ---------------------------
def render_banner(state, active):
    if ss.status == "idle":
        banner_ph.markdown('<div class="banner idle">Ready. Press <b>Start session</b> in the sidebar.</div>',
                           unsafe_allow_html=True)
    elif ss.status == "paused":
        banner_ph.markdown('<div class="banner idle">Paused. Camera is off.</div>',
                           unsafe_allow_html=True)
    elif state == "calibrating":
        banner_ph.markdown('<div class="banner warn">Calibrating: look at your screen naturally '
                           '(phone on the desk is fine)...</div>', unsafe_allow_html=True)
    elif state == "distracted":
        banner_ph.markdown(f'<div class="banner bad">{REASON_TEXT[active[0]]}</div>',
                           unsafe_allow_html=True)
    else:
        banner_ph.markdown('<div class="banner ok">Focused. Keep going!</div>',
                           unsafe_allow_html=True)


def render_metrics(mon):
    vals = [
        ("Focus score", f"{mon.score:.0f}%" if mon else "-"),
        ("Session time", fmt_time(mon.monitored) if mon else "00:00"),
        ("Focused", fmt_time(mon.focused) if mon else "00:00"),
        ("Distracted", fmt_time(mon.distracted) if mon else "00:00"),
        ("Distractions", str(mon.events) if mon else "0"),
    ]
    for ph, (label, value) in zip(m_ph, vals):
        ph.metric(label, value)


def render_diag(mon):
    d = mon.diag if mon else {}
    if not show_debug or not d:
        diag_ph.empty()
        return
    pitch = f"{d['pitch']:+.2f}" if d["pitch"] is not None else "n/a"
    diag_ph.caption(
        f"Face: {'yes' if d['face'] else 'NOT SEEN'}  |  "
        f"head down: {'yes' if d['head_down'] else 'no'} (pitch {pitch}, needs > {cfg['head_down']:.2f})  |  "
        f"hands: {d['hands']}  |  pen grip: {'yes' if d['pen'] else 'no'}  |  phones: {d['phones']}  |  "
        + (f"eye openness {d['ear']:.2f} (counts as closed below {d['ear_thr']:.2f})"
           if d["ear"] is not None else "eyes: n/a")
    )


def render_chart(mon):
    if mon and len(mon.timeline) > 1:
        s = pd.Series(mon.timeline).rolling(30, min_periods=1).mean() * 100
        chart_ph.line_chart(pd.DataFrame({"Focus %": s.tail(300)}))


# ----------------------------- initial render ------------------------------
mon = ss.mon
render_banner("idle", [])
render_metrics(mon)
render_chart(mon)
if ss.status == "idle":
    frame_ph.info("Camera is off. Start a session to begin.")
elif ss.status == "paused":
    frame_ph.info("Paused. Press Resume to continue.")

# ----------------------------- live loop -----------------------------------
if ss.status == "running":
    alarm, channel = get_alarm()

    phone_model = None
    if HAS_YOLO and cfg["phone_on"]:
        try:
            phone_model = get_phone_model(cfg["phone_model"])
        except Exception as e:  # e.g. no internet for the first weights download
            st.sidebar.warning(f"Phone detector unavailable: {e}")

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(int(cam_idx), backend)
    if not cap.isOpened():
        frame_ph.error("Could not open the camera. Close other apps using it, or change the camera index.")
        ss.status = "paused"
    else:
        last_ui = last_chart = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    frame_ph.error("Lost the camera feed.")
                    ss.status = "paused"
                    break
                img, state, active = mon.process(frame, cfg, phone_model)

                if state == "distracted" and sound:
                    if not channel.get_busy():
                        channel.play(alarm, loops=-1)
                else:
                    channel.stop()

                frame_ph.image(img, channels="RGB")

                now = time.time()
                if now - last_ui > 0.25:
                    last_ui = now
                    render_banner(state, active)
                    render_metrics(mon)
                    render_diag(mon)
                if now - last_chart > 2.0:
                    last_chart = now
                    render_chart(mon)
        finally:  # runs when Streamlit reruns (button / slider) or the loop ends
            cap.release()
            channel.stop()