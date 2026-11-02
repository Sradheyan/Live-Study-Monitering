"""
Study Live Monitor - v1
Plays an alarm when you get distracted while studying.

Detects:
  1. No face in frame (left the desk)
  2. Looking away (head turned / tilted beyond calibrated range)
  3. Eyes closed (drowsy)

Keys:  q = quit (shows session summary)   r = recalibrate

Install:  pip install opencv-python mediapipe==0.10.14 pygame numpy
Run:      python study_monitor.py
"""

import time
import cv2
import numpy as np
import pygame
import mediapipe as mp

# ----------------------------- CONFIG --------------------------------------
CAMERA_INDEX = 0
CALIBRATION_SECONDS = 5

# How long a distraction must last before the alarm fires (seconds)
NO_FACE_SECS = 3.0
LOOK_AWAY_SECS = 3.0
EYES_CLOSED_SECS = 2.0

# Tolerances, as deviation from your calibrated "focused" pose (ratio units)
YAW_TOLERANCE = 0.17         # left / right turn
PITCH_UP_TOLERANCE = 0.12    # looking up
PITCH_DOWN_TOLERANCE = 0.22  # looking down (larger so reading notes is OK)
EAR_CLOSED_FACTOR = 0.70     # eyes "closed" if EAR < 70% of your baseline

SMOOTHING = 0.4              # 0..1, higher = more responsive, less smooth
ALARM_FREQ_HZ = 880
# ---------------------------------------------------------------------------

# MediaPipe Face Mesh landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP, FOREHEAD, CHIN = 1, 10, 152
FACE_LEFT, FACE_RIGHT = 234, 454


def make_alarm():
    """Build a beep-beep sound in memory (no audio file needed)."""
    sr = 44100
    t = np.linspace(0, 0.35, int(sr * 0.35), False)
    beep = np.sin(2 * np.pi * ALARM_FREQ_HZ * t) * 0.6
    gap = np.zeros(int(sr * 0.25))
    wave = np.concatenate([beep, gap])
    wave = (wave * 32767).astype(np.int16)

    # Match the mixer's channel count (some devices force stereo)
    channels = pygame.mixer.get_init()[2]
    if channels == 2:
        wave = np.column_stack((wave, wave))
    return pygame.sndarray.make_sound(wave)


def pt(lm, idx, w, h):
    return np.array([lm[idx].x * w, lm[idx].y * h])


def eye_aspect_ratio(lm, idxs, w, h):
    p1, p2, p3, p4, p5, p6 = [pt(lm, i, w, h) for i in idxs]
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (
        2.0 * np.linalg.norm(p1 - p4) + 1e-6
    )


def head_ratios(lm, w, h):
    """Return (yaw_ratio, pitch_ratio, ear). Roughly 0.5 when facing the camera."""
    nose = pt(lm, NOSE_TIP, w, h)
    left, right = pt(lm, FACE_LEFT, w, h), pt(lm, FACE_RIGHT, w, h)
    top, bottom = pt(lm, FOREHEAD, w, h), pt(lm, CHIN, w, h)
    yaw = (nose[0] - left[0]) / (right[0] - left[0] + 1e-6)
    pitch = (nose[1] - top[1]) / (bottom[1] - top[1] + 1e-6)
    ear = (
        eye_aspect_ratio(lm, LEFT_EYE, w, h) + eye_aspect_ratio(lm, RIGHT_EYE, w, h)
    ) / 2.0
    return yaw, pitch, ear


def put(img, text, y, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def main():
    pygame.mixer.init(frequency=44100, size=-16, channels=1)
    alarm = make_alarm()
    alarm_channel = pygame.mixer.Channel(0)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise SystemExit("Could not open camera. Try a different CAMERA_INDEX.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Calibration state
    calibrating = True
    calib_start = time.time()
    calib_samples = []
    base_yaw = base_pitch = base_ear = None

    # Smoothed values
    s_yaw = s_pitch = s_ear = None

    # Timers: when each condition started (None = not currently happening)
    since = {"no_face": None, "look_away": None, "eyes_closed": None}
    limits = {
        "no_face": NO_FACE_SECS,
        "look_away": LOOK_AWAY_SECS,
        "eyes_closed": EYES_CLOSED_SECS,
    }
    labels = {
        "no_face": "Where did you go?",
        "look_away": "Eyes on the screen!",
        "eyes_closed": "Wake up!",
    }

    # Session stats
    focused_time = distracted_time = 0.0
    distraction_events = 0
    was_distracted = False
    last = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        dt, last = now - last, now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        face = result.multi_face_landmarks[0].landmark if result.multi_face_landmarks else None

        # ------------------------- calibration ------------------------------
        if calibrating:
            elapsed = now - calib_start
            put(frame, "CALIBRATING: look at your screen naturally", 30, (0, 255, 255))
            put(frame, f"{max(0, CALIBRATION_SECONDS - elapsed):.1f}s", 60, (0, 255, 255))
            if face:
                calib_samples.append(head_ratios(face, w, h))
            if elapsed >= CALIBRATION_SECONDS:
                if len(calib_samples) >= 15:
                    arr = np.array(calib_samples)
                    base_yaw, base_pitch, base_ear = np.median(arr, axis=0)
                    calibrating = False
                    s_yaw = s_pitch = s_ear = None
                    for k in since:
                        since[k] = None
                    last = time.time()
                else:  # no face seen, try again
                    calib_samples = []
                    calib_start = time.time()
            cv2.imshow("Study Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        # ------------------------- detection --------------------------------
        flags = {"no_face": False, "look_away": False, "eyes_closed": False}

        if face is None:
            flags["no_face"] = True
        else:
            yaw, pitch, ear = head_ratios(face, w, h)
            if s_yaw is None:
                s_yaw, s_pitch, s_ear = yaw, pitch, ear
            else:
                a = SMOOTHING
                s_yaw = a * yaw + (1 - a) * s_yaw
                s_pitch = a * pitch + (1 - a) * s_pitch
                s_ear = a * ear + (1 - a) * s_ear

            d_yaw = s_yaw - base_yaw
            d_pitch = s_pitch - base_pitch  # positive = looking down

            if abs(d_yaw) > YAW_TOLERANCE:
                flags["look_away"] = True
            if d_pitch > PITCH_DOWN_TOLERANCE or d_pitch < -PITCH_UP_TOLERANCE:
                flags["look_away"] = True
            if s_ear < base_ear * EAR_CLOSED_FACTOR:
                flags["eyes_closed"] = True

            put(frame, f"yaw {d_yaw:+.2f}  pitch {d_pitch:+.2f}  EAR {s_ear:.2f}", h - 15, scale=0.5)

        # ------------------------- timers -----------------------------------
        active = []
        for name, on in flags.items():
            if on:
                if since[name] is None:
                    since[name] = now
                if now - since[name] >= limits[name]:
                    active.append(name)
            else:
                since[name] = None

        distracted = len(active) > 0
        if distracted:
            distracted_time += dt
            if not was_distracted:
                distraction_events += 1
            if not alarm_channel.get_busy():
                alarm_channel.play(alarm, loops=-1)
        else:
            focused_time += dt
            alarm_channel.stop()
        was_distracted = distracted

        # ------------------------- overlay ----------------------------------
        total = focused_time + distracted_time
        focus_pct = 100 * focused_time / total if total > 0 else 100
        if distracted:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            put(frame, labels[active[0]], 35, (0, 0, 255), 0.9)
        else:
            put(frame, "Focused", 35, (0, 255, 0), 0.9)
        put(frame, f"Focus {focus_pct:.0f}%  |  Distractions: {distraction_events}", 65)
        put(frame, "q: quit   r: recalibrate", h - 40, scale=0.5)

        cv2.imshow("Study Monitor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            calibrating = True
            calib_start = time.time()
            calib_samples = []
            alarm_channel.stop()

    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.quit()

    total = focused_time + distracted_time
    print("\n===== Session summary =====")
    print(f"Total time      : {total / 60:.1f} min")
    print(f"Focused         : {focused_time / 60:.1f} min")
    print(f"Distracted      : {distracted_time / 60:.1f} min")
    print(f"Distraction hits: {distraction_events}")
    if total > 0:
        print(f"Focus score     : {100 * focused_time / total:.0f}%")


if __name__ == "__main__":
    main()