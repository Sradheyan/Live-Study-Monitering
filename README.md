# 🎯 Study Focus Monitor

A real-time webcam study assistant that plays an alarm when you get distracted. It checks whether you are looking at your work, using your phone, nodding off, or away from your desk, and shows live stats and session history in a browser dashboard.

Everything runs **locally on your computer**. Video is analysed frame by frame in memory and is never saved or uploaded.
>

## Features

**Distraction detection**
- 👀 **Looking away**: head-pose tracking (left, right, up), measured against your own calibrated baseline
- 😴 **Drowsiness**: alarm when your eyes stay closed too long
- 🚶 **Left the desk**: alarm when your face is out of view
- 📱 **Phone detection** (YOLOv8): alarm when you pick up your phone
  - A phone resting on the desk during calibration is ignored
  - Size and shape filters cut false alarms from pens and small objects

**Study-friendly rules**
- 📓 **Notebook mode**: looking down to write or read notes does not trigger the alarm, and closed-eye detection pauses while you write
- ✍️ **Pen rule** (optional): looking down is only OK while your hand shows a pen grip (thumb and index finger pinched)

**Dashboard (Streamlit)**
- Start, Pause, Recalibrate and Stop and save buttons
- Live focus score, session timer, focused and distracted time, distraction count
- Rolling 30-second focus chart
- Sliders for every alarm delay and sensitivity setting
- Debug line under the video showing what the system currently sees
- **History** tab with charts: focus score per session, minutes per day, and which distraction type hits you most (saved to a local CSV)
- Custom alarm sound support
- Three video display modes, in case one does not work in your browser

## How it works

| Check | Method |
|---|---|
| Head pose and eye openness | [MediaPipe](https://developers.google.com/mediapipe) Face Mesh landmarks compared with a baseline from a 5-second calibration |
| Pen grip | MediaPipe Hands: thumb and index fingertips pinched together, relative to palm size |
| Phone | [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) "cell phone" class, plus size and shape filtering |
| Alarm | `pygame`, playing a generated beep or your own audio file |
| Interface | [Streamlit](https://streamlit.io) and OpenCV |

An alarm only fires when a distraction lasts longer than its delay (for example 3 seconds of looking away), so blinks and quick glances are ignored.

## Requirements

- **Python 3.12** (MediaPipe 0.10.14 does not support Python 3.13 or newer)
- A webcam
- Windows, macOS or Linux

## Usage

```bash
python -m streamlit run study_dashboard.py
```

1. Open the address Streamlit prints (usually `http://localhost:8501`).
2. Press **Start session** in the sidebar.
3. During the 5-second calibration, look at your screen naturally. Keep your phone still on the desk if you want it ignored.
4. Study. Use **Pause** to turn the camera off, **Recalibrate** if you move, and **Stop and save** to store the session.

The first run with phone detection downloads the YOLO model, so you need an internet connection once.

### Custom alarm sound

Place an audio file named `alarm.wav`, `alarm.ogg` or `alarm.mp3` next to `study_dashboard.py` (WAV or OGG is the most reliable). You can also edit the `ALARM_FILES` list at the top of the script. Without a file, a built-in beep is used.

### Video display modes

If the live video is blank in your browser, change **Video display** in the sidebar:

| Mode | Description |
|---|---|
| Compatibility (in page) | Default. Frames are embedded directly in the page |
| Standard (st.image) | Streamlit's built-in image display |
| Separate window (OpenCV) | Opens the camera view in its own desktop window |

## Tuning

Use the sidebar sliders together with the debug line under the video.

| Problem | Try |
|---|---|
| Alarm fires too easily | Increase the alarm delays or tolerances |
| "Eyes closed" alarms while writing | Keep Notebook mode on, or turn off **Detect closed eyes** |
| Pen detected as a phone | Raise **Minimum phone size** or lower **Max phone shape** |
| Phone not detected | Lower **Detection confidence**, or use the **Accurate** detector |
| Pen grip not recognized | Raise **Pen-grip sensitivity**, and make sure your writing hand is visible |
| Slow or laggy video | Use the **Fast** phone detector, or turn phone detection off |

## Limitations

- The pen rule detects a **pen-like grip**, not the pen itself, and your writing hand must be visible to the camera.
- Phones seen edge-on, very small or far away can be missed.
- Poor lighting or a camera angle that cuts off your face reduces accuracy.
- Thresholds depend on your camera and seating, so calibrate and tune for your setup.
- Phone detection runs a few times per second rather than on every frame.

## Troubleshooting

| Error or symptom | Fix |
|---|---|
| `No module named 'cv2'` | You are outside the virtual environment. Activate it and use `python -m streamlit run ...` |
| `No matching distribution found for mediapipe==0.10.14` | Use Python 3.12 |
| `GetPrototype` error | Run `python -m pip install "protobuf<5"` |
| `No module named 'pip._internal'` or `streamlit.proto` | The venv is corrupted or was moved. Delete it and create a new one |
| Camera does not open | Close other apps using the camera, allow desktop apps in Windows camera privacy settings, or change **Camera index** |
| Live video is blank | Try another **Video display** mode, another browser, or a hard refresh (`Ctrl+Shift+R`) |
| `Face: NOT SEEN` | Improve lighting, keep your whole face in view, sit 40 to 80 cm from the camera |

## Project structure

```
.
├── study_dashboard.py     # Streamlit app: detection, alarm, dashboard
├── study_sessions.csv     # created automatically after your first saved session
├── alarm.wav              # optional custom alarm sound
├── .gitignore
└── README.md
```

## Privacy

The camera feed is processed in memory on your own machine. No video or images are stored or sent anywhere. Only session summaries (times, scores and counts) are written to `study_sessions.csv`.

## Roadmap

- Train a custom model for pen and notebook detection
- Break reminders (Pomodoro mode)
- Exportable session reports


## Acknowledgements

[MediaPipe](https://developers.google.com/mediapipe), [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [OpenCV](https://opencv.org), [Streamlit](https://streamlit.io), [pygame](https://www.pygame.org)
