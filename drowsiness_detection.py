import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import threading
import time
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# =========================
# Config
# =========================
MODEL_PATH     = os.path.join(HERE, 'best.keras')
IMG_SIZE       = (150, 150)
CLASSES        = ['open', 'close']
CLOSED_THRESH  = 20
SMOOTH_WINDOW  = 5
ALARM_SOUND    = os.path.join(HERE, 'Fahh.mp3')
ALARM_COOLDOWN = 3.0           
FACE_CASCADE   = os.path.join(HERE, 'haarcascade_frontalface_default.xml')
EYE_CASCADE    = os.path.join(HERE, 'haarcascade_eye.xml')


# =========================
# Sound alarm
# Tries pygame first (best cross-platform support),
# falls back to OS-level commands if pygame isn't installed.
# Playback state lives on the instance instead of module globals.
# =========================
class AlarmPlayer:
    def __init__(self, sound_path=ALARM_SOUND, cooldown=ALARM_COOLDOWN):
        self.sound_path      = sound_path
        self.cooldown        = cooldown
        self.last_alarm_time = 0.0
        self.playing         = False

        if not os.path.exists(self.sound_path):
            print(f"[WARNING] Alarm file not found: '{self.sound_path}'")
            print("          Set ALARM_SOUND to a valid .wav or .mp3 path.")
        else:
            print(f"[INFO] Alarm sound loaded: {self.sound_path}")

    def _play_sound_thread(self):
        """Runs in a background thread so it never blocks the video loop."""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(self.sound_path)
            pygame.mixer.music.play()
            # Wait until playback finishes before clearing the flag
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except ImportError:
            # pygame not installed — fall back to OS commands
            if os.name == 'nt':                          # Windows
                import winsound
                winsound.PlaySound(self.sound_path, winsound.SND_FILENAME)
            elif os.uname().sysname == 'Darwin':         # macOS
                os.system(f'afplay "{self.sound_path}"')
            else:                                        # Linux
                os.system(f'aplay "{self.sound_path}" 2>/dev/null')
        except Exception as e:
            print(f"[ALARM ERROR] {e}")
        finally:
            self.playing = False

    def trigger(self):
        """Fire the alarm sound in a non-blocking background thread."""
        now = time.time()
        if self.playing:
            return                                       
        if now - self.last_alarm_time < self.cooldown:
            return                                      
        self.last_alarm_time = now
        self.playing         = True
        t = threading.Thread(target=self._play_sound_thread, daemon=True)
        t.start()


# =========================
# Eye prediction
# =========================
def predict_eye(model, eye_img):
    img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)

    preds      = model.predict(img, verbose=0)[0]
    class_id   = int(np.argmax(preds))
    label      = CLASSES[class_id]
    confidence = float(preds[class_id])
    return label, confidence


# =========================
# Get eye regions (cascade + geometric fallback)
# =========================
def get_eye_regions(eye_cascade, frame, gray, fx, fy, fw, fh):
    face_gray  = gray[fy:fy + fh, fx:fx + fw]
    face_color = frame[fy:fy + fh, fx:fx + fw]
    top_gray   = face_gray[:fh // 2, :]
    top_color  = face_color[:fh // 2, :]

    eyes = eye_cascade.detectMultiScale(
        top_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30)
    )

    if len(eyes) == 2:
        eyes = sorted(eyes, key=lambda e: e[0])
        results = []
        for (ex, ey, ew, eh) in eyes:
            crop   = top_color[ey:ey + eh, ex:ex + ew]
            abs_x1 = fx + ex
            abs_y1 = fy + ey
            abs_x2 = abs_x1 + ew
            abs_y2 = abs_y1 + eh
            results.append((crop, abs_x1, abs_y1, abs_x2, abs_y2))
        return results

    # Geometric fallback
    regions = [
        (int(fx + fw * 0.08), int(fy + fh * 0.20),
         int(fx + fw * 0.45), int(fy + fh * 0.50)),
        (int(fx + fw * 0.55), int(fy + fh * 0.20),
         int(fx + fw * 0.92), int(fy + fh * 0.50)),
    ]
    results = []
    for (x1, y1, x2, y2) in regions:
        crop = frame[y1:y2, x1:x2]
        results.append((crop, x1, y1, x2, y2))
    return results


# =========================
# Main
# =========================
def main():
    # ----- Load model -----
    model = tf.keras.models.load_model(MODEL_PATH)

    # ----- Detectors -----
    # Loaded from the XML files shipped in this repo rather than from
    # cv2.data.haarcascades, which OpenCV 5 no longer populates.
    if not hasattr(cv2, 'CascadeClassifier'):
        raise SystemExit(
            f"[ERROR] OpenCV {cv2.__version__} has no CascadeClassifier. "
            "OpenCV 5 dropped Haar cascades; install the 4.x line:\n"
            '           pip install "opencv-python>=4.8,<5"'
        )

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE)
    eye_cascade  = cv2.CascadeClassifier(EYE_CASCADE)
    for name, cascade in (('face', face_cascade), ('eye', eye_cascade)):
        if cascade.empty():
            raise SystemExit(f"[ERROR] Failed to load the {name} cascade XML.")

    # ----- State -----
    closed_frames = 0
    left_hist     = deque(maxlen=SMOOTH_WINDOW)
    right_hist    = deque(maxlen=SMOOTH_WINDOW)
    alarm         = AlarmPlayer()

    # ----- Webcam -----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit(
            "[ERROR] Cannot open camera (index 0). Check that a webcam is "
            "connected and that camera permission is granted."
        )

    print("[INFO] Running. Press Q to quit.")

    try:
        # =========================
        # Main loop
        # =========================
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, minNeighbors=5, minSize=(120, 120)
            )

            eye_labels = []

            for (fx, fy, fw, fh) in faces:
                cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (255, 255, 0), 2)

                eyes_data = get_eye_regions(eye_cascade, frame, gray, fx, fy, fw, fh)
                histories = [left_hist, right_hist]

                for i, (eye_img, x1, y1, x2, y2) in enumerate(eyes_data):
                    if eye_img is None or eye_img.size == 0:
                        continue

                    label, conf = predict_eye(model, eye_img)
                    histories[i].append(label)

                    smoothed = (
                        'close'
                        if histories[i].count('close') > SMOOTH_WINDOW // 2
                        else 'open'
                    )
                    eye_labels.append(smoothed)

                    color = (0, 255, 0) if smoothed == 'open' else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame, f"{smoothed} {conf:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
                    )

            # =========================
            # Drowsiness logic
            # =========================
            both_closed = len(eye_labels) >= 2 and all(l == 'close' for l in eye_labels)

            if both_closed:
                closed_frames += 1
            else:
                closed_frames = max(0, closed_frames - 1)

            # =========================
            # Alarm trigger
            # =========================
            is_drowsy = closed_frames >= CLOSED_THRESH
            if is_drowsy:
                alarm.trigger()   

            # =========================
            # Overlay
            # =========================
            if is_drowsy:
                # Flashing red banner when drowsy
                if int(time.time() * 2) % 2 == 0:           # blink every 0.5s
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (0, 0, 180), -1)
                cv2.putText(frame, "DROWSY! WAKE UP!", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 4)
            else:
                cv2.putText(frame, "Awake", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            cv2.putText(frame, f"Closed Frames: {closed_frames}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow("Driver Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # =========================
        # Cleanup (runs even if the loop raises)
        # =========================
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
