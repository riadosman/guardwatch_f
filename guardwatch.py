from datetime import datetime
from ultralytics import YOLO
import json
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import pygame
import threading
import os
import logging

from signals import ear_hesapla, upscale_roi, PerclosBuffer, compute_head_pose, GlobalCalibrator


CONFIG_FILE_NAME = "config.json"

takip_listesi = {}
sonraki_id = 0


with open(CONFIG_FILE_NAME, "r") as dosya:
    ayarlar = json.load(dosya)


class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        try:
            os.fsync(self.stream.fileno())
        except (OSError, ValueError):
            pass


_log_formatter = logging.Formatter("%(asctime)s | %(message)s")
_file_handler = FlushingFileHandler(ayarlar["log_dosyasi"], encoding="utf-8")
_file_handler.setFormatter(_log_formatter)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_log_formatter)

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_root_logger.addHandler(_file_handler)
_root_logger.addHandler(_console_handler)

DURUM_ONCELIK = {
    "NORMAL":     0,
    "HAREKETSIZ": 1,
    "GOZ_KAPALI": 2,
    "UYUYOR":     3,
}

DURUM_RENK = {
    "NORMAL":     (0,   200,   0),
    "HAREKETSIZ": (0,   165, 255),
    "GOZ_KAPALI": (0,   100, 255),
    "UYUYOR":     (0,     0, 255),
}

def set_durum_if_higher(state, new_durum):
    if DURUM_ONCELIK[new_durum] > DURUM_ONCELIK[state["durum"]]:
        state["durum"]      = new_durum
        state["durum_renk"] = DURUM_RENK[new_durum]

def oklid_mesafe(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def merkez_noktasi(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def id_to_color(id_):
    return (
        (id_ * ayarlar["redcolor_multiplier"])   % ayarlar["color_modulo"],
        (id_ * ayarlar["greencolor_multiplier"]) % ayarlar["color_modulo"],
        (id_ * ayarlar["bluecolor_multiplier"])  % ayarlar["color_modulo"]
    )

def extract_roi(frame, x1, y1, x2, y2, padding=0):
    h_frame, w_frame = frame.shape[:2]
    x1p = max(0,       x1 - padding)
    y1p = max(0,       y1 - padding)
    x2p = min(w_frame, x2 + padding)
    y2p = min(h_frame, y2 + padding)
    if x2p <= x1p or y2p <= y1p:
        return None, (x1p, y1p, x2p, y2p)
    return frame[y1p:y2p, x1p:x2p], (x1p, y1p, x2p, y2p)


class RTSPGrabber:
    def __init__(self, source, warmup_sec=3.0):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self._lock = threading.Lock()
        self._latest = None
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

        baslangic = time.time()
        while time.time() - baslangic < warmup_sec:
            with self._lock:
                if self._latest is not None:
                    return
            time.sleep(0.05)

    def _reader(self):
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self._lock:
                self._latest = frame

    def isOpened(self):
        return self.cap.isOpened()

    def read(self):
        with self._lock:
            frame = self._latest
        if frame is None:
            return False, None
        return True, frame

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()

def alarm_cal():
    if pygame.mixer.music.get_busy():
        return
    pygame.mixer.music.play()

def frame_kaydet(frame,id_, state):
    zaman = datetime.now().strftime(ayarlar["zaman_formati"]) # "%Y-%m-%d"
    klasor = os.path.join("kayitlar", zaman)
    os.makedirs(klasor, exist_ok=True)
    frame_yolu = os.path.join("kayitlar", zaman, f"ihlal_{id_}_{state['durum']}.jpg")
    result = cv2.imwrite(frame_yolu, frame)
    if result:
        print(f"Frame kaydedildi: {frame_yolu}")
    else:
        print(f"Frame kaydedilemedi: {frame_yolu}")



model     = YOLO(ayarlar["Yolo_modeli"])

kamera_kaynagi = ayarlar["kamera_id"]
if isinstance(kamera_kaynagi, str):
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = RTSPGrabber(kamera_kaynagi)
else:
    cap = cv2.VideoCapture(kamera_kaynagi)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    logging.error(f"Kamera acilamadi: {kamera_kaynagi}")
    raise RuntimeError(f"Kamera acilamadi: {kamera_kaynagi}")

genislik = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
yukseklik = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
logging.info(f"Kamera baglandi: {kamera_kaynagi} ({genislik}x{yukseklik})")

pygame.mixer.init()
pygame.mixer.music.load(ayarlar["sound_file"])

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=ayarlar["face_landmarker_path"]),
    running_mode=vision.RunningMode.IMAGE,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

calibrator = GlobalCalibrator(ayarlar)

ROI_PADDING = ayarlar.get("roi_padding", 10)

# HAFTA 2 GUN 5
if not os.path.exists("kayitlar"):
    os.mkdir("kayitlar")

logging.info(f"Program basladi. Kayitlar klasoru olusturuldu: {ayarlar['log_dosyasi']}")
while True:
    
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=ayarlar["yolo_confidence"], classes=[ayarlar["yolo_class_id"]])

    detections = []
    
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            detections.append((x1, y1, x2, y2))
            


    new_tracks     = {}
    kullanilan_idler = set()

    for det in detections:
        x1, y1, x2, y2 = det
        cx, cy = map(int, merkez_noktasi((x1, y1), (x2, y2)))

        min_mesafe = float("inf")
        matched_id = None

        for id_, state in takip_listesi.items():
            if id_ in kullanilan_idler:
                continue
            mesafe = oklid_mesafe(state["merkez"], (cx, cy))
            oran = genislik / ayarlar["referans_genislik"]
            max_mesafe = int(ayarlar["referans_mesafe"] * oran)
            if mesafe < min_mesafe and mesafe < max_mesafe:
                min_mesafe = mesafe
                matched_id = id_

        if matched_id is not None:
            kullanilan_idler.add(matched_id)
        else:
            matched_id  = sonraki_id
            sonraki_id += 1
            logging.info(f"YENI TAKIP BASLADI | ID: {matched_id}")

        prev_state = takip_listesi.get(matched_id, {})

        new_tracks[matched_id] = {
            "kutu":             (x1, y1, x2, y2),
            "merkez":           (cx, cy),
            "prev_center":      prev_state.get("merkez", (cx, cy)),
            "eye_closed_start": prev_state.get("eye_closed_start"),
            "not_moving_start": prev_state.get("not_moving_start"),
            "durum":            "NORMAL",
            "prev_durum":       prev_state.get("durum", "NORMAL"),
            "durum_renk":       DURUM_RENK["NORMAL"],
            "perclos_buffer":   prev_state.get("perclos_buffer", PerclosBuffer(ayarlar["perclos_window_sec"])),
            "perclos":          prev_state.get("perclos", 0.0),
            "pitch":            prev_state.get("pitch", 0.0),
        }

    for eski_id in takip_listesi.keys() - new_tracks.keys():
        logging.info(f"TAKIP KAYBEDILDI | ID: {eski_id}")

    takip_listesi = new_tracks

    for id_, state in takip_listesi.items():

        x1, y1, x2, y2 = state["kutu"]

        roi, (rx1, ry1, rx2, ry2) = extract_roi(frame, x1, y1, x2, y2, padding=ROI_PADDING)
        if roi is None or roi.size == 0:
            continue

        roi_for_mp = upscale_roi(roi, ayarlar["mediapipe_min_size"])
        h_roi, w_roi = roi_for_mp.shape[:2]

        dist = oklid_mesafe(state["prev_center"], state["merkez"])

        if dist < ayarlar["hareket_piksel_esigi"]:
            if state["not_moving_start"] is None:
                state["not_moving_start"] = time.time()
            else:
                elapsed = time.time() - state["not_moving_start"]
                if elapsed >= ayarlar["hareketsizlik_limit_sn"] and state["durum"] != "HAREKETSIZ":
                    set_durum_if_higher(state, "HAREKETSIZ")
                    t = threading.Thread(target=frame_kaydet, args=(roi.copy(),id_, state))
                    t.start()
        else:
            state["not_moving_start"] = None

        state["prev_center"] = state["merkez"]

        rgb_roi      = cv2.cvtColor(roi_for_mp, cv2.COLOR_BGR2RGB)
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
        face_results = landmarker.detect(mp_image)

        if face_results.face_landmarks:
            for face_landmarks in face_results.face_landmarks:

                left_eye_landmarks  = []
                right_eye_landmarks = []

                for i in ayarlar["LEFT_EYE_IDX"]:
                    lm = face_landmarks[i]
                    left_eye_landmarks.append((int(lm.x * w_roi), int(lm.y * h_roi)))

                for i in ayarlar["RIGHT_EYE_IDX"]:
                    lm = face_landmarks[i]
                    right_eye_landmarks.append((int(lm.x * w_roi), int(lm.y * h_roi)))

                ear_value = ear_hesapla(left_eye_landmarks, right_eye_landmarks)

                calibrator.update(ear_value)
                ear_threshold = calibrator.ear_closed()

                now_ts = time.time()
                state["perclos_buffer"].add(now_ts, ear_value < ear_threshold)
                state["perclos"] = state["perclos_buffer"].percent(now_ts)

                pitch, roll, yaw = compute_head_pose(face_landmarks, (h_roi, w_roi))
                state["pitch"] = pitch

                eye_closed = ear_value < ear_threshold
                head_down  = pitch > ayarlar["head_pitch_drowsy"]
                if eye_closed or head_down:
                    tag = "GOZ" if eye_closed else "BAS_DUSUK"
                    if state["eye_closed_start"] is None:
                        state["eye_closed_start"] = time.time()
                    else:
                        eye_elapsed = time.time() - state["eye_closed_start"]

                        if eye_elapsed >= ayarlar["goz_kapali_limit_sn"] and state["durum"] != "GOZ_KAPALI":
                            set_durum_if_higher(state, "GOZ_KAPALI")
                            t = threading.Thread(target=frame_kaydet, args=(roi.copy(),id_, state))
                            t.start()

                            if state["not_moving_start"] is not None:
                                move_elapsed = time.time() - state["not_moving_start"]
                                if move_elapsed >= ayarlar["hareketsizlik_limit_sn"]:
                                    set_durum_if_higher(state, "UYUYOR")
                                    alarm_cal()
                                    t = threading.Thread(target=frame_kaydet, args=(roi.copy(),id_, state))
                                    t.start()
                else:
                    state["eye_closed_start"] = None

        if state["durum"] != state.get("prev_durum", "NORMAL"):
            if state["durum"] == "NORMAL":
                logging.info(f"IHLAL BITTI | ID: {id_}")
            else:
                logging.info(f"{state['durum']} IHLALI BASLADI | ID: {id_}")

        state["prev_durum"] = state["durum"]
        renk = id_to_color(id_)
        cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
        cv2.putText(frame, f"ID: {id_}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, ayarlar["fontScale"], renk, 2)
        cv2.putText(frame, state["durum"], (x2 - 120, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, ayarlar["fontScale"], state["durum_renk"], 2)
        cv2.putText(frame, f"PCL:{state['perclos']:.0f}% P:{state['pitch']:.0f}",
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, ayarlar["fontScale"] * 0.7,
                    state["durum_renk"], 1)
        

    cv2.imshow("Guard Watch", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 