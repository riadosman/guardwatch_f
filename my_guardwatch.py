from ultralytics import YOLO 
import json 
import mediapipe as mp 
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision 
import cv2 
import time 
import pygame 

CONFIG_FILE_NAME = "config.json" 

takip_listesi = {}
sonraki_id = 0

with open(CONFIG_FILE_NAME, "r") as dosya: 
    ayarlar = json.load(dosya) 

def oklid_mesafe(p1, p2): 
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5 

def merkez_noktasi(p1, p2): 
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2) 

def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB-xA) * max(0, yB-yA)
    box1Area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2Area = (box2[2]-box2[0]) * (box2[3]-box2[1])

    return interArea / (box1Area + box2Area - interArea + 1e-6)

def ear_hesapla(e1, e2): 
    p11, p12, p13, p14, p15, p16 = e1 
    p21, p22, p23, p24, p25, p26 = e2 

    e1_ear = (oklid_mesafe(p12, p16) + oklid_mesafe(p13, p15)) / (2 * oklid_mesafe(p11, p14)) 
    e2_ear = (oklid_mesafe(p22, p26) + oklid_mesafe(p23, p25)) / (2 * oklid_mesafe(p21, p24)) 

    return (e1_ear + e2_ear) / 2.0 

def id_to_color(id_): 
    return ( 
        (id_ * ayarlar["redcolor_multiplier"]) % ayarlar["color_modulo"], 
        (id_ * ayarlar["greencolor_multiplier"]) % ayarlar["color_modulo"], 
        (id_ * ayarlar["bluecolor_multiplier"]) % ayarlar["color_modulo"] 
    ) 

model = YOLO(ayarlar["Yolo_modeli"]) 
cap = cv2.VideoCapture(ayarlar["kamera_id"]) 

pygame.mixer.init() 
pygame.mixer.music.load(ayarlar["sound_file"]) 

options = vision.FaceLandmarkerOptions( 
   base_options = python.BaseOptions(model_asset_path=ayarlar["face_landmarker_path"]), 
   running_mode = vision.RunningMode.IMAGE, 
) 

landmarker = vision.FaceLandmarker.create_from_options(options) 

while True: 
    info = [0, "NORMAL", ayarlar["green_color"]] 
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

    new_tracks = {}
    used_dets = set()

    for det in detections:
        best_iou = 0
        matched_id = None

        for id_, state in takip_listesi.items():
            score = iou(state["kutu"], det)
            if score > best_iou and score > 0.3:
                best_iou = score
                matched_id = id_

        if matched_id is None:
            matched_id = sonraki_id
            sonraki_id += 1

        x1, y1, x2, y2 = det
        cx, cy = merkez_noktasi((x1, y1), (x2, y2))
        cx, cy = int(cx), int(cy)

        prev_state = takip_listesi.get(matched_id, {})

        new_tracks[matched_id] = {
            "kutu": (x1, y1, x2, y2),
            "merkez": (cx, cy),
            "prev_center": prev_state.get("merkez", (cx, cy)),
            "eye_closed_start": prev_state.get("eye_closed_start"),
            "not_moving_start": prev_state.get("not_moving_start"),
        }

    takip_listesi = new_tracks

    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    # mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame) 
    # results = landmarker.detect(mp_image) 
    # height, width, _ = frame.shape 


    for id_, state in takip_listesi.items():


        x1, y1, x2, y2 = state["kutu"]
        cx, cy = state["merkez"]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)

        face_results = landmarker.detect(mp_image)

        h, w, _ = crop.shape
        renk = id_to_color(id_)

        cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2) 
        cv2.putText(frame, f"ID: {id_}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, ayarlar["fontScale"], renk, 2) 

        dist = oklid_mesafe(state["prev_center"], state["merkez"])

        if dist < ayarlar["hareket_piksel_esigi"]:
            if state["not_moving_start"] is None:
                state["not_moving_start"] = time.time()
            else:
                elapsed = time.time() - state["not_moving_start"]
                if elapsed >= ayarlar["hareketsizlik_limit_sn"]:
                    info = [1, "HAREKETSIZLIK TESPIT EDILDI!", ayarlar["warning_color"]]
        else:
            state["not_moving_start"] = None

        state["prev_center"] = state["merkez"]

        if results.face_landmarks:
            left_eye_landmarks = [] 
            right_eye_landmarks = [] 

            for face_landmarks in results.face_landmarks: 
                for i in ayarlar["LEFT_EYE_IDX"]: 
                    lm = face_landmarks[i]  
                    mx, my = int(lm.x * width), int(lm.y * height) 
                    left_eye_landmarks.append((mx, my)) 
                 
                for i in ayarlar["RIGHT_EYE_IDX"]: 
                    lm = face_landmarks[i] 
                    mx, my = int(lm.x * width), int(lm.y * height) 
                    right_eye_landmarks.append((mx, my)) 

            ear_value = ear_hesapla(left_eye_landmarks, right_eye_landmarks) 

            if ear_value < ayarlar["ear_threshold"]: 
                if state["eye_closed_start"] is None: 
                    state["eye_closed_start"] = time.time() 
                else: 
                    elapsed = time.time() - state["eye_closed_start"] 
                    if elapsed >= ayarlar["goz_kapali_limit_sn"]: 
                        info = [2, "GOZ KAPALI", ayarlar["warning_color"]] 
            else:
                state["eye_closed_start"] = None

        # if face_results.face_landmarks:
        #     left_eye_landmarks = []
        #     right_eye_landmarks = []

        #     for face_landmarks in face_results.face_landmarks:
        #         for i in ayarlar["LEFT_EYE_IDX"]:
        #             lm = face_landmarks[i]
        #             mx, my = int(lm.x * w), int(lm.y * h)
        #             left_eye_landmarks.append((mx, my))

        #         for i in ayarlar["RIGHT_EYE_IDX"]:
        #             lm = face_landmarks[i]
        #             mx, my = int(lm.x * w), int(lm.y * h)
        #             right_eye_landmarks.append((mx, my))

        #     ear_value = ear_hesapla(left_eye_landmarks, right_eye_landmarks)

        #     if ear_value < ayarlar["ear_threshold"]:
        #         if state["eye_closed_start"] is None:
        #             state["eye_closed_start"] = time.time()
        #         else:
        #             elapsed = time.time() - state["eye_closed_start"]

        #             if elapsed >= ayarlar["goz_kapali_limit_sn"]:
        #                 info = [2, "GOZ KAPALI", ayarlar["warning_color"]]
        #     else:
        #         state["eye_closed_start"] = None


    cv2.putText(frame, info[1], (30, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, ayarlar["fontScale"], info[2], ayarlar["text_thickness"]) 

    cv2.imshow("Guard Watch", frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
