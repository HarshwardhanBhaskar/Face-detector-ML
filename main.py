import cv2
import os
import time
import csv
import numpy as np
from datetime import datetime
from deepface import DeepFace
import mediapipe as mp
from twilio.rest import Client
from scipy.spatial.distance import cosine

# Twilio Configuration
import os
TWILIO_SID = os.getenv("TWILIO_SID", "your_twilio_sid_here")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "your_twilio_auth_token_here")
TWILIO_PHONE = "+1 831 603 6870"
ALERT_PHONE = "+91 8651649921"

client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)

def send_alert_sms(message):
    try:
        client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=ALERT_PHONE
        )
    except Exception as e:
        print("Twilio Error:", e)

#  Logging
def log_event(name, event):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {event}: {name}")
    with open("log.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name, event])

# --- Setup ---
if not os.path.exists("unknown_faces"):
    os.makedirs("unknown_faces")

if not os.path.exists("log.csv"):
    with open("log.csv", "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Name", "Event"])

# State
inside_people = {}  # name and last seen timestamp
UNKNOWN_EMBEDDINGS = []
UNKNOWN_COUNTER = 0
INTRUDER_ALERT_COOLDOWN = 60
last_alert_time = 0
EXIT_TIMEOUT = 4

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam...")

def is_new_unknown_face(embedding, threshold=0.45):
    for known_emb in UNKNOWN_EMBEDDINGS:
        dist = cosine(embedding, known_emb)
        if dist < threshold:
            return False
    return True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    current_time = time.time()
    detected_names = set()

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            w, h = int(bboxC.width * iw), int(bboxC.height * ih)
            x = max(0, min(x, iw - 1))
            y = max(0, min(y, ih - 1))
            w = min(w, iw - x)
            h = min(h, ih - y)
            face_crop = frame[y:y+h, x:x+w]

            name = "Unknown"  # default name
            try:
                embedding = DeepFace.represent(face_crop, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]

                result = DeepFace.find(
                    img_path=face_crop,
                    db_path="known_faces",
                    enforce_detection=False,
                    model_name="VGG-Face"
                )

                if len(result) > 0 and result[0].shape[0] > 0:
                    identity_path = result[0].iloc[0]['identity']
                    name = os.path.basename(identity_path).split('.')[0]
                    detected_names.add(name)

                    if name not in inside_people:
                        log_event(name, "Entry")
                    inside_people[name] = current_time

                else:
                    if is_new_unknown_face(embedding):
                        UNKNOWN_EMBEDDINGS.append(embedding)

                        intruder_id = f"unknown_{int(current_time)}"
                        cv2.imwrite(f"unknown_faces/{intruder_id}.jpg", face_crop)

                        name = f"Unknown_{UNKNOWN_COUNTER}"
                        UNKNOWN_COUNTER += 1

                        if current_time - last_alert_time > INTRUDER_ALERT_COOLDOWN:
                            log_event("UNKNOWN", "Intruder Detected")
                            send_alert_sms("🚨 ALERT: Unknown person detected entering WRM Mill!")
                            last_alert_time = current_time

                        detected_names.add(name)
                    else:
                        name = "Unknown"
                        detected_names.add(name)

            except Exception as e:
                print("Error in face recognition:", e)
                name = "Unknown"
                detected_names.add(name)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Check for exits
    to_remove = []
    for person, last_seen in inside_people.items():
        if current_time - last_seen > EXIT_TIMEOUT:
            log_event(person, "Exit")
            to_remove.append(person)
    for person in to_remove:
        inside_people.pop(person)

    cv2.putText(frame, f"People Inside: {len(inside_people)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("WRM Entry Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
