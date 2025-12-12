# =====================================================
# Real-time Mask Detection using Mediapipe FaceMesh + MobileNetV2
# =====================================================

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import os

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "../saved_models/mask_detector_mobilenet_optimized.h5"
IMG_SIZE = 128
CONFIDENCE_THRESHOLD = 0.75
SHOW_FACE_WINDOW = False

labels = ["Mask Worn Incorrectly", "With Mask", "Without Mask"]

# ===============================
# LOAD MODEL
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}.")
model = load_model(MODEL_PATH)
print("✅ Loaded model:", MODEL_PATH)

# ===============================
# MEDIAPIPE FACE MESH SETUP
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# ===============================
# START CAPTURE
# ===============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam.")

print("🎥 Starting Mediapipe FaceMesh-based mask detection. Press 'q' to quit.")

fps_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.convertScaleAbs(frame, alpha=1.3, beta=25)

    results = face_mesh.process(frame_rgb)
    ih, iw = frame.shape[:2]

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get all landmark points
            xs = [int(p.x * iw) for p in face_landmarks.landmark]
            ys = [int(p.y * ih) for p in face_landmarks.landmark]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            # Add margin
            margin = 20
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(iw, x2 + margin), min(ih, y2 + margin)

            # Extract face region
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocess for model
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_input = face_resized.astype("float32") / 255.0
            face_input = np.expand_dims(face_input, axis=0)

            # Predict mask status
            pred = model.predict(face_input, verbose=0)[0]
            max_conf = float(np.max(pred))
            pred_index = int(np.argmax(pred))

            if max_conf < CONFIDENCE_THRESHOLD:
                label_text = f"Uncertain ({round(max_conf*100,2)}%)"
                box_color = (255, 255, 255)
            else:
                label_text = f"{labels[pred_index]} ({round(max_conf*100,2)}%)"
                if labels[pred_index] == "With Mask":
                    box_color = (0, 255, 0)
                elif labels[pred_index] == "Without Mask":
                    box_color = (0, 0, 255)
                else:
                    box_color = (0, 165, 255)

            # Draw bounding box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            # Optional debug
            if SHOW_FACE_WINDOW:
                cv2.imshow("FaceCrop", face_resized)

            # Optional: draw landmarks
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)
            )

    # FPS counter
    fps = 1.0 / (time.time() - fps_time + 1e-9)
    fps_time = time.time()
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Overlay title
    cv2.putText(frame, "AI Mask Detection (FaceMesh)", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Mask Detection - Press 'q' to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
print("🛑 Stopped.")
