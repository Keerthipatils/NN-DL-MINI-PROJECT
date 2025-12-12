import tensorflow as tf
import cv2
import mediapipe as mp
import numpy as np
import matplotlib
import sklearn
import platform
import os

print("\n========== 🧠 ENVIRONMENT CHECK ==========\n")

# ================================
# 🧩 TensorFlow
# ================================
print("🔹 TensorFlow Version:", tf.__version__)

# Handle new TF/Keras versions safely
try:
    import keras
    print("🔹 Keras Version:", keras.__version__)
except Exception:
    print("🔹 Keras Version: (built-in in TensorFlow)")


# Check for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU Detected:", gpus[0])
else:
    print("⚠️ No GPU detected — running on CPU (still fine for this project).")

# Check ml-dtypes compatibility
try:
    import ml_dtypes
    print("🔹 ml-dtypes Version:", ml_dtypes.__version__)
except ImportError:
    print("⚠️ ml-dtypes not found — TensorFlow may fail if missing.")

# ================================
# 🎥 OpenCV
# ================================
print("\n🔹 OpenCV Version:", cv2.__version__)
cam = cv2.VideoCapture(0)
if cam.isOpened():
    print("✅ Webcam Detected.")
    cam.release()
else:
    print("⚠️ Webcam Not Accessible — check permissions or reconnect camera.")

# ================================
# 💡 Mediapipe
# ================================
print("\n🔹 Mediapipe Version:", mp.__version__)

# ================================
# 🧮 NumPy, Matplotlib, scikit-learn
# ================================
print("\n🔹 NumPy Version:", np.__version__)
print("🔹 Matplotlib Version:", matplotlib.__version__)
print("🔹 scikit-learn Version:", sklearn.__version__)

# ================================
# 🖥️ System Info
# ================================
print("\n========== 💻 SYSTEM INFO ==========")
print("OS:", platform.system(), platform.release())
print("Python Version:", platform.python_version())
print("Current Working Directory:", os.getcwd())

print("\n✅ Environment check completed successfully.\n")
print("If all versions are shown and no errors above, you’re ready to train and run live detection.")
