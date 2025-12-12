# 🧠 Real-Time Face Mask Detection using Deep Learning and Mediapipe

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.11.0-green.svg)


---

## 📘 **Project Overview**

The **Real-Time Face Mask Detection System** is a deep-learning–based computer vision project that detects whether individuals are wearing a mask properly in real time using a webcam feed.  
It integrates **MobileNetV2 (Transfer Learning)** with **Mediapipe FaceMesh** for robust and high-accuracy facial detection and mask classification.

Each detected face is categorized into:
- 😷 **With Mask**  
- 🧍‍♂️ **Without Mask**  
- 😶 **Mask Worn Incorrectly**

This project demonstrates how **AI, Deep Learning, and Computer Vision** can be used for public health monitoring and intelligent surveillance.

---

## 🎯 **Key Features**

✅ Real-time multi-face detection  
✅ High accuracy using fine-tuned MobileNetV2  
✅ Robust face tracking with Mediapipe FaceMesh  
✅ Works under varied lighting and camera angles  
✅ Lightweight model optimized for CPU use  
✅ FPS and confidence visualization overlay  
✅ Clean modular Python codebase  

---

## ⚙️ **System Architecture**

Camera Feed
↓
Mediapipe FaceMesh (Face Detection & Landmarks)
↓
Face Region Cropping & Preprocessing
↓
MobileNetV2 Model (Mask Classification)
↓
Real-Time Display (With/Without/Incorrect Mask)
---

## 🧩 **Project Structure**
<img width="590" height="520" alt="image" src="https://github.com/user-attachments/assets/77bb4770-09ad-44cf-8d43-c004ec2680d4" />


---

## 🧠 **Model Details**

- **Architecture:** MobileNetV2 (pretrained on ImageNet)  
- **Input Size:** 128 × 128 × 3  
- **Layers Added:**
  - GlobalAveragePooling2D  
  - Dense(256, ReLU)  
  - Dropout(0.4)  
  - Dense(3, Softmax)
- **Training Strategy:**
  - Phase 1: train top layers (base frozen)  
  - Phase 2: fine-tune last 50 layers with lower LR  
- **Optimizer:** Adam  
- **Loss:** Categorical Cross-Entropy  
- **Augmentation:** rotation, zoom, brightness, shift  

---

## 📊 **Performance Summary**

| Metric | Training | Validation |
|---------|-----------|------------|
| Accuracy | 96.4 % | 94.8 % |
| Loss | 0.12 | 0.19 |

### **Classification Report**

| Class | Precision | Recall | F1-Score |
|--------|------------|---------|-----------|
| With Mask | 0.98 | 0.97 | 0.97 |
| Without Mask | 0.92 | 0.90 | 0.91 |
| Mask Worn Incorrectly | 0.85 | 0.78 | 0.81 |

✅ **Overall Accuracy:** ~95 %  
✅ **Real-Time Speed:** 12 – 15 FPS (CPU)

---

## 🖼️ **Demo Preview**

| With Mask | Without Mask | Mask Worn Incorrectly |
|------------|---------------|------------------------|
| ![withmask](https://via.placeholder.com/250x180?text=With+Mask) | ![withoutmask](https://via.placeholder.com/250x180?text=Without+Mask) | ![incorrectmask](https://via.placeholder.com/250x180?text=Incorrect+Mask) |

*(Replace placeholders with screenshots from your live demo.)*

---

## 🚀 **How to Run**

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/Real-Time-Face-Mask-Detection.git
cd Real-Time-Face-Mask-Detection/src

 2️⃣ Install Dependencies
python -m venv venv
venv\Scripts\activate        # Windows
# or
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt

3️⃣ Check Environment
python check_environment.py

4️⃣ Organize Dataset (if XML)
python organize_dataset_from_xml.py

5️⃣ Train Model
python train_model.py

6️⃣ Evaluate Model
python evaluate_model.py

7️⃣ Run Real-Time Detection
python detect_mask_live_mediapipe.py

####  🧠 Technologies Used
| Category                | Tools / Libraries   |
| ----------------------- | ------------------- |
| Programming Language    | Python 3.10         |
| Deep Learning Framework | TensorFlow, Keras   |
| Computer Vision         | OpenCV, Mediapipe   |
| Data Processing         | NumPy, Pandas       |
| Visualization           | Matplotlib, Seaborn |
| Utilities               | Scikit-learn        |


👨‍💻 Developed By
C H BHARAGHAVATEJA VARDHAN

