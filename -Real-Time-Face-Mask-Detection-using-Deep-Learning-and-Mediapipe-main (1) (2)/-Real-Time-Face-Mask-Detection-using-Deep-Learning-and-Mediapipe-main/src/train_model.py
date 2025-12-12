# =====================================================
# Mask Detection Model - MobileNetV2 Fine-tuned + Class Weights
# =====================================================
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt
import os

# ===============================
# CONFIGURATION
# ===============================
DATASET_DIR = "../dataset_sorted"
MODEL_PATH = "../saved_models/mask_detector_mobilenet_optimized.h5"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
FINE_TUNE_EPOCHS = 10  # more fine-tuning
FINE_TUNE_LAYERS = 50  # unfreeze deeper layers

# ===============================
# DATA AUGMENTATION (STRONG)
# ===============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    shear_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.6, 1.4),
    horizontal_flip=True,
    validation_split=0.1
)

train_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("\n✅ Classes found:", train_data.class_indices)

# ===============================
# COMPUTE CLASS WEIGHTS
# ===============================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))
print("\n⚖️ Class Weights:", class_weights)

# ===============================
# BUILD BASE MODEL
# ===============================
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freeze all layers first
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# ===============================
# COMPILE MODEL (PHASE 1)
# ===============================
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print("\n🧠 Phase 1: Training top layers only...")
history1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

# ===============================
# FINE-TUNING (PHASE 2)
# ===============================
for layer in base_model.layers[-FINE_TUNE_LAYERS:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print(f"\n🎯 Phase 2: Fine-tuning last {FINE_TUNE_LAYERS} layers...")
history2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=FINE_TUNE_EPOCHS,
    class_weight=class_weights
)

# Combine histories
history = {}
for key in history1.history.keys():
    history[key] = history1.history[key] + history2.history[key]

# ===============================
# SAVE MODEL
# ===============================
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
model.save(MODEL_PATH)
print(f"\n✅ Model saved successfully at: {MODEL_PATH}")

# ===============================
# PLOT TRAINING RESULTS
# ===============================
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()

os.makedirs("../results", exist_ok=True)
plt.savefig("../results/training_performance_optimized.png")
plt.show()
