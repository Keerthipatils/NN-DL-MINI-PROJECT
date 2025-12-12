import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = tf.keras.models.load_model("../saved_models/mask_detector_mobilenet_finetuned.h5")

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_data = test_datagen.flow_from_directory(
    "../dataset_sorted",
    target_size=(128,128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Evaluate
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)

print("\nClassification Report:\n")
print(classification_report(test_data.classes, y_pred, target_names=list(test_data.class_indices.keys())))

# Confusion Matrix
cm = confusion_matrix(test_data.classes, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_data.class_indices.keys(),
            yticklabels=test_data.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("../results/confusion_matrix.png")
plt.show()
