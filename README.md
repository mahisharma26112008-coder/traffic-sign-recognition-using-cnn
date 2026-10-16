# traffic-sign-recognition-using-cnn

1. Introduction
Traffic Sign Recognition Using CNN is a Deep Learning project that identifies and classifies traffic signs from images. It uses a Convolutional Neural Network (CNN) to recognize signs such as Stop, No Entry, Speed Limit, and Turn Left.
This system helps self-driving cars and Advanced Driver Assistance Systems (ADAS) understand road signs and improve road safety.
2. Objective
The main objectives of this project are:
- To recognize traffic signs automatically from images.
- To classify traffic signs into their correct categories.
- To use CNN for accurate image classification.
- To support intelligent transportation systems and autonomous vehicles.
3. What is CNN?
CNN stands for Convolutional Neural Network. It is a Deep Learning algorithm mainly used for image processing and recognition.
CNN automatically extracts important features from images, such as edges, shapes, colors, and patterns, and uses them to classify images into different categories.
4. Working of the Project
1. Input Image
A traffic sign image is provided to the system.

2. Preprocessing
Images are resized, normalized, and prepared for training.

3. CNN Model
Convolution, activation, and pooling layers extract image features.

4. Classification
The model predicts the traffic sign category.

5. Output
Example: Stop Sign, Speed Limit, or No Entry.


5. Technologies Used
- Python: Programming language used to develop the project.
- TensorFlow/Keras: Libraries used to build and train the CNN model.
- CNN: Deep Learning algorithm used for image classification.
- NumPy: Used for numerical operations and array handling.
- Matplotlib: Used to display images and training graphs.
- OpenCV: Can be used for image processing.
6. Dataset
The project can use the German Traffic Sign Recognition Benchmark (GTSRB) dataset. It contains images of different traffic sign categories.
The dataset is divided into:
- Training dataset: Used to train the CNN model.
- Validation dataset: Used to tune and evaluate the model during training.
- Testing dataset: Used to check the model's performance on unseen images.
7. Main CNN Layers
Layer	Function
Convolutional Layer	Extracts features such as edges and shapes.
ReLU Activation	Introduces non-linearity into the model.
Pooling Layer	Reduces the size of feature maps.
Flatten Layer	Converts feature maps into a one-dimensional vector.
Dense Layer	Learns patterns for classification.
Softmax Layer	Produces class probabilities for the output categories.
8. Applications
1. Self-driving cars.
2. Driver assistance systems.
3. Intelligent transportation systems.
4. Road safety monitoring.
5. Smart traffic management.
9. Advantages
- Automatically recognizes traffic signs.
- Can classify multiple types of traffic signs.
- Learns image features automatically.
- Can help improve road safety.
- Can be integrated into intelligent vehicle systems.
10. Limitations
- Requires a suitable dataset for training.
- Performance may decrease in poor lighting or bad weather.
- Blurred or damaged signs may be difficult to recognize.
- Training may require significant computing resources for large datasets.
11. Conclusion
Traffic Sign Recognition Using CNN is a Deep Learning project that uses image classification to identify different traffic signs. The CNN model learns visual features from training images and predicts the correct traffic sign category for new images. This technology can support autonomous driving, driver assistance systems, and safer transportation.

"""
Traffic Sign Recognition with a CNN (Keras / TensorFlow)
Dataset: GTSRB (German Traffic Sign Recognition Benchmark), 43 classes
Kaggle: https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign

Expected folder layout:
    gtsrb/
        Train/0/*.png ... Train/42/*.png
        Test/*.png
        Test.csv        (columns: Path, ClassId, ...)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ------------------------- Config -------------------------
DATA_DIR = "gtsrb"
IMG_SIZE = 32
NUM_CLASSES = 43
BATCH_SIZE = 64
EPOCHS = 30

# ------------------------- Load training data -------------------------
def load_train(data_dir):
    images, labels = [], []
    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(data_dir, "Train", str(class_id))
        for fname in os.listdir(class_dir):
            try:
                img = Image.open(os.path.join(class_dir, fname)).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img))
                labels.append(class_id)
            except Exception as e:
                print(f"Skipping {fname}: {e}")
    return np.array(images, dtype="float32") / 255.0, np.array(labels)


def load_test(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "Test.csv"))
    images = []
    for path in df["Path"]:
        img = Image.open(os.path.join(data_dir, path)).convert("RGB")
        images.append(np.array(img.resize((IMG_SIZE, IMG_SIZE))))
    return np.array(images, dtype="float32") / 255.0, df["ClassId"].values


print("Loading data...")
X, y = load_train(DATA_DIR)
X_test, y_test = load_test(DATA_DIR)
print("Train:", X.shape, "Test:", X_test.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

y_train_cat = to_categorical(y_train, NUM_CLASSES)
y_val_cat = to_categorical(y_val, NUM_CLASSES)
y_test_cat = to_categorical(y_test, NUM_CLASSES)

# ------------------------- Model -------------------------
# No horizontal flip in augmentation: it would change the meaning of
# signs such as left/right turn arrows.
augment = tf.keras.Sequential([
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
])

def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        augment,

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_model()
model.summary()

# ------------------------- Train -------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ModelCheckpoint("traffic_sign_cnn.keras", save_best_only=True, monitor="val_accuracy"),
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
)

# ------------------------- Evaluate -------------------------
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

y_pred = np.argmax(model.predict(X_test), axis=1)
print(classification_report(y_test, y_pred))

# Training curves
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="val")
ax[0].set_title("Accuracy"); ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="val")
ax[1].set_title("Loss"); ax[1].legend()
plt.tight_layout()
plt.savefig("training_curves.png")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix"); plt.colorbar()
plt.xlabel("Predicted"); plt.ylabel("True")
plt.savefig("confusion_matrix.png")

# ------------------------- Predict a single image -------------------------
def predict_sign(image_path, model_path="traffic_sign_cnn.keras"):
    m = tf.keras.models.load_model(model_path)
    img = Image.open(image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img, dtype="float32") / 255.0, axis=0)
    probs = m.predict(arr)[0]
    class_id = int(np.argmax(probs))
    return class_id, float(probs[class_id])

# Example:
# cid, conf = predict_sign("gtsrb/Test/00001.png")
# print(CLASS_NAMES[cid], conf)
