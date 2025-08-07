import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# -------------------------- Data Loading & Preprocessing --------------------------

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Reshape for CNN: (samples, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)

def preprocess_svhn_image(image):
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(gray, (3, 3), 0)
    thresh = cv.adaptiveThreshold(
        blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV, 11, 2
    )
    dilated = cv.dilate(thresh, np.ones((2, 2), np.uint8), iterations=1)
    resized = cv.resize(dilated, (28, 28), interpolation=cv.INTER_AREA)
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(28, 28, 1)

# ----------------------------- Model Building -----------------------------

def build_cnn_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ------------------------------ Training ------------------------------

def train_model(model, x_train, y_train):
    history = model.fit(x_train, y_train, epochs=6, validation_split=0.1)
    model.save("cnn_digit_model.h5")
    return history

# ------------------------------ Evaluation ------------------------------

def evaluate_model(model, x_test, y_test):
    loss, acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {acc:.4f}, Loss: {loss:.4f}")
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# --------------------------- Visualization ---------------------------

def plot_history(history):
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.show()

# ------------------------- Predict SVHN Digits -------------------------

def predict_svhn_samples(model, dataset, num_samples=5):
    for i in range(num_samples):
        item = dataset["test"][i]
        image = np.array(item["image"])
        true_label = item["label"]
        processed = preprocess_svhn_image(image)
        prediction = np.argmax(model.predict(processed.reshape(1, 28, 28, 1)))
        
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title(f"Original: {true_label}")
        plt.axis('off')
        
        plt.subplot(1,2,2)
        plt.imshow(processed.reshape(28, 28), cmap="gray")
        plt.title(f"Predicted: {prediction}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# -------------------- Predict from Custom Image --------------------

def predict_custom_digit(model, path):
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if image is None:
        print("❌ Image not found.")
        return
    image = cv.resize(image, (28, 28))
    image = 255 - image  # Invert for white background
    image = image.astype("float32") / 255.0
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Digit: {digit}")
    plt.axis('off')
    plt.show()

# ------------------------- Main Pipeline -------------------------

if __name__ == "__main__":
    print("🚀 Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    print("🔧 Building CNN model...")
    model = build_cnn_model()

    print("🎯 Training model...")
    history = train_model(model, x_train, y_train)

    print("📊 Evaluating model...")
    evaluate_model(model, x_test, y_test)

    print("📈 Plotting training history...")
    plot_history(history)

    print("📦 Loading SVHN dataset...")
    svhn = load_dataset("ufldl-stanford/svhn", "cropped_digits")

    print("🔍 Predicting on SVHN samples...")
   # predict_svhn_samples(model, svhn, num_samples=5)

    # Uncomment to test on your custom image
    predict_custom_digit(model, "C:/Users/Nidhi/OneDrive/Desktop/Digit recognition/4.png")
