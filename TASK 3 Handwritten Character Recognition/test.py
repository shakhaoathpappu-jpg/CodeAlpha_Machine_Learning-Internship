import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ------------------------------------
# Load Trained Model
# ------------------------------------
print("=" * 50)
print("Loading Trained Model...")
print("=" * 50)

model = load_model("saved_model/cnn_model.keras")

print("Model Loaded Successfully!\n")

# ------------------------------------
# Load MNIST Test Dataset
# ------------------------------------
(_, _), (X_test, y_test) = mnist.load_data()

# Normalize
X_test = X_test.astype("float32") / 255.0

# Reshape
X_test = X_test.reshape(-1, 28, 28, 1)

print(f"Testing Images : {X_test.shape}")
print(f"Testing Labels : {y_test.shape}")

# ------------------------------------
# Predict
# ------------------------------------
print("\nPredicting...\n")

predictions = model.predict(X_test, verbose=0)

y_pred = np.argmax(predictions, axis=1)

# ------------------------------------
# Accuracy
# ------------------------------------
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"Test Accuracy : {accuracy:.4f}")
print("=" * 50)

# ------------------------------------
# Classification Report
# ------------------------------------
print("\nClassification Report\n")

print(classification_report(y_test, y_pred))

# ------------------------------------
# Confusion Matrix
# ------------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("output/confusion_matrix.png")

plt.show()

print("\nConfusion Matrix Saved Successfully!")
print("Location : output/confusion_matrix.png")