import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# -----------------------------
# Load Trained Model
# -----------------------------
print("=" * 50)
print("Loading Trained Model...")
print("=" * 50)

model = load_model("saved_model/cnn_model.keras")

print("Model Loaded Successfully!")

# -----------------------------
# Image Path
# -----------------------------
image_path = "samples/digit.png"

# -----------------------------
# Read Image
# -----------------------------
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print(f"Error: Cannot find image -> {image_path}")
    exit()

# -----------------------------
# Resize Image
# -----------------------------
image = cv2.resize(image, (28, 28))

# -----------------------------
# Invert Colors
# -----------------------------
image = 255 - image

# -----------------------------
# Normalize
# -----------------------------
image = image.astype("float32") / 255.0

# -----------------------------
# Reshape
# -----------------------------
input_image = image.reshape(1, 28, 28, 1)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(input_image, verbose=0)

predicted_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100

print("\nPrediction Completed!")
print(f"Predicted Digit : {predicted_digit}")
print(f"Confidence       : {confidence:.2f}%")

# -----------------------------
# Display Image
# -----------------------------
plt.imshow(image, cmap="gray")
plt.title(f"Prediction: {predicted_digit}")
plt.axis("off")
plt.show()