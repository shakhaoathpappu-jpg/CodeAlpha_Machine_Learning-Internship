import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

# ====================================================
# Load Model
# ====================================================

print("=" * 50)
print("Loading Trained Model...")
print("=" * 50)

model = load_model("saved_model/cnn_model.keras")

print("Model Loaded Successfully!")

# ====================================================
# Image Path
# ====================================================

image_path = "samples/digit.png"

# ====================================================
# Read Image
# ====================================================

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Image Not Found!")
    exit()

# ====================================================
# Threshold
# ====================================================

_, thresh = cv2.threshold(
    image,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# ====================================================
# Find Largest Contour
# ====================================================

contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if len(contours) == 0:
    print("No Digit Found!")
    exit()

largest = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(largest)

digit = thresh[y:y+h, x:x+w]

# ====================================================
# Resize keeping Aspect Ratio
# ====================================================

size = 20

h, w = digit.shape

if h > w:

    new_h = size
    new_w = int(w * size / h)

else:

    new_w = size
    new_h = int(h * size / w)

digit = cv2.resize(digit, (new_w, new_h))

# ====================================================
# Put Digit at Center of 28x28 Canvas
# ====================================================

canvas = np.zeros((28, 28), dtype=np.uint8)

x_offset = (28 - new_w) // 2
y_offset = (28 - new_h) // 2

canvas[
    y_offset:y_offset + new_h,
    x_offset:x_offset + new_w
] = digit

# ====================================================
# Normalize
# ====================================================

input_image = canvas.astype("float32") / 255.0
input_image = input_image.reshape(1, 28, 28, 1)

# ====================================================
# Prediction
# ====================================================

prediction = model.predict(input_image, verbose=0)

predicted_digit = np.argmax(prediction)

confidence = float(np.max(prediction)) * 100

print("\nPrediction Completed!")
print(f"Predicted Digit : {predicted_digit}")
print(f"Confidence       : {confidence:.2f}%")

# ====================================================
# Show Image
# ====================================================

plt.figure(figsize=(4,4))
plt.imshow(canvas, cmap="gray")
plt.title(f"Prediction : {predicted_digit}\nConfidence : {confidence:.2f}%")
plt.axis("off")
plt.show()