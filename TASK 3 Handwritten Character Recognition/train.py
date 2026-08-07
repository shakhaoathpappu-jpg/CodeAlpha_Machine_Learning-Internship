import os
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

from model import create_model


# -----------------------------
# Create folders if not exists
# -----------------------------
os.makedirs("saved_model", exist_ok=True)
os.makedirs("output", exist_ok=True)


# -----------------------------
# Load MNIST Dataset
# -----------------------------
print("=" * 50)
print("Loading MNIST Dataset...")
print("=" * 50)

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(f"Training Images : {X_train.shape}")
print(f"Training Labels : {y_train.shape}")
print(f"Testing Images  : {X_test.shape}")
print(f"Testing Labels  : {y_test.shape}")


# -----------------------------
# Normalize Images
# -----------------------------
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Add Channel Dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)


print("\nDataset Preprocessing Completed.")
print(f"Training Shape : {X_train.shape}")
print(f"Testing Shape  : {X_test.shape}")


# -----------------------------
# Build CNN Model
# -----------------------------
print("\nBuilding CNN Model...")

model = create_model()

model.summary()


# -----------------------------
# Train Model
# -----------------------------
print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)


# -----------------------------
# Evaluate Model
# -----------------------------
print("\nEvaluating Model...\n")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("=" * 50)
print(f"Test Accuracy : {test_accuracy:.4f}")
print(f"Test Loss     : {test_loss:.4f}")
print("=" * 50)


# -----------------------------
# Save Model
# -----------------------------
model_path = "saved_model/cnn_model.keras"

model.save(model_path)

print(f"\nModel Saved Successfully!")
print(f"Location : {model_path}")


# -----------------------------
# Accuracy Graph
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("output/accuracy.png")

plt.show()


# -----------------------------
# Loss Graph
# -----------------------------
plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("output/loss.png")

plt.show()

print("\nTraining Completed Successfully!")