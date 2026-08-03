import numpy as np
from feature_extraction import load_dataset, encode_labels

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical

print("=" * 50)
print("Speech Emotion Recognition Training")
print("=" * 50)

dataset_path = "audio_speech_actors_01-24"

print("\nLoading Dataset...")

X, y = load_dataset(dataset_path)

y, encoder = encode_labels(y)

print("\nDataset Loaded Successfully!")

print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# CNN input shape
X = X.reshape(X.shape[0], 40, 174, 1)

# One-hot encoding
y = to_categorical(y)

print("\nTrain/Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

print("\nBuilding CNN Model...")

model = Sequential()

model.add(
    Conv2D(
        32,
        (3,3),
        activation="relu",
        input_shape=(40,174,1)
    )
)

model.add(MaxPooling2D((2,2)))

model.add(
    Conv2D(
        64,
        (3,3),
        activation="relu"
    )
)

model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(8, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nTraining Started...\n")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=20,
    batch_size=32
)

print("\nEvaluating...")

loss, accuracy = model.evaluate(X_test,y_test)

print("\nTest Accuracy:", accuracy)

model.save("emotion_model.keras")

print("\nModel Saved Successfully!")