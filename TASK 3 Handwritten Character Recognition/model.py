from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout


def create_model():
    model = Sequential()

    # First Convolution Layer
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            input_shape=(28, 28, 1),
        )
    )

    # First Pooling Layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolution Layer
    model.add(
        Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
        )
    )

    # Second Pooling Layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten Layer
    model.add(Flatten())

    # Fully Connected Layer
    model.add(Dense(128, activation="relu"))

    # Dropout
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(10, activation="softmax"))

    # Compile Model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = create_model()
    model.summary()