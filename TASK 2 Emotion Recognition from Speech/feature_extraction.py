import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import joblib

# Emotion mapping for RAVDESS
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


def extract_features(file_path, max_pad_len=174):

    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        # Pad or truncate
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(
                mfcc,
                pad_width=((0, 0), (0, pad_width)),
                mode='constant'
            )
        else:
            mfcc = mfcc[:, :max_pad_len]

        return mfcc

    except Exception as e:
        print("Error:", file_path)
        print(e)
        return None


def load_dataset(dataset_path):

    X = []
    y = []

    for root, dirs, files in os.walk(dataset_path):

        for file in files:

            if file.endswith(".wav"):

                file_path = os.path.join(root, file)

                # Example filename:
                # 03-01-05-01-01-01-01.wav

                emotion_code = file.split("-")[2]

                emotion = emotion_map.get(emotion_code)

                if emotion is None:
                    continue

                feature = extract_features(file_path)

                if feature is not None:
                    X.append(feature)
                    y.append(emotion)

    X = np.array(X)
    y = np.array(y)

    print("Total Samples:", len(X))
    print("Feature Shape:", X.shape)

    return X, y


def encode_labels(labels):

    encoder = LabelEncoder()

    labels_encoded = encoder.fit_transform(labels)

    joblib.dump(encoder, "label_encoder.pkl")

    return labels_encoded, encoder


if __name__ == "__main__":

    dataset_path = "audio_speech_actors_01-24"

    X, y = load_dataset(dataset_path)

    y, encoder = encode_labels(y)

    print("\nTotal Samples:", len(X))
    print("Feature Shape:", X.shape)

    print("\nEmotion Classes:")
    for i, emotion in enumerate(encoder.classes_):
        print(f"{i} -> {emotion}")