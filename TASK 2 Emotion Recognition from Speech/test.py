import numpy as np
import librosa
import joblib

from tensorflow.keras.models import load_model

# ===============================
# Load Model and Label Encoder
# ===============================

model = load_model("emotion_model.keras")
encoder = joblib.load("label_encoder.pkl")

print("Model Loaded Successfully!")

# ===============================
# Feature Extraction
# ===============================

def extract_features(file_path, max_pad_len=174):

    audio, sample_rate = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    if mfcc.shape[1] < max_pad_len:

        pad_width = max_pad_len - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0,0),(0,pad_width)),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :max_pad_len]

    return mfcc

# ===============================
# Test Audio
# ===============================

audio_file = input("Enter WAV file path: ")

feature = extract_features(audio_file)

feature = feature.reshape(1,40,174,1)

prediction = model.predict(feature)

predicted_class = np.argmax(prediction)

emotion = encoder.inverse_transform([predicted_class])[0]

confidence = np.max(prediction)*100

print("\nPrediction Result")
print("------------------------")
print("Emotion :", emotion)
print("Confidence : {:.2f}%".format(confidence))