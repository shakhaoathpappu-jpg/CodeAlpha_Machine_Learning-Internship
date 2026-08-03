import numpy as np
from feature_extraction import load_dataset, encode_labels

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

print("\nEmotion Classes:")
for i, emotion in enumerate(encoder.classes_):
    print(f"{i} -> {emotion}")