# Speech Emotion Recognition from Speech

## Project Overview

This project is developed as part of my **Machine Learning Internship at CodeAlpha**.

The objective of this project is to recognize human emotions from speech audio using **Deep Learning**. The model analyzes speech recordings, extracts audio features (MFCCs), and predicts the emotion expressed by the speaker.

---

# Problem Statement

Human speech contains emotional information in addition to spoken words.

The goal of this project is to classify speech into one of several predefined emotions automatically.

This type of system can be useful in:

* Virtual Assistants
* Customer Service Analytics
* Mental Health Monitoring
* Human-Computer Interaction
* Call Center Analysis

---

# Dataset

Dataset Used:

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

The dataset contains professional emotional speech recordings from multiple actors.

### Dataset Structure

```
audio_speech_actors_01-24/

    Actor_01/
    Actor_02/
    ...
    Actor_24/
```

Each folder contains multiple `.wav` audio files.

Example filename:

```
03-01-05-01-01-01-01.wav
```

The third number represents the emotion label.

---

# Emotion Classes

| Code | Emotion   |
| ---- | --------- |
| 01   | Neutral   |
| 02   | Calm      |
| 03   | Happy     |
| 04   | Sad       |
| 05   | Angry     |
| 06   | Fearful   |
| 07   | Disgust   |
| 08   | Surprised |

After label encoding:

| Encoded Label | Emotion   |
| ------------- | --------- |
| 0             | Angry     |
| 1             | Calm      |
| 2             | Disgust   |
| 3             | Fearful   |
| 4             | Happy     |
| 5             | Neutral   |
| 6             | Sad       |
| 7             | Surprised |

---

# Technologies Used

* Python
* NumPy
* Librosa
* TensorFlow / Keras
* Scikit-learn
* Joblib

---

# Project Structure

```
Speech_Emotion_Recognition/

│
├── audio_speech_actors_01-24/
│
├── feature_extraction.py
├── train.py
├── test.py
│
├── emotion_model.keras
├── label_encoder.pkl
│
└── README.md
```

---

# Workflow

The project follows these steps.

---

## Step 1 — Load Dataset

The RAVDESS dataset is loaded from the local directory.

The program scans every Actor folder and reads all WAV audio files.

---

## Step 2 — Extract Audio Features

For every audio file:

* Load audio using Librosa
* Extract MFCC (Mel Frequency Cepstral Coefficients)
* Generate 40 MFCC features
* Pad or truncate every sample to length 174

Final feature size:

```
(40,174)
```

---

## Step 3 — Create Dataset

All extracted features are stored into:

```
X
```

Emotion labels are stored into:

```
y
```

Dataset size:

```
1440 samples
```

Feature shape:

```
(1440,40,174)
```

---

## Step 4 — Encode Labels

Emotion names are converted into numerical labels using:

```
LabelEncoder
```

The encoder is saved as

```
label_encoder.pkl
```

This encoder is later used during prediction.

---

## Step 5 — Train/Test Split

The dataset is divided into

* Training Set
* Testing Set

using Scikit-learn.

---

## Step 6 — Build Deep Learning Model

A neural network is created using TensorFlow/Keras.

The model learns patterns from MFCC features to classify emotions.

---

## Step 7 — Train the Model

The model is trained for multiple epochs.

Training process displays:

* Training Accuracy
* Validation Accuracy
* Loss
* Validation Loss

---

## Step 8 — Evaluate the Model

After training, the model is evaluated on the testing dataset.

Example result:

```
Test Accuracy

53.47%
```

---

## Step 9 — Save the Model

The trained model is saved as

```
emotion_model.keras
```

This allows the model to be reused without retraining.

---

## Step 10 — Predict Emotion

The `test.py` script loads:

* Saved Model
* Label Encoder

The user provides the path of a WAV audio file.

The system predicts:

* Emotion
* Confidence Score

Example:

```
Prediction Result

Emotion : Angry

Confidence : 99.94%
```

---

# Features

* Automatic feature extraction
* MFCC-based audio processing
* Deep Learning classification
* Label encoding
* Model saving
* Real-time prediction
* Confidence score generation

---

# Results

Dataset Size

```
1440 audio samples
```

Number of Emotion Classes

```
8
```

Model Output

```
Emotion Prediction
Confidence Score
```

Example

```
Emotion : Angry

Confidence : 99.94%
```

---

# Future Improvements

* Improve model accuracy using CNN-LSTM architecture
* Apply audio data augmentation
* Use larger emotional speech datasets
* Build a real-time microphone prediction system
* Develop a Streamlit web application
* Deploy the model on the cloud

---

# Installation

Install the required libraries:

```bash
pip install numpy
pip install librosa
pip install scikit-learn
pip install tensorflow
pip install joblib
```

---

# How to Run

### Feature Extraction

```bash
python feature_extraction.py
```

### Train Model

```bash
python train.py
```

### Test Model

```bash
python test.py
```

---

# Author

**Shakhaoath Pappu**

Machine Learning Intern — CodeAlpha

---

## GitHub Repository Structure

Your repository should look like this:

```
Speech_Emotion_Recognition/
│
├── audio_speech_actors_01-24/
│   ├── Actor_01
│   ├── Actor_02
│   ├── ...
│   └── Actor_24
│
├── feature_extraction.py
├── train.py
├── test.py
├── emotion_model.keras
├── label_encoder.pkl
├── README.md
