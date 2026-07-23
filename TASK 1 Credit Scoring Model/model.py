# ============================================
# Credit Scoring Model
# Dataset: Give Me Some Credit (Kaggle)
# ============================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================
# Load Dataset
# ============================================

print("=" * 60)
print("Credit Scoring Model")
print("=" * 60)

# Current folder where model.py exists
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("\nCurrent Folder:")
print(BASE_DIR)

print("\nFiles Found:")
print(os.listdir(BASE_DIR))

# Dataset Path
csv_file = os.path.join(BASE_DIR, "cs-training.csv")

print("\nDataset Path:")
print(csv_file)

# Check file exists
if not os.path.exists(csv_file):
    print("\nERROR!")
    print("cs-training.csv was NOT found.")
    print("\nMake sure the dataset is inside the same folder as model.py")
    exit()

print("\nLoading Dataset...")

df = pd.read_csv(csv_file)

print("\nDataset Loaded Successfully!")

# ============================================
# Basic Information
# ============================================

print("\nFirst 5 Rows\n")
print(df.head())

print("\nShape")
print(df.shape)

print("\nColumns")
print(df.columns.tolist())

print("\nMissing Values")
print(df.isnull().sum())

# ============================================
# Remove Unnecessary Column
# ============================================

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# ============================================
# Fill Missing Values
# ============================================

print("\nHandling Missing Values...")

imputer = SimpleImputer(strategy="median")

df = pd.DataFrame(
    imputer.fit_transform(df),
    columns=df.columns
)

# ============================================
# Feature Engineering
# ============================================

print("Creating Features...")

df["TotalLatePayments"] = (
    df["NumberOfTime30-59DaysPastDueNotWorse"]
    + df["NumberOfTime60-89DaysPastDueNotWorse"]
    + df["NumberOfTimes90DaysLate"]
)

df["IncomePerDependent"] = (
    df["MonthlyIncome"] /
    (df["NumberOfDependents"] + 1)
)

# ============================================
# Features & Target
# ============================================

TARGET = "SeriousDlqin2yrs"

X = df.drop(TARGET, axis=1)
y = df[TARGET]

# ============================================
# Train Test Split
# ============================================

print("\nSplitting Dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ============================================
# Train Model
# ============================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

print("Training Complete!")

# ============================================
# Prediction
# ============================================

print("\nMaking Predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ============================================
# Evaluation
# ============================================

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print("Accuracy :", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred), 4))
print("Recall   :", round(recall_score(y_test, y_pred), 4))
print("F1 Score :", round(f1_score(y_test, y_pred), 4))
print("ROC AUC  :", round(roc_auc_score(y_test, y_prob), 4))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ============================================
# Feature Importance
# ============================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("TOP IMPORTANT FEATURES")
print("=" * 60)

print(importance)

# ============================================
# Sample Prediction
# ============================================

print("\n" + "=" * 60)
print("SAMPLE PREDICTION")
print("=" * 60)

sample = X.iloc[[0]]

prediction = model.predict(sample)

if prediction[0] == 0:
    print("Customer Prediction : GOOD CREDIT")
else:
    print("Customer Prediction : BAD CREDIT")

print("\nProject Completed Successfully!")


# ============================================
# Save Predictions to CSV
# ============================================

prediction_df = pd.DataFrame({
    "Actual": y_test.map({
        0: "Good Credit",
        1: "Bad Credit"
    }).values,

    "Predicted": pd.Series(y_pred).map({
        0: "Good Credit",
        1: "Bad Credit"
    })
})

prediction_df.to_csv("prediction_results.csv", index=False)

print("Prediction CSV Saved Successfully!")