"""
train_models.py
---------------
Trains and saves three models:
  1. Heart Disease    → RandomForestClassifier  → models/heart_model.pkl
  2. Diabetes         → SVM (SVC)                → models/diabetes_model.pkl
  3. Pneumonia        → MobileNetV2 (TL)        → models/pneumonia_model.h5

Run after setup_data.py:
    python train_models.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress warnings and create models directory
warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)

# Configuration for Image Model
IMG_SIZE = 224   # MobileNetV2 default input size
BATCH_SIZE = 32
EPOCHS = 15

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Heart Disease — RandomForestClassifier
# ═══════════════════════════════════════════════════════════════════════════════

def train_heart():
    """
    UCI Cleveland Heart Disease dataset.
    Cleaning: Converts '?' strings to NaN and binarizes targets.
    """
    print("\n" + "=" * 60)
    print("  [1/3] Training Heart Disease Model (Random Forest)")
    print("=" * 60)

    # 1. Load the data
    if not os.path.exists('data/heart.csv'):
        print("  [✗] data/heart.csv not found! Run setup_data.py first.")
        return
        
    df = pd.read_csv('data/heart.csv')

    # 2. Cleaning for UCI Dataset
    # Convert '?' to NaN and handle numeric conversion[cite: 1]
    df = df.replace('?', np.nan)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with missing values (UCI dataset has 14 clinical variables)[cite: 1]
    df.dropna(inplace=True)

    # Binarize target: 0 = no disease, 1+ = disease present[cite: 1]
    # The original UCI target has values 0-4.[cite: 1]
    df.iloc[:, -1] = (df.iloc[:, -1] > 0).astype(int)

    # Separate features (X) and target (y)[cite: 1]
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Cleaned Data Rows: {len(df)}")
    print(f"  Accuracy : {acc:.4f}")

    # Persist model and scaler
    joblib.dump(model,  "models/heart_model.pkl")
    joblib.dump(scaler, "models/scaler_heart.pkl")
    print("  [✓] Saved: models/heart_model.pkl + models/scaler_heart.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Diabetes — Support Vector Machine (SVM)
# ═══════════════════════════════════════════════════════════════════════════════

def train_diabetes():
    print("\n" + "=" * 60)
    print("  [2/3] Training Diabetes Model (SVM)")
    print("=" * 60)

    if not os.path.exists('data/diabetes.csv'):
        print("  [✗] data/diabetes.csv not found!")
        return

    df = pd.read_csv("data/diabetes.csv")

    # Replace biological 0s with median
    zero_impute_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_impute_cols:
        df[col] = df[col].replace(0, df[col].median())

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # SVM with probability enabled
    model = SVC(kernel="rbf", C=10, probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    print(f"\n  Accuracy : {accuracy_score(y_test, y_pred):.4f}")

    joblib.dump(model,  "models/diabetes_model.pkl")
    joblib.dump(scaler, "models/scaler_diabetes.pkl")
    print("  [✓] Saved: models/diabetes_model.pkl + models/scaler_diabetes.pkl")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Pneumonia — MobileNetV2 Transfer Learning
# ═══════════════════════════════════════════════════════════════════════════════

def train_pneumonia():
    print("\n" + "=" * 60)
    print("  [3/3] Training Pneumonia Model (MobileNetV2)")
    print("=" * 60)

    TRAIN_DIR = "data/chest_xray/train"
    TEST_DIR  = "data/chest_xray/test"

    if not os.path.exists(TRAIN_DIR):
        print("  [✗] Training data not found.")
        return

    # Image Generators
    train_datagen = ImageDataGenerator(
        rescale=1.0/255, rotation_range=15, width_shift_range=0.1,
        height_shift_range=0.1, horizontal_flip=True, validation_split=0.15
    )
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="binary", subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="binary", subset="validation"
    )

    # Transfer Learning Architecture
    base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False 

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

    # Training
    print("\n  Training classification head...")
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, 
              callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

    model.save("models/pneumonia_model.h5")
    print("  [✓] Saved: models/pneumonia_model.h5")

# ═══════════════════════════════════════════════════════════════════════════════
# Execution
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train_heart()
    train_diabetes()
    train_pneumonia()
    print("\n[✓] All models trained and saved successfully.")