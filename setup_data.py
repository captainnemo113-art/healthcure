"""
setup_data.py
-------------
Automated data setup for HealthCure.
1. Heart Disease: Fetched via UCI ML Repository library[cite: 1].
2. Diabetes: Manual download instructions (Kaggle).
3. Pneumonia: Automated download via kagglehub.
"""

import os
import shutil
import pandas as pd
from ucimlrepo import fetch_ucirepo 
import kagglehub

# ── Directory bootstrap ───────────────────────────────────────────────────────
DIRS = ["data", "models", "static/uploads"]
for d in DIRS:
    os.makedirs(d, exist_ok=True)
    print(f"[✓] Directory ready: {d}")


# ── 1. Heart Disease Dataset (UCI Repository) ────────────────────────────────
HEART_PATH = "data/heart.csv"

if not os.path.exists(HEART_PATH):
    print("\n[→] Fetching Heart Disease dataset from UCI...")
    try:
        # Fetching dataset ID 45 (Heart Disease)[cite: 1]
        heart_disease = fetch_ucirepo(id=45) 
        
        # Extract features and targets[cite: 1]
        X = heart_disease.data.features 
        y = heart_disease.data.targets 
        
        # Combine into a single CSV for training
        df = pd.concat([X, y], axis=1)
        df.to_csv(HEART_PATH, index=False)
        print(f"[✓] Saved to {HEART_PATH}")
    except Exception as e:
        print(f"[✗] Heart download failed: {e}")
else:
    print(f"[✓] Heart Disease dataset already exists.")


# ── 2. Diabetes Dataset (Kaggle - Manual) ─────────────────────────────────────
DIABETES_PATH = "data/diabetes.csv"

if not os.path.exists(DIABETES_PATH):
    print("\n[!] Diabetes dataset requires manual download:")
    print("    URL: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database")
    print("    Place 'diabetes.csv' inside the 'data/' folder.")
else:
    print(f"[✓] Diabetes dataset already exists.")


# ── 3. Pneumonia Chest X-ray Dataset (Kagglehub) ──────────────────────────────
CHEST_DIR = "data/chest_xray"

if not os.path.exists(CHEST_DIR):
    print("\n[→] Downloading Pneumonia dataset via kagglehub...")
    try:
        # Download latest version automatically
        download_path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        print(f"[✓] Downloaded to: {download_path}")

        # Locate the actual images inside the downloaded cache
        source_path = os.path.join(download_path, "chest_xray")
        
        # Move it to our project's data folder for easy access
        shutil.move(source_path, CHEST_DIR)
        print(f"[✓] Dataset moved to {CHEST_DIR}")
        
    except Exception as e:
        print(f"[✗] Kagglehub download failed: {e}")
        print("    Ensure KAGGLE_USERNAME and KAGGLE_KEY are set in your environment.")
else:
    print(f"[✓] Chest X-ray dataset already exists.")

print("\n[✓] Setup complete. Run 'python train_models.py' next.")
# Change this at the bottom of train_models.py:
if __name__ == "__main__":
    # train_heart()     <-- Add a # to skip
    # train_diabetes()  <-- Add a # to skip
    train_pneumonia()   # Only run this one
    print("\n[✓] Pneumonia model trained and saved successfully.")