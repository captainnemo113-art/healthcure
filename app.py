"""
app.py
------
Flask backend for HealthCure.
Routes:
  /                  → Home page
  /heart             → Heart Disease prediction (GET form / POST result)
  /diabetes          → Diabetes prediction     (GET form / POST result)
  /pneumonia         → Pneumonia prediction    (GET form / POST result)
"""

import os
import uuid
import numpy as np
import joblib

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# ── App Configuration ─────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "healthcure-secret-2024"   # change in production

UPLOAD_FOLDER  = os.path.join("static", "uploads")
ALLOWED_EXTS   = {"png", "jpg", "jpeg"}
IMG_SIZE       = 224   # must match training

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB limit


# ── Model Loading (lazy, cached at first request) ─────────────────────────────
_heart_model      = None
_heart_scaler     = None
_diabetes_model   = None
_diabetes_scaler  = None
_pneumonia_model  = None


def load_models():
    """Load all models once and cache them in module-level variables."""
    global _heart_model, _heart_scaler
    global _diabetes_model, _diabetes_scaler
    global _pneumonia_model

    if _heart_model is None:
        _heart_model     = joblib.load("models/heart_model.pkl")
        _heart_scaler    = joblib.load("models/scaler_heart.pkl")

    if _diabetes_model is None:
        _diabetes_model  = joblib.load("models/diabetes_model.pkl")
        _diabetes_scaler = joblib.load("models/scaler_diabetes.pkl")

    if _pneumonia_model is None:
        _pneumonia_model = tf.keras.models.load_model("models/pneumonia_model.h5")


# ── Utility Helpers ───────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def preprocess_xray(image_path: str) -> np.ndarray:
    """
    Load an X-ray image, resize to 224×224, normalise to [0, 1],
    and expand dims to (1, 224, 224, 3) for model input.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Landing / home page."""
    return render_template("index.html")


# ─── Heart Disease ────────────────────────────────────────────────────────────

@app.route("/heart", methods=["GET", "POST"])
def heart():
    result = None
    confidence = None

    if request.method == "POST":
        load_models()
        try:
            # Collect the 13 UCI Cleveland features from the form
            features = [
                float(request.form["age"]),
                float(request.form["sex"]),
                float(request.form["cp"]),
                float(request.form["trestbps"]),
                float(request.form["chol"]),
                float(request.form["fbs"]),
                float(request.form["restecg"]),
                float(request.form["thalach"]),
                float(request.form["exang"]),
                float(request.form["oldpeak"]),
                float(request.form["slope"]),
                float(request.form["ca"]),
                float(request.form["thal"]),
            ]

            X = np.array(features).reshape(1, -1)
            X_scaled = _heart_scaler.transform(X)

            prediction  = _heart_model.predict(X_scaled)[0]
            prob        = _heart_model.predict_proba(X_scaled)[0]
            confidence  = round(float(max(prob)) * 100, 2)
            result      = "Heart Disease Detected" if prediction == 1 else "No Heart Disease Detected"

        except (ValueError, KeyError) as e:
            flash(f"Input error: {e}. Please fill in all fields correctly.", "danger")

    return render_template("heart.html", result=result, confidence=confidence)


# ─── Diabetes ─────────────────────────────────────────────────────────────────

@app.route("/diabetes", methods=["GET", "POST"])
def diabetes():
    result = None
    confidence = None

    if request.method == "POST":
        load_models()
        try:
            features = [
                float(request.form["pregnancies"]),
                float(request.form["glucose"]),
                float(request.form["blood_pressure"]),
                float(request.form["skin_thickness"]),
                float(request.form["insulin"]),
                float(request.form["bmi"]),
                float(request.form["dpf"]),        # Diabetes Pedigree Function
                float(request.form["age"]),
            ]

            X = np.array(features).reshape(1, -1)
            X_scaled = _diabetes_scaler.transform(X)

            prediction  = _diabetes_model.predict(X_scaled)[0]
            prob        = _diabetes_model.predict_proba(X_scaled)[0]
            confidence  = round(float(max(prob)) * 100, 2)
            result      = "Diabetic" if prediction == 1 else "Non-Diabetic"

        except (ValueError, KeyError) as e:
            flash(f"Input error: {e}. Please fill in all fields correctly.", "danger")

    return render_template("diabetes.html", result=result, confidence=confidence)


# ─── Pneumonia ────────────────────────────────────────────────────────────────

@app.route("/pneumonia", methods=["GET", "POST"])
def pneumonia():
    result     = None
    confidence = None
    img_path   = None

    if request.method == "POST":
        load_models()

        if "xray" not in request.files:
            flash("No file part in request.", "danger")
            return redirect(request.url)

        file = request.files["xray"]

        if file.filename == "":
            flash("No file selected.", "warning")
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash("Only PNG/JPG images are accepted.", "warning")
            return redirect(request.url)

        # Save with a unique name to avoid collisions
        ext      = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # Preprocess and predict
        img_array   = preprocess_xray(save_path)
        raw_score   = float(_pneumonia_model.predict(img_array)[0][0])

        # MobileNetV2 with sigmoid: score > 0.5 → PNEUMONIA (class 1 in generator)
        if raw_score > 0.5:
            result     = "Pneumonia Detected"
            confidence = round(raw_score * 100, 2)
        else:
            result     = "Normal (No Pneumonia)"
            confidence = round((1 - raw_score) * 100, 2)

        # Pass relative path so the template can render the uploaded image
        img_path = f"uploads/{filename}"

    return render_template("pneumonia.html", result=result,
                           confidence=confidence, img_path=img_path)


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

