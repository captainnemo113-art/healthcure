"""
app.py
------
Flask backend for HealthCure.

Routes:
  /                  -> Home page
  /register          -> User registration
  /login             -> User login
  /logout            -> User logout
  /heart             -> Heart Disease prediction (GET form / POST result)
  /diabetes          -> Diabetes prediction     (GET form / POST result)
  /pneumonia         -> Pneumonia prediction    (GET form / POST result)
  /healthz           -> Simple health check
  /history           -> View past prediction records
  /api/history       -> API to fetch past prediction records
  /api/predict/heart -> API endpoint for heart prediction
  /api/predict/diabetes -> API endpoint for diabetes prediction
"""

import os
import uuid

import joblib
import numpy as np
import tensorflow as tf

from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user, login_required, login_user, logout_user

from PIL import Image
from werkzeug.utils import secure_filename
from models_db import User, PredictionHistory, db

# ── App Configuration ─────────────────────────────────────────────────────────
load_dotenv()

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "healthcure-dev-secret")
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "sqlite:///healthcure.db",
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message_category = "warning"

UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTS = {"png", "jpg", "jpeg"}
IMG_SIZE = 224  # must match training

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit


# ── Model Loading (lazy, cached at first request) ─────────────────────────────
_heart_model = None
_heart_scaler = None
_diabetes_model = None
_diabetes_scaler = None
_pneumonia_model = None


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


@app.cli.command("init-db")
def init_db():
    db.create_all()
    print("Database tables created.")


def load_heart_model():
    """Load the heart model only when a heart prediction is requested."""
    global _heart_model, _heart_scaler

    if _heart_model is None:
        _heart_model = joblib.load("models/heart_model.pkl")
        _heart_scaler = joblib.load("models/scaler_heart.pkl")

    return _heart_model, _heart_scaler


def load_diabetes_model():
    """Load the diabetes model only when a diabetes prediction is requested."""
    global _diabetes_model, _diabetes_scaler

    if _diabetes_model is None:
        _diabetes_model = joblib.load("models/diabetes_model.pkl")
        _diabetes_scaler = joblib.load("models/scaler_diabetes.pkl")

    return _diabetes_model, _diabetes_scaler


def load_pneumonia_model():
    """Load the image model only when an X-ray prediction is requested."""
    global _pneumonia_model

    if _pneumonia_model is None:
        _pneumonia_model = tf.keras.models.load_model("models/pneumonia_model.h5")

    return _pneumonia_model


# ── Utility Helpers ───────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS


def preprocess_xray(image_path: str) -> np.ndarray:
    """Load an X-ray image, resize to 224×224, normalise to [0, 1],
    and expand dims to (1, 224, 224, 3) for model input.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def save_prediction(prediction_type, input_summary, result, confidence):
    """Helper method to explicitly store data to the DB context."""
    history = PredictionHistory(
        user_id=current_user.id,
        prediction_type=prediction_type,
        input_summary=input_summary,
        result=result,
        confidence=confidence,
    )
    db.session.add(history)
    db.session.commit()


def prediction_to_dict(prediction):
    return {
        "id": prediction.id,
        "prediction_type": prediction.prediction_type,
        "input_summary": prediction.input_summary,
        "result": prediction.result,
        "confidence": prediction.confidence,
        "created_at": prediction.created_at.isoformat(),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/healthz")
def healthz():
    """Cheap route for deployment checks; does not load ML models."""
    return {"status": "ok"}


@app.route("/")
def index():
    """Landing / home page."""
    return render_template("index.html")


# ─── Auth Routes ──────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("Please fill in all fields.", "danger")
            return redirect(url_for("register"))

        existing_user = User.query.filter_by(email=email).first()
        if existing_user:
            flash("An account with this email already exists.", "warning")
            return redirect(url_for("login"))

        user = User(name=name, email=email)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        login_user(user)
        flash("Registration successful.", "success")
        return redirect(url_for("index"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user is None or not user.check_password(password):
            flash("Invalid email or password.", "danger")
            return redirect(url_for("login"))

        login_user(user)
        flash("Logged in successfully.", "success")
        return redirect(url_for("index"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))


# ─── Heart Disease ────────────────────────────────────────────────────────────

@app.route("/heart", methods=["GET", "POST"])
@login_required
def heart():
    result = None
    confidence = None

    if request.method == "POST":
        heart_model, heart_scaler = load_heart_model()
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
            X_scaled = heart_scaler.transform(X)

            prediction = heart_model.predict(X_scaled)[0]
            prob = heart_model.predict_proba(X_scaled)[0]
            confidence = round(float(max(prob)) * 100, 2)
            result = (
                "Heart Disease Detected"
                if prediction == 1
                else "No Heart Disease Detected"
            )

            input_summary = (
                f"age={features[0]}, sex={features[1]}, cp={features[2]}, "
                f"trestbps={features[3]}, chol={features[4]}, fbs={features[5]}, "
                f"restecg={features[6]}, thalach={features[7]}, exang={features[8]}, "
                f"oldpeak={features[9]}, slope={features[10]}, ca={features[11]}, thal={features[12]}"
            )

            save_prediction("Heart Disease", input_summary, result, confidence)

        except (ValueError, KeyError) as e:
            flash(f"Input error: {e}. Please fill in all fields correctly.", "danger")

    return render_template("heart.html", result=result, confidence=confidence)


# ─── Diabetes ─────────────────────────────────────────────────────────────────

@app.route("/diabetes", methods=["GET", "POST"])
@login_required
def diabetes():
    result = None
    confidence = None

    if request.method == "POST":
        diabetes_model, diabetes_scaler = load_diabetes_model()
        try:
            features = [
                float(request.form["pregnancies"]),
                float(request.form["glucose"]),
                float(request.form["blood_pressure"]),
                float(request.form["skin_thickness"]),
                float(request.form["insulin"]),
                float(request.form["bmi"]),
                float(request.form["dpf"]),  # Diabetes Pedigree Function
                float(request.form["age"]),
            ]

            X = np.array(features).reshape(1, -1)
            X_scaled = diabetes_scaler.transform(X)

            prediction = diabetes_model.predict(X_scaled)[0]
            prob = diabetes_model.predict_proba(X_scaled)[0]
            confidence = round(float(max(prob)) * 100, 2)
            result = "Diabetic" if prediction == 1 else "Non-Diabetic"

            input_summary = (
                f"pregnancies={features[0]}, glucose={features[1]}, "
                f"blood_pressure={features[2]}, skin_thickness={features[3]}, "
                f"insulin={features[4]}, bmi={features[5]}, dpf={features[6]}, age={features[7]}"
            )

            save_prediction("Diabetes", input_summary, result, confidence)

        except (ValueError, KeyError) as e:
            flash(f"Input error: {e}. Please fill in all fields correctly.", "danger")

    return render_template("diabetes.html", result=result, confidence=confidence)


# ─── Pneumonia ────────────────────────────────────────────────────────────────

@app.route("/pneumonia", methods=["GET", "POST"])
@login_required
def pneumonia():
    result = None
    confidence = None
    img_path = None

    if request.method == "POST":
        pneumonia_model = load_pneumonia_model()

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
        ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # Preprocess and predict
        img_array = preprocess_xray(save_path)
        raw_score = float(pneumonia_model.predict(img_array)[0][0])

        # MobileNetV2 with sigmoid: score > 0.5 → PNEUMONIA (class 1 in generator)
        if raw_score > 0.5:
            result = "Pneumonia Detected"
            confidence = round(raw_score * 100, 2)
        else:
            result = "Normal (No Pneumonia)"
            confidence = round((1 - raw_score) * 100, 2)

        # Pass relative path so the template can render the uploaded image
        img_path = f"uploads/{filename}"

        input_summary = f"uploaded_file={filename}"
        save_prediction("Pneumonia", input_summary, result, confidence)

    return render_template(
        "pneumonia.html", result=result, confidence=confidence, img_path=img_path
    )


# ─── History View Routes ──────────────────────────────────────────────────────

@app.route("/history")
@login_required
def history():
    predictions = (
        PredictionHistory.query
        .filter_by(user_id=current_user.id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )
    return render_template("history.html", predictions=predictions)


@app.route("/api/history")
@login_required
def api_history():
    predictions = (
        PredictionHistory.query
        .filter_by(user_id=current_user.id)
        .order_by(PredictionHistory.created_at.desc())
        .all()
    )
    return jsonify([prediction_to_dict(item) for item in predictions])


@app.route("/api/me")
@login_required
def api_me():
    return jsonify({
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
    })


# ─── API endpoints ────────────────────────────────────────────────────────────

@app.route("/api/predict/heart", methods=["POST"])
@login_required
def api_predict_heart():
    data = request.get_json() or {}

    required_fields = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        features = [float(data[field]) for field in required_fields]

        heart_model, heart_scaler = load_heart_model()
        x_value = np.array(features).reshape(1, -1)
        x_scaled = heart_scaler.transform(x_value)

        prediction = heart_model.predict(x_scaled)[0]
        probability = heart_model.predict_proba(x_scaled)[0]
        confidence = round(float(max(probability)) * 100, 2)
        result = (
            "Heart Disease Detected"
            if prediction == 1
            else "No Heart Disease Detected"
        )

        input_summary = ", ".join(
            f"{field}={data[field]}" for field in required_fields
        )
        save_prediction("Heart Disease", input_summary, result, confidence)

        return jsonify({
            "prediction_type": "Heart Disease",
            "result": result,
            "confidence": confidence,
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400


@app.route("/api/predict/diabetes", methods=["POST"])
@login_required
def api_predict_diabetes():
    data = request.get_json() or {}

    required_fields = [
        "pregnancies", "glucose", "blood_pressure", "skin_thickness",
        "insulin", "bmi", "dpf", "age"
    ]

    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
        }), 400

    try:
        features = [float(data[field]) for field in required_fields]

        diabetes_model, diabetes_scaler = load_diabetes_model()
        x_value = np.array(features).reshape(1, -1)
        x_scaled = diabetes_scaler.transform(x_value)

        prediction = diabetes_model.predict(x_scaled)[0]
        probability = diabetes_model.predict_proba(x_scaled)[0]
        confidence = round(float(max(probability)) * 100, 2)
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"

        input_summary = ", ".join(
            f"{field}={data[field]}" for field in required_fields
        )
        save_prediction("Diabetes", input_summary, result, confidence)

        return jsonify({
            "prediction_type": "Diabetes",
            "result": result,
            "confidence": confidence,
        })

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

@app.before_request
def create_tables_once():
    if not getattr(app, "_tables_created", False):
        db.create_all()
        app._tables_created = True
# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)