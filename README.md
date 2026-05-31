# HealthCure

Flask app for three prediction flows:

- Heart disease: `scaler_heart.pkl` + `heart_model.pkl`
- Diabetes: `scaler_diabetes.pkl` + `diabetes_model.pkl`
- Pneumonia: `pneumonia_model.h5`

## Project Layout

```text
app.py
requirements.txt
models/
  heart_model.pkl
  scaler_heart.pkl
  diabetes_model.pkl
  scaler_diabetes.pkl
  pneumonia_model.h5
templates/
static/
```

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`.

## Test

```bash
pip install pytest
python -m pytest
```

The smoke tests check that all GET routes render without loading the ML models.

## Notes

The app lazy-loads only the model needed for the selected route. This keeps a heart or diabetes request from failing just because the TensorFlow pneumonia model is unavailable.
