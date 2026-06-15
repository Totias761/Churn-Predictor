import joblib
import numpy as np
import pandas as pd

# Load model artifacts once at startup
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("features_list.pkl")

def predict(input_data: dict) -> dict:
    """
    Takes a dictionary of raw feature values,
    scales them and returns churn prediction.
    """
    # Build dataframe in correct feature order
    df = pd.DataFrame([input_data])[feature_names]

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return {
        "churned": bool(prediction),
        "churn_probability": round(float(probability), 3),
        "risk_level": (
            "High" if probability > 0.7
            else "Medium" if probability > 0.4
            else "Low"
        )
    }