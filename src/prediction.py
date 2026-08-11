"""
prediction.py

Saved model, scaler, aur cluster names use karke naye customer ka segment predict karta hai.
"""

import joblib
import numpy as np
import pandas as pd


def load_model_artifacts(model_dir="../models/saved_models/"):
    """
    Saved KMeans model, scaler, feature columns, aur cluster names load karta hai.

    Returns:
        dict: sab artifacts ka mapping
    """
    artifacts = {
        "model": joblib.load(f"{model_dir}customer_segment_predictor.pkl"),
        "scaler": joblib.load(f"{model_dir}scaler.pkl"),
        "feature_columns": joblib.load(f"{model_dir}feature_columns.pkl"),
        "cluster_names": joblib.load(f"{model_dir}cluster_names.pkl"),
    }
    return artifacts


def predict_segment(customer_data, artifacts):
    """
    Ek naye customer ka segment predict karta hai.

    Parameters:
        customer_data (dict): jaise {"recency_days": 50, "frequency": 2, "monetary": 300,
                                       "avg_review_score": 4.5, "avg_delivery_Days": 8}
        artifacts (dict): load_model_artifacts() se aaya dictionary

    Returns:
        str: predicted segment name
    """
    feature_columns = artifacts["feature_columns"]
    scaler = artifacts["scaler"]
    model = artifacts["model"]
    cluster_names = artifacts["cluster_names"]

    # Input ko sahi order mein DataFrame banate hain
    input_df = pd.DataFrame([customer_data])[feature_columns]

    # IMPORTANT: sirf transform, fit_transform NAHI (scaler already trained hai)
    scaled_input = scaler.transform(input_df)

    cluster_number = model.predict(scaled_input)[0]
    segment_name = cluster_names[cluster_number]

    return segment_name


if __name__ == "__main__":
    # Example usage / quick test
    artifacts = load_model_artifacts()

    sample_customer = {
        "recency_days": 50,
        "frequency": 2,
        "monetary": 300,
        "avg_review_score": 4.5,
        "avg_delivery_Days": 8,
    }

    predicted_segment = predict_segment(sample_customer, artifacts)
    print(f"Predicted segment: {predicted_segment}")