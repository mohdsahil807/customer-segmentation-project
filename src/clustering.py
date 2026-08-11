"""
clustering.py

Customer features ko scale karke KMeans clustering se segments banata hai.
Isko 05_Model_Training.ipynb notebook se convert kiya gaya hai.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans


FEATURE_COLUMNS = ["recency_days", "frequency", "monetary", "avg_review_score", "avg_delivery_Days"]

CLUSTER_NAMES = {
    0: "Lost / Inactive",
    1: "At Risk / Dissatisfied",
    2: "High Value",
    3: "Engaged / Recent",
}


def load_features(path="../data/processed/customer_features.csv"):
    """
    Customer feature table load karta hai.
    """
    return pd.read_csv(path)


def cap_outliers(series, factor=1.5):
    """
    IQR method se ek column ke extreme outliers ko cap karta hai.
    Extreme values ko delete nahi karta, sirf upper/lower limit tak seemit karta hai.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.clip(lower=lower, upper=upper)


def apply_outlier_capping(df, columns=("recency_days", "frequency", "monetary", "avg_delivery_Days")):
    """
    Di gayi columns pe outlier capping apply karta hai.
    """
    df_capped = df.copy()
    for col in columns:
        df_capped[col] = cap_outliers(df_capped[col])
    return df_capped


def scale_features(df_capped, feature_columns=FEATURE_COLUMNS):
    """
    RobustScaler se features ko scale karta hai (outliers-resistant scaling).

    Returns:
        X_scaled (np.array), fitted scaler object
    """
    X = df_capped[feature_columns]
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def train_kmeans(X_scaled, n_clusters=4, random_state=42):
    """
    KMeans model train karta hai aur cluster labels return karta hai.

    Returns:
        trained KMeans model, cluster labels array
    """
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    return kmeans, cluster_labels


def assign_segment_names(df_capped, cluster_labels, cluster_names=CLUSTER_NAMES):
    """
    Cluster numbers ko business-meaningful naam se map karta hai.
    """
    df_capped = df_capped.copy()
    df_capped["cluster"] = cluster_labels
    df_capped["segment_name"] = df_capped["cluster"].map(cluster_names)
    return df_capped


def save_model_artifacts(
    kmeans_model,
    scaler,
    feature_columns=FEATURE_COLUMNS,
    cluster_names=CLUSTER_NAMES,
    output_dir="../models/saved_models/",
):
    """
    Trained model, scaler, feature columns, aur cluster names ko save karta hai
    taaki future predictions ke liye dobara training na karni pade.
    """
    joblib.dump(kmeans_model, f"{output_dir}customer_segment_predictor.pkl")
    joblib.dump(scaler, f"{output_dir}scaler.pkl")
    joblib.dump(feature_columns, f"{output_dir}feature_columns.pkl")
    joblib.dump(cluster_names, f"{output_dir}cluster_names.pkl")
    print("Model, scaler, aur metadata saved successfully!")


def save_segmented_data(df_capped, output_path="../data/processed/customer_segments.csv"):
    """
    Final segmented customer data ko CSV mein save karta hai.
    """
    df_capped.to_csv(output_path, index=False)
    print(f"Segmented data saved to {output_path}")
    print(f"Final shape: {df_capped.shape}")


def run_clustering_pipeline(
    input_path="../data/processed/customer_features.csv",
    output_path="../data/processed/customer_segments.csv",
    model_dir="../models/saved_models/",
):
    """
    Poora clustering pipeline:
    load -> outlier capping -> scale -> train KMeans -> assign names -> save
    """
    df = load_features(input_path)
    df_capped = apply_outlier_capping(df)

    X_scaled, scaler = scale_features(df_capped)
    kmeans_model, cluster_labels = train_kmeans(X_scaled)

    df_capped = assign_segment_names(df_capped, cluster_labels)

    save_model_artifacts(kmeans_model, scaler)
    save_segmented_data(df_capped, output_path)

    return df_capped, kmeans_model, scaler


if __name__ == "__main__":
    run_clustering_pipeline()