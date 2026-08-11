"""
feature_engineering.py

Order-level data ko customer-level RFM + behavioral features mein convert karta hai.
Isko 04_Feature_Engineering.ipynb notebook se convert kiya gaya hai.
"""

import pandas as pd


def load_cleaned_data(path="../data/intermediate/cleaned_orders.csv"):
    """
    Cleaned CSV load karta hai aur date columns wapas datetime mein convert karta hai
    (CSV save/load se datetime text ban jaata hai, isliye dobara convert zaroori hai).
    """
    df = pd.read_csv(path)

    date_columns = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "shipping_limit_date",
        "review_creation_date",
        "review_answer_timestamp",
    ]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


def filter_delivered_orders(df):
    """
    Sirf 'delivered' status wale orders rakhta hai,
    kyunki cancelled/unavailable orders customer behavior ko sahi represent nahi karte.
    """
    return df[df["order_status"] == "delivered"].copy()


def calculate_recency(df):
    """
    Har customer ka Recency (last order se din) calculate karta hai.
    Reference date = dataset ki max order date (historical data hai, "aaj" nahi).
    """
    reference_date = df["order_purchase_timestamp"].max()

    recency_df = df.groupby("customer_unique_id")["order_purchase_timestamp"].max().reset_index()
    recency_df.columns = ["customer_unique_id", "last_order_date"]
    recency_df["recency_days"] = (reference_date - recency_df["last_order_date"]).dt.days

    return recency_df[["customer_unique_id", "recency_days"]]


def calculate_frequency(df):
    """
    Har customer ne kitne unique orders kiye, calculate karta hai.
    """
    frequency_df = df.groupby("customer_unique_id")["order_id"].nunique().reset_index()
    frequency_df.columns = ["customer_unique_id", "frequency"]
    return frequency_df


def calculate_monetary(df):
    """
    Har customer ka total lifetime spend (payment_value ka sum) calculate karta hai.
    """
    monetary_df = df.groupby("customer_unique_id")["payment_value"].sum().reset_index()
    monetary_df.columns = ["customer_unique_id", "monetary"]
    return monetary_df


def calculate_review_and_delivery(df):
    """
    Bonus features: avg review score aur avg delivery time per customer.
    """
    review_df = df.groupby("customer_unique_id")["review_score"].mean().reset_index()
    review_df.columns = ["customer_unique_id", "avg_review_score"]

    df = df.copy()
    df["delivery_time_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    delivery_df = df.groupby("customer_unique_id")["delivery_time_days"].mean().reset_index()
    delivery_df.columns = ["customer_unique_id", "avg_delivery_Days"]

    return review_df, delivery_df


def merge_features(recency_df, frequency_df, monetary_df, review_df, delivery_df):
    """
    Saare feature dataframes ko customer_unique_id pe merge karke
    final customer-level feature table banata hai.
    """
    customer_features = recency_df.merge(frequency_df, on="customer_unique_id", how="left")
    customer_features = customer_features.merge(monetary_df, on="customer_unique_id", how="left")
    customer_features = customer_features.merge(review_df, on="customer_unique_id", how="left")
    customer_features = customer_features.merge(delivery_df, on="customer_unique_id", how="left")

    return customer_features


def handle_missing_features(customer_features):
    """
    Missing avg_review_score aur avg_delivery_Days ko column ke mean se fill karta hai
    (bahut kam % missing tha, isliye fill karna safe hai).
    """
    customer_features = customer_features.copy()
    customer_features["avg_review_score"] = customer_features["avg_review_score"].fillna(
        customer_features["avg_review_score"].mean()
    )
    customer_features["avg_delivery_Days"] = customer_features["avg_delivery_Days"].fillna(
        customer_features["avg_delivery_Days"].mean()
    )
    return customer_features


def save_features(customer_features, output_path="../data/processed/customer_features.csv"):
    """
    Final feature table ko CSV mein save karta hai.
    """
    customer_features.to_csv(output_path, index=False)
    print(f"Customer features saved to {output_path}")
    print(f"Final shape: {customer_features.shape}")


def run_feature_engineering_pipeline(
    input_path="../data/intermediate/cleaned_orders.csv",
    output_path="../data/processed/customer_features.csv",
):
    """
    Poora feature engineering pipeline:
    load -> filter delivered -> RFM + bonus features -> merge -> clean -> save
    """
    df = load_cleaned_data(input_path)
    df = filter_delivered_orders(df)

    recency_df = calculate_recency(df)
    frequency_df = calculate_frequency(df)
    monetary_df = calculate_monetary(df)
    review_df, delivery_df = calculate_review_and_delivery(df)

    customer_features = merge_features(recency_df, frequency_df, monetary_df, review_df, delivery_df)
    customer_features = handle_missing_features(customer_features)

    save_features(customer_features, output_path)
    return customer_features


if __name__ == "__main__":
    run_feature_engineering_pipeline()