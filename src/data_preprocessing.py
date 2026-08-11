"""
data_preprocessing.py

Merged data ko clean karta hai — missing values, date conversion, duplicates.
Isko 02_Data_Preprocessing.ipynb notebook se convert kiya gaya hai.
"""

import pandas as pd


DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
    "review_creation_date",
    "review_answer_timestamp",
]


def load_merged_data(path="../data/intermediate/merged_orders.csv"):
    """
    Merged CSV load karta hai.
    """
    df = pd.read_csv(path)
    return df


def convert_dates(df, date_columns=DATE_COLUMNS):
    """
    Text columns ko datetime format mein convert karta hai.
    """
    df = df.copy()
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def clean_missing_values(df):
    """
    Missing values handle karta hai:
    - High-missing text columns (review comments) drop karta hai
    - Category missing ko 'unknown' se fill karta hai
    - Payment ke chhote missing rows drop karta hai
    """
    df = df.copy()

    cols_to_drop = ["review_comment_title", "review_comment_message"]
    existing_cols = [col for col in cols_to_drop if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)

    df["product_category_name"] = df["product_category_name"].fillna("unknown")
    df["product_category_name_english"] = df["product_category_name_english"].fillna("unknown")

    df = df.dropna(subset=["payment_value"])

    return df


def save_cleaned_data(df, output_path="../data/intermediate/cleaned_orders.csv"):
    """
    Cleaned dataframe ko CSV mein save karta hai.
    """
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")
    print(f"Final shape: {df.shape}")


def run_preprocessing_pipeline(
    input_path="../data/intermediate/merged_orders.csv",
    output_path="../data/intermediate/cleaned_orders.csv",
):
    """
    Poora preprocessing pipeline: load -> dates convert -> clean -> save
    """
    df = load_merged_data(input_path)
    df = convert_dates(df)
    df = clean_missing_values(df)
    save_cleaned_data(df, output_path)
    return df


if __name__ == "__main__":
    run_preprocessing_pipeline()