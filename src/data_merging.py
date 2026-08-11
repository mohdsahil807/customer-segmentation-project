"""
data_merging.py

Raw Olist CSV files ko merge karke ek single flat table banata hai.
Isko 01_Data_Merging.ipynb notebook se convert kiya gaya hai.
"""

import pandas as pd


def load_raw_data(raw_path="../data/raw/"):
    """
    Saari 7 raw CSV files load karta hai.

    Parameters:
        raw_path (str): raw data folder ka path

    Returns:
        dict: har dataset ka naam -> DataFrame mapping
    """
    data = {
        "customers": pd.read_csv(f"{raw_path}olist_customers_dataset.csv"),
        "orders": pd.read_csv(f"{raw_path}olist_orders_dataset.csv"),
        "order_items": pd.read_csv(f"{raw_path}olist_order_items_dataset.csv"),
        "payments": pd.read_csv(f"{raw_path}olist_order_payments_dataset.csv"),
        "reviews": pd.read_csv(f"{raw_path}olist_order_reviews_dataset.csv"),
        "products": pd.read_csv(f"{raw_path}olist_products_dataset.csv"),
        "category_translation": pd.read_csv(f"{raw_path}product_category_name_translation.csv"),
    }
    return data


def merge_all(data):
    """
    Saare datasets ko step-by-step merge karta hai ek single flat table mein.

    Parameters:
        data (dict): load_raw_data() se aaya dictionary

    Returns:
        pd.DataFrame: merged dataset
    """
    df = data["orders"].merge(data["customers"], on="customer_id", how="left")
    df = df.merge(data["order_items"], on="order_id", how="left")

    products = data["products"].merge(
        data["category_translation"], on="product_category_name", how="left"
    )
    df = df.merge(products, on="product_id", how="left")

    df = df.merge(data["payments"], on="order_id", how="left")
    df = df.merge(data["reviews"], on="order_id", how="left")

    return df


def save_merged_data(df, output_path="../data/intermediate/merged_orders.csv"):
    """
    Merged dataframe ko CSV mein save karta hai.
    """
    df.to_csv(output_path, index=False)
    print(f"Merged data saved to {output_path}")
    print(f"Final shape: {df.shape}")


def run_merging_pipeline(raw_path="../data/raw/", output_path="../data/intermediate/merged_orders.csv"):
    """
    Poora merging pipeline ek function call se chalata hai:
    load -> merge -> save

    Returns:
        pd.DataFrame: merged dataset
    """
    data = load_raw_data(raw_path)
    df = merge_all(data)
    save_merged_data(df, output_path)
    return df


if __name__ == "__main__":
    run_merging_pipeline()