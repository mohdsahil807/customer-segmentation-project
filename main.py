"""
main.py

End-to-end pipeline runner — raw data se leke trained model tak,
ek command se poora pipeline chalata hai.

Usage:
    python main.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_merging import run_merging_pipeline
from data_preprocessing import run_preprocessing_pipeline
from feature_engineering import run_feature_engineering_pipeline
from clustering import run_clustering_pipeline


def run_full_pipeline():
    """
    Poora pipeline sequentially chalata hai:
    1. Data Merging
    2. Data Preprocessing
    3. Feature Engineering
    4. Clustering (KMeans + model saving)
    """
    print("=" * 60)
    print("STEP 1: Data Merging")
    print("=" * 60)
    run_merging_pipeline(
        raw_path="data/raw/",
        output_path="data/intermediate/merged_orders.csv",
    )

    print("\n" + "=" * 60)
    print("STEP 2: Data Preprocessing")
    print("=" * 60)
    run_preprocessing_pipeline(
        input_path="data/intermediate/merged_orders.csv",
        output_path="data/intermediate/cleaned_orders.csv",
    )

    print("\n" + "=" * 60)
    print("STEP 3: Feature Engineering")
    print("=" * 60)
    run_feature_engineering_pipeline(
        input_path="data/intermediate/cleaned_orders.csv",
        output_path="data/processed/customer_features.csv",
    )

    print("\n" + "=" * 60)
    print("STEP 4: Clustering & Model Training")
    print("=" * 60)
    run_clustering_pipeline(
        input_path="data/processed/customer_features.csv",
        output_path="data/processed/customer_segments.csv",
        model_dir="models/saved_models/",
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE! 🎉")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()