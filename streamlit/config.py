"""
config.py

Dashboard ke liye shared constants — colors, paths, aur metadata.
Isse app.py mein hardcoded values kam ho jaati hain.
"""

# ============ COLOR PALETTE ============
NEON_COLORS = {
    'High Value': '#00f0ff',
    'Engaged / Recent': '#00ff9f',
    'At Risk / Dissatisfied': '#ff007f',
    'Lost/ Inactive': '#a855f7'
}

# ============ DATA PATHS ============
SEGMENTS_DATA_PATH = "../data/processed/customer_segments.csv"
FEATURES_DATA_PATH = "../data/processed/customer_features.csv"

# ============ MODEL PATHS ============
MODEL_PATH = "../models/saved_models/customer_segment_predictor.pkl"
SCALER_PATH = "../models/saved_models/scaler.pkl"
FEATURE_COLUMNS_PATH = "../models/saved_models/feature_columns.pkl"
CLUSTER_NAMES_PATH = "../models/saved_models/cluster_names.pkl"

# ============ APP METADATA ============
PAGE_TITLE = "SEGMENTIQ - Smart Insights"
PAGE_ICON = "🔮"
AUTHOR_NAME = "Sahil"