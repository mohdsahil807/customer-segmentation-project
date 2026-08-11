"""
utils.py

Common helper functions jo poore project mein reuse hote hain —
logging, file I/O helpers, aur outlier capping (jo clustering.py mein bhi use hota hai).
"""

import os
import pandas as pd


def cap_outliers(series, factor=1.5):
    """
    IQR method se ek pandas Series ke extreme outliers ko cap karta hai.
    Values ko delete nahi karta, sirf lower/upper limit tak seemit karta hai.

    Parameters:
        series (pd.Series): jis column pe capping karni hai
        factor (float): IQR multiplier, standard practice hai 1.5

    Returns:
        pd.Series: capped series
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return series.clip(lower=lower, upper=upper)


def ensure_directory_exists(path):
    """
    Agar diya gaya folder path exist nahi karta, use bana deta hai.
    Isse save operations se pehle "folder not found" errors avoid hote hain.
    """
    os.makedirs(path, exist_ok=True)


def print_shape_log(df, step_name):
    """
    Pipeline ke har step ke baad dataframe ka shape print karta hai —
    debugging aur progress tracking ke liye useful.
    """
    print(f"[{step_name}] Shape: {df.shape}")


def check_missing_values(df, threshold=0):
    """
    Dataframe ke har column mein missing values ka count aur % return karta hai,
    sirf un columns ke liye jinme missing count threshold se zyada hai.

    Returns:
        pd.DataFrame: column-wise missing summary
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100

    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_percent": missing_percent.round(2),
    })

    summary = summary[summary["missing_count"] > threshold].sort_values(
        by="missing_percent", ascending=False
    )
    return summary