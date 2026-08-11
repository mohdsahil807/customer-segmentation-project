"""
generate_report.py

Saare EDA/model figures aur business insights ko ek PDF report mein compile karta hai.
Run karne ka tarika: notebooks/ folder ke andar se -> python generate_report.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

FIGURES_DIR = "../reports/figures/"
OUTPUT_PATH = "../reports/report.pdf"

# Sirf .png figures include karenge (HTML interactive files PDF mein nahi chalti)
FIGURE_ORDER = [
    ("payment_value_distribution.png", "Payment Value Distribution — right-skewed, most transactions are low value"),
    ("delivery_time_distribution.png", "Delivery Time Distribution — most orders deliver within 5-15 days"),
    ("delivery_vs_review.png", "Delivery Time vs Review Score — clear negative relationship"),
    ("monthly_orders_trend.png", "Monthly Orders Trend — steady growth with a seasonal spike"),
    ("correlation_heatmap.png", "Correlation Heatmap — delivery time most correlated with review score"),
    ("payment_type_distribution.png", "Payment Type Distribution — credit card dominates"),
    ("orders_by_state.png", "Orders by Customer State — São Paulo leads by a wide margin"),
    ("rfm_boxplots.png", "RFM Boxplots — Monetary and Frequency show heavy outliers"),
    ("elbow_method_capped.png", "Elbow Method (after outlier capping) — supports k=4"),
    ("silhouette_scores_capped.png", "Silhouette Scores (after capping) — balanced clusters at k=4"),
]


def add_title_page(pdf):
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.5, 0.6, "Customer Segmentation Analysis", ha="center", fontsize=24, weight="bold")
    fig.text(0.5, 0.53, "Olist E-Commerce Dataset — RFM-Based Clustering Report", ha="center", fontsize=13)
    fig.text(0.5, 0.15, "Prepared by Sahil", ha="center", fontsize=11, color="gray")
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def add_summary_page(pdf):
    summary_text = (
        "Executive Summary\n\n"
        "This report presents a customer segmentation analysis of 93,357 delivered-order\n"
        "customers from the Olist e-commerce platform, based on RFM (Recency, Frequency,\n"
        "Monetary) features plus review score and delivery time.\n\n"
        "Four segments were identified using KMeans clustering (k=4), validated with the\n"
        "Elbow Method and Silhouette Score: High Value, Engaged/Recent, At Risk/Dissatisfied,\n"
        "and Lost/Inactive.\n\n"
        "Key Finding: Delivery time is the strongest driver of customer satisfaction,\n"
        "showing a -0.30 correlation with review score. The At Risk/Dissatisfied segment\n"
        "has the slowest average delivery time (17 days) and the lowest review score (1.64),\n"
        "confirming this relationship.\n\n"
        "Only 3% of customers make a repeat purchase, highlighting retention as a major\n"
        "growth opportunity.\n\n"
        "Recommendations:\n"
        "  1. Improve delivery logistics, prioritizing the At Risk segment.\n"
        "  2. Launch win-back campaigns for the Lost/Inactive segment.\n"
        "  3. Build a loyalty program for the High Value segment.\n"
    )
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.95, summary_text, ha="left", va="top", fontsize=11, wrap=True)
    plt.axis("off")
    pdf.savefig(fig)
    plt.close(fig)


def add_figure_page(pdf, image_path, caption):
    fig = plt.figure(figsize=(8.5, 11))
    img = mpimg.imread(image_path)
    ax = fig.add_axes([0.05, 0.15, 0.9, 0.75])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.5, 0.08, caption, ha="center", fontsize=10, wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def generate_report():
    with PdfPages(OUTPUT_PATH) as pdf:
        add_title_page(pdf)
        add_summary_page(pdf)

        for filename, caption in FIGURE_ORDER:
            path = os.path.join(FIGURES_DIR, filename)
            if os.path.exists(path):
                add_figure_page(pdf, path, caption)
            else:
                print(f"Skipped (not found): {filename}")

    print(f"Report generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_report()