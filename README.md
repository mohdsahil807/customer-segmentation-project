# Customer Segmentation — Olist E-Commerce

An end-to-end data science project that segments 93,357 Olist e-commerce customers into actionable business groups using RFM analysis and KMeans clustering, presented through an interactive Streamlit dashboard.

## 🎯 Project Overview

This project analyzes the [Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) to identify distinct customer segments based on purchasing behavior. The goal is to help the business understand which customers are most valuable, which are at risk of churning, and where to focus retention efforts.

## 🔑 Key Finding

**Delivery time is the strongest driver of customer satisfaction** — not price or payment method. Correlation analysis showed a -0.30 relationship between delivery delay and review score. The "At Risk/Dissatisfied" segment has the slowest average delivery time (17 days) and lowest review score (1.64 / 5), directly confirming this pattern.

## 📊 Customer Segments

| Segment | Size | Profile |
|---|---|---|
| **High Value** | 14,530 (15.6%) | Highest spenders, good satisfaction — top retention priority |
| **Engaged / Recent** | 32,381 (34.7%) | Most recent activity, fastest delivery, best reviews |
| **At Risk / Dissatisfied** | 15,309 (16.4%) | Lowest reviews, slowest delivery — needs urgent attention |
| **Lost / Inactive** | 31,137 (33.3%) | Long time since last order, but had a good past experience |

## 🛠️ Methodology

1. **Data Merging** — Combined 7 relational Olist CSV files into a single flat table (119,143 rows)
2. **Data Preprocessing** — Handled missing values, converted date types, removed low-value columns
3. **EDA** — Explored distributions, correlations, and time trends across 7 visualizations
4. **Feature Engineering** — Built RFM (Recency, Frequency, Monetary) features plus average review score and delivery time per customer
5. **Model Training** — Discovered that extreme outliers (e.g. a single ₹109,312 order) were distorting KMeans clustering; applied IQR-based outlier capping before training
6. **Clustering** — KMeans (k=4), validated with the Elbow Method and Silhouette Score
7. **Dashboard** — Built an interactive Streamlit app to visualize segments and business metrics

## 📁 Project Structure


## 🚀 How to Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the notebooks in order** (01 → 06) to regenerate the data pipeline, or use the pre-processed data already included in `data/processed/`.

**3. Launch the dashboard:**
```bash
cd streamlit
streamlit run app.py
```

## 📈 Dashboard Features

- Real-time KPIs (total customers, monetary value, orders, repeat rate)
- Interactive 3D cluster visualization (PCA-reduced)
- Segment distribution donut chart and comparison bar charts
- Business insights and recommendations per page

## 🧰 Tech Stack

Python · Pandas · NumPy · Scikit-learn · Plotly · Matplotlib · Seaborn · Streamlit

## 👤 Author

**Mohd Sahil**

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
