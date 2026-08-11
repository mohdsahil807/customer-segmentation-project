"""
visualization.py

Reusable plotting functions — EDA charts aur cluster visualizations ke liye.
Isko 03_EDA.ipynb aur 06_Model_Evaluation.ipynb notebooks se convert kiya gaya hai.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.decomposition import PCA


def plot_distribution(df, column, title, xlabel, save_path=None, bins=50):
    """
    Ek numeric column ka histogram + KDE distribution plot banata hai.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_correlation_heatmap(df, numeric_cols, save_path=None):
    """
    Diye gaye numeric columns ka correlation heatmap banata hai.
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_monthly_trend(df, date_column, id_column, save_path=None):
    """
    Monthly order trend (unique orders per month) line chart banata hai.
    """
    df = df.copy()
    df["order_month"] = df[date_column].dt.to_period("M")
    monthly_counts = df.groupby("order_month")[id_column].nunique()

    plt.figure(figsize=(14, 5))
    monthly_counts.plot(kind="line", marker="o")
    plt.title("Monthly Orders Trend")
    plt.xlabel("Month")
    plt.ylabel("Number of Unique Orders")
    plt.xticks(rotation=45)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def create_pca_components(df, feature_columns, scaler, n_components=3, random_state=42):
    """
    Features ko scale karke PCA se n dimensions mein compress karta hai
    (3D visualization ke liye use hota hai).

    Returns:
        pd.DataFrame: original df + PCA1, PCA2, PCA3 columns, aur explained variance ratio
    """
    df = df.copy()
    X = df[feature_columns]
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    for i in range(n_components):
        df[f"PCA{i+1}"] = X_pca[:, i]

    return df, pca.explained_variance_ratio_


def plot_3d_clusters(df, color_column, color_map, title="Customer Segments — 3D Visualization"):
    """
    PCA components use karke interactive 3D scatter plot banata hai, segments ke colors ke saath.
    """
    fig = px.scatter_3d(
        df,
        x="PCA1", y="PCA2", z="PCA3",
        color=color_column,
        color_discrete_map=color_map,
        opacity=0.75,
        title=title,
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(
        template="plotly_white",
        height=700,
        legend_title_text="Segment",
    )
    return fig


def plot_segment_comparison(df, group_column, metric_columns, color_map):
    """
    Har segment ke key metrics (recency, monetary, review, delivery) compare karne ke liye
    ek 2x2 grid mein bar charts banata hai.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    summary = df.groupby(group_column)[metric_columns].mean().reset_index()

    fig = make_subplots(rows=2, cols=2, subplot_titles=metric_columns)
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for metric, pos in zip(metric_columns, positions):
        fig.add_trace(
            go.Bar(
                x=summary[group_column],
                y=summary[metric],
                marker_color=[color_map.get(seg, "#888888") for seg in summary[group_column]],
                showlegend=False,
            ),
            row=pos[0], col=pos[1],
        )

    fig.update_layout(template="plotly_white", height=700, title_text="Segment Comparison")
    return fig