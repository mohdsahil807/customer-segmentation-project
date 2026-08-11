import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import base64
import joblib
from sklearn.decomposition import PCA

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="SEGMENTIQ - Smart Insights",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. BACKGROUND LOADER
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None

possible_paths = [
    os.path.join(PROJECT_ROOT, "assets", "background.jpg"),
    os.path.join(CURRENT_DIR, "assets", "background.jpg"),
    os.path.join(CURRENT_DIR, "..", "assets", "background.jpg"),
    os.path.join(os.getcwd(), "assets", "background.jpg"),
    "assets/background.jpg"
]

bg_base64 = None
for path in possible_paths:
    if os.path.exists(path):
        bg_base64 = get_base64_of_bin_file(path)
        break

if bg_base64:
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bg_base64}") !important;
        background-size: cover !important;
        background-position: center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        color: #e2e8f0;
    }}
    </style>
    """
else:
    bg_css = """
    <style>
    .stApp {
        background: radial-gradient(circle at 50% 20%, #150b2e 0%, #070814 70%, #03040a 100%) !important;
        color: #e2e8f0;
    }
    </style>
    """

st.markdown(bg_css, unsafe_allow_html=True)

# ==========================================
# 3. ADVANCED CSS
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Rajdhani:wght@500;600;700&family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    section[data-testid="stSidebar"] {
        background: rgba(8, 10, 24, 0.45) !important;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border-right: 1px solid rgba(168, 85, 247, 0.2) !important;
        box-shadow: 10px 0 30px rgba(0, 0, 0, 0.5);
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child,
    div[data-testid="stSidebar"] div[role="radiogroup"] input,
    div[data-testid="stSidebar"] div[role="radiogroup"] [data-testid="stMarkdownContainer"] ~ div,
    div[data-testid="stSidebar"] div[role="radiogroup"] label > div[role="radio"] {
        display: none !important;
        visibility: hidden !important;
        width: 0px !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 8px !important;
    }

    div[data-testid="stSidebar"] label[data-baseweb="radio"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        padding: 10px 16px !important;
        margin-bottom: 6px !important;
        transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease, padding-left 0.2s ease !important;
        cursor: pointer !important;
    }

    div[data-testid="stSidebar"] label[data-baseweb="radio"]:hover {
        background: rgba(168, 85, 247, 0.18) !important;
        border-color: rgba(168, 85, 247, 0.5) !important;
        padding-left: 22px !important;
        box-shadow: 0 0 18px rgba(168, 85, 247, 0.4) !important;
    }

    div[data-testid="stSidebar"] label[data-baseweb="radio"] p {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    div[data-testid="stSidebar"] label[data-baseweb="radio"]:hover p {
        color: #ffffff !important;
    }

    div[data-testid="stSidebar"] label[data-baseweb="radio"][aria-checked="true"] {
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #d946ef 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        box-shadow: 0 0 25px rgba(168, 85, 247, 0.6) !important;
    }

    div[data-testid="stSidebar"] label[data-baseweb="radio"][aria-checked="true"] p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .glow-icon {
        filter: drop-shadow(0px 0px 6px currentColor);
        vertical-align: middle;
        display: inline-block;
    }

    div[data-testid="stSidebar"] button {
        background: rgba(20, 24, 48, 0.5) !important;
        border: 1px solid rgba(168, 85, 247, 0.3) !important;
        color: #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }

    div[data-testid="stSidebar"] button:hover {
        background: rgba(124, 58, 237, 0.4) !important;
        border-color: #a855f7 !important;
        box-shadow: 0 0 20px rgba(168, 85, 247, 0.5) !important;
    }

    .admin-card {
        background: rgba(18, 20, 45, 0.55);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 16px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 0 20px rgba(59, 130, 246, 0.2);
        margin-top: 25px;
    }

    .admin-avatar-container {
        position: relative;
    }

    .admin-avatar {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1d4ed8, #7e22ce);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        color: white;
        border: 2px solid rgba(255, 255, 255, 0.25);
    }

    .online-indicator {
        width: 11px;
        height: 11px;
        background-color: #10b981;
        border-radius: 50%;
        position: absolute;
        bottom: 1px;
        right: 1px;
        border: 2px solid #080a18;
        box-shadow: 0 0 8px #10b981;
    }

    .main-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 26px;
        font-weight: 800;
        letter-spacing: 3px;
        color: #ffffff;
        text-shadow: 0 0 15px rgba(168, 85, 247, 0.8);
        text-align: center;
        margin-bottom: 25px;
    }

    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 1.5px;
        color: #cbd5e1;
        margin-bottom: 12px;
        text-transform: uppercase;
    }

    .delta-up { color: #10b981; text-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }
    .delta-down { color: #f43f5e; text-shadow: 0 0 8px rgba(244, 63, 94, 0.5); }
    label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)
# ==========================================
# 4. FEATHER VECTOR ICON GENERATOR FUNCTION
# ==========================================
def get_feather_icon(icon_name, color="#00f0ff", size=20):
    svgs = {
        "grid": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color}; margin-right: 10px;"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>',
        "pie-chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color}; margin-right: 10px;"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>',
        "bar-chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color}; margin-right: 10px;"><line x1="12" y1="20" x2="12" y2="10"></line><line x1="18" y1="20" x2="18" y2="4"></line><line x1="6" y1="20" x2="6" y2="16"></line></svg>',
        "activity": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color}; margin-right: 10px;"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>',
        "repeat": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color}; margin-right: 10px;"><polyline points="17 1 21 5 17 9"></polyline><path d="M3 11V9a4 4 0 0 1 4-4h14"></path><polyline points="7 23 3 19 7 15"></polyline><path d="M21 13v2a4 4 0 0 1-4 4H3"></path></svg>',
        "map-pin": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color}; margin-right: 10px;"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg>',
        "file-text": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color}; margin-right: 10px;"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>',
        "bell": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color}; margin-right: 10px;"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>',
        "settings": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color}; margin-right: 10px;"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>',
        "users": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color};"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>',
        "dollar": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color};"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>',
        "shopping-bag": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color};"><path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"></path><line x1="3" y1="6" x2="21" y2="6"></line><path d="M16 10a4 4 0 0 1-8 0"></path></svg>',
        "trending-up": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color};"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>',
        "alert-triangle": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="glow-icon" style="color:{color};"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>'
    }
    return svgs.get(icon_name, "")

# ==========================================
# 5. DATA LOADER
# ==========================================
@st.cache_data
def load_all_data():
    segments_path = os.path.join(PROJECT_ROOT, "data", "processed", "customer_segments.csv")
    features_path = os.path.join(PROJECT_ROOT, "data", "processed", "customer_features.csv")

    df = None
    if os.path.exists(segments_path):
        df = pd.read_csv(segments_path)
    elif os.path.exists(features_path):
        df = pd.read_csv(features_path)

    if df is not None:
        cols = {c.lower(): c for c in df.columns}
        cluster_col = None
        for col_name in ['segment_name', 'segment', 'segments', 'cluster_labels', 'cluster']:
            if col_name in cols:
                cluster_col = cols[col_name]
                break

        if cluster_col and cluster_col != 'Cluster':
            df.rename(columns={cluster_col: 'Cluster'}, inplace=True)
        elif 'Cluster' not in df.columns:
            df['Cluster'] = 'High Value'

        if 'monetary' not in df.columns and 'monetary_value' in df.columns:
            df['monetary'] = df['monetary_value']
        if 'frequency' not in df.columns and 'order_count' in df.columns:
            df['frequency'] = df['order_count']
        if 'recency' not in df.columns and 'recency_days' in df.columns:
            df['recency'] = df['recency_days']

        for dim, default_val in [('Dimension 1', 'recency'), ('Dimension 2', 'frequency'), ('Dimension 3', 'monetary')]:
            if dim not in df.columns:
                if default_val in df.columns:
                    df[dim] = df[default_val]
                else:
                    df[dim] = np.random.normal(0, 5, len(df))
        return df
    else:
        np.random.seed(42)
        n = 1000
        clusters = np.random.choice(
            ["High Value", "Loyal Customers", "Potential Loyalists", "At Risk", "New Customers"],
            size=n, p=[0.225, 0.201, 0.193, 0.187, 0.194]
        )
        return pd.DataFrame({
            'customer_id': [f"CUST_{i:04d}" for i in range(n)],
            'Cluster': clusters,
            'Dimension 1': np.random.normal(0, 5, n),
            'Dimension 2': np.random.normal(0, 5, n),
            'Dimension 3': np.random.normal(0, 5, n),
            'recency': np.random.randint(1, 365, n),
            'frequency': np.random.randint(1, 15, n),
            'monetary': np.random.uniform(20, 2000, n),
            'region': np.random.choice(['North America', 'Europe', 'Asia Pacific', 'Latin America'], n)
        })

df = load_all_data()

# Uncapped original features for accurate KPI numbers
df_uncapped = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "customer_features.csv"))

total_customers = len(df)
total_revenue = df_uncapped['monetary'].sum()
total_orders = int(df_uncapped['frequency'].sum())
avg_order_value = (total_revenue / total_orders) if total_orders > 0 else 0
repeat_customers_pct = (df_uncapped['frequency'] > 1).mean() * 100

neon_colors = {
    'High Value': '#00f0ff',
    'Engaged / Recent': '#00ff9f',
    'At Risk / Dissatisfied': '#ff007f',
    'Lost/ Inactive': '#a855f7'
}

# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 25px; padding: 4px;">
            <div style="width: 46px; height: 46px; border-radius: 14px; background: radial-gradient(circle, #c084fc 0%, #6366f1 100%); display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 0 20px rgba(192, 132, 252, 0.6);">
                🧿
            </div>
            <div>
                <div style="font-family: 'Orbitron'; font-weight: 800; font-size: 18px; color: #fff; letter-spacing: 1px;">SEGMENTIQ</div>
                <div style="font-size: 11px; color: #c084fc; letter-spacing: 1px; font-weight: 600;">Smart Insights</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    nav_options = ["Overview", "Insights", "Segments", "Reports"]

    page = st.radio(
        "Navigation",
        nav_options,
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("""
        <div class="admin-card">
            <div class="admin-avatar-container">
                <div class="admin-avatar">👤</div>
                <div class="online-indicator"></div>
            </div>
            <div>
                <div style="color: #ffffff; font-weight: 700; font-size: 14px;">Sahil</div>
                <div style="color: #94a3b8; font-size: 11px;">Deployed by Mohd Sahil</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# KPI RENDER FUNCTION
def render_kpi(col, icon_name, title, val, delta, is_up, color, glow):
    delta_class = "delta-up" if is_up else "delta-down"
    icon_svg = get_feather_icon(icon_name, color, size=24)
    with col:
        st.markdown(f"""
            <div style="background: rgba(18, 20, 45, 0.65); backdrop-filter: blur(12px); border: 1px solid {color}88; border-radius: 14px; padding: 16px 18px; box-shadow: 0 0 18px {glow};">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <div style="color: {color}; font-size: 11px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; text-shadow: 0 0 8px {color};">{title}</div>
                    {icon_svg}
                </div>
                <div style="font-family: 'Rajdhani', sans-serif; font-size: 22px; font-weight: 800; color: #ffffff; text-shadow: 0 0 12px {color}; white-space: nowrap;">{val}</div>
                <div class="{delta_class}" style="font-size: 11px; font-weight: 600; margin-top: 4px;">{delta}</div>
            </div>
        """, unsafe_allow_html=True)

def render_summary_box(text_html):
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">PROJECT NOTES</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: rgba(18, 20, 45, 0.65); backdrop-filter: blur(12px); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 14px; padding: 22px; line-height: 1.8; color: #cbd5e1; font-size: 14px;">
    {text_html}
    </div>
    """, unsafe_allow_html=True)


def overview_summary():
    render_summary_box(f"""
    This dashboard provides a real-time overview of <b style="color:#00f0ff;">{total_customers:,} delivered-order customers</b>, classified into <b style="color:#a855f7;">4 segments</b> based on RFM (Recency, Frequency, Monetary) features. Total lifetime spend, total orders, and repeat purchase rate together indicate overall business health.
    <br><br>
    The 3D visualization plots each segment in a distinct color (5 features compressed into 3 dimensions via PCA, capturing ~88% of the variance), making the spatial separation between clusters clearly visible.
    """)


def segments_summary():
    render_summary_box(f"""
    Segmentation was performed using KMeans clustering (k=4), validated with both the <b style="color:#00f0ff;">Elbow Method</b> and <b style="color:#00ff9f;">Silhouette Score</b>. An initial run produced a suspiciously high silhouette score (0.98) — investigation revealed that extreme outliers (e.g. a single ₹109,312 order) were forming tiny, business-meaningless clusters.
    <br><br>
    To fix this, <b style="color:#ff007f;">IQR-based outlier capping</b> was applied before final clustering, resulting in 4 well-balanced segments (14K–32K customers each) that are directly actionable for marketing teams.
    <br><br>
    The <b style="color:#00f0ff;">High Value</b> segment, despite being the smallest, generates the highest per-customer spending — making it the top priority for retention efforts.
    """)


def insights_summary():
    render_summary_box(f"""
    The most important finding from EDA (Exploratory Data Analysis) is that <b style="color:#ff007f;">delivery time is the strongest driver of customer satisfaction</b> — not price or payment method. Correlation analysis showed a clear negative relationship (-0.30) between delivery delay and review score.
    <br><br>
    This pattern is most visible in the <b style="color:#ff007f;">At Risk/Dissatisfied</b> segment — their average delivery time is the highest across all segments (17 days), and their average review score is just 1.64, compared to 4.5+ for other segments.
    <br><br>
    Another key observation: only <b style="color:#00ff9f;">{repeat_customers_pct:.1f}% of customers make a repeat purchase</b> — representing a major growth opportunity through retention.
    """)


def reports_summary():
    render_summary_box(f"""
    This export contains the final segmented dataset for <b style="color:#a855f7;">{total_customers:,} customers</b>, including RFM features, behavioral metrics, and assigned segment labels.
    <br><br>
    Two key business recommendations emerge: <b style="color:#ff007f;">improve delivery logistics</b> (urgent for the At Risk segment), and build <b style="color:#00ff9f;">retention/win-back campaigns</b> (for the Lost/Inactive segment, whose past experience was positive but who have since gone inactive).
    """)


# ==========================================
# 7. DASHBOARD PAGES
# ==========================================

if page == "Overview":
    st.markdown('<div class="main-title">CUSTOMER SEGMENTATION DASHBOARD</div>', unsafe_allow_html=True)

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    render_kpi(kpi1, "users", "TOTAL CUSTOMERS", f"{total_customers:,}", "Delivered orders only", True, "#00f0ff", "rgba(0, 240, 255, 0.35)")
    render_kpi(kpi2, "dollar", "TOTAL MONETARY", f"₹{total_revenue/100000:.1f}L", "Lifetime spend, all customers", True, "#ff007f", "rgba(255, 0, 127, 0.35)")
    render_kpi(kpi3, "shopping-bag", "TOTAL ORDERS", f"{total_orders:,}", "Sum of frequency", True, "#00ff9f", "rgba(0, 255, 159, 0.35)")
    render_kpi(kpi4, "trending-up", "AVG ORDER VALUE", f"₹{avg_order_value:,.2f}", "Monetary / Orders", True, "#ffb703", "rgba(255, 183, 3, 0.35)")
    render_kpi(kpi5, "alert-triangle", "REPEAT CUSTOMERS", f"{repeat_customers_pct:.1f}%", "Ordered more than once", False, "#a855f7", "rgba(168, 85, 247, 0.35)")

    st.markdown("<br>", unsafe_allow_html=True)
    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown('<div class="section-header">CUSTOMER SEGMENT CLUSTERS (3D VIEW)</div>', unsafe_allow_html=True)
        fig_3d = px.scatter_3d(
            df, x='Dimension 1', y='Dimension 2', z='Dimension 3',
            color='Cluster', color_discrete_map=neon_colors, opacity=0.85
        )

# Glow effect: bada, halka transparent marker layer
        fig_3d.update_traces(
    marker=dict(
        size=6,
        opacity=0.9,
        line=dict(width=0),
    ),
    selector=dict(mode='markers')
)

# Extra glow halo layer - same points, bigger and more transparent
        import plotly.graph_objects as go
        for segment_name, color in neon_colors.items():
            segment_data = df[df['Cluster'] == segment_name]
            if len(segment_data) > 0:
                fig_3d.add_trace(go.Scatter3d(
                    x=segment_data['Dimension 1'],
                    y=segment_data['Dimension 2'],
                    z=segment_data['Dimension 3'],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color,
                        opacity=0.03,
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        fig_3d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(backgroundcolor="rgba(10, 12, 30, 0.4)", gridcolor="#2e1065"),
                yaxis=dict(backgroundcolor="rgba(10, 12, 30, 0.4)", gridcolor="#2e1065"),
                zaxis=dict(backgroundcolor="rgba(10, 12, 30, 0.4)", gridcolor="#2e1065"),
            ),
            legend=dict(font=dict(color="#e2e8f0"), bgcolor="rgba(15,18,38,0.5)"),
            margin=dict(l=0, r=0, b=0, t=0), height=480
        )
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">SEGMENT DISTRIBUTION</div>', unsafe_allow_html=True)
        counts = df['Cluster'].value_counts().reset_index()
        counts.columns = ['Cluster', 'Count']

        fig_donut = px.pie(
            counts, names='Cluster', values='Count', hole=0.65,
            color='Cluster', color_discrete_map=neon_colors
        )
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
            annotations=[dict(text=f'{total_customers:,}<br><span style="font-size:11px;color:#94a3b8;">Total Customers</span>', x=0.5, y=0.5, font_size=17, font_color="#ffffff", showarrow=False)],
            margin=dict(l=10, r=10, b=10, t=10), height=200
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown('<div style="text-align:center; font-size:11px; font-weight:700; color:#cbd5e1; margin-bottom: -15px;">REPEAT RATE</div>', unsafe_allow_html=True)
            fig_g1 = go.Figure(go.Indicator(
                mode="gauge+number", value=repeat_customers_pct,
                number={'suffix': "%", 'font': {'color': "#ff007f", 'size': 22}},
                gauge={'axis': {'range': [None, 10]}, 'bar': {'color': "#ff007f"}, 'bgcolor': "rgba(15,18,38,0.4)"}
            ))
            fig_g1.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=140, margin=dict(l=10, r=10, b=10, t=20))
            st.plotly_chart(fig_g1, use_container_width=True)

        with g_col2:
            st.markdown('<div style="text-align:center; font-size:11px; font-weight:700; color:#cbd5e1; margin-bottom: -15px;">AVG REVIEW SCORE</div>', unsafe_allow_html=True)
            fig_g2 = go.Figure(go.Indicator(
                mode="gauge+number", value=df_uncapped['avg_review_score'].mean(),
                number={'font': {'color': "#00f0ff", 'size': 22}},
                gauge={'axis': {'range': [0, 5]}, 'bar': {'color': "#00f0ff"}, 'bgcolor': "rgba(15,18,38,0.4)"}
            ))
            fig_g2.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=140, margin=dict(l=10, r=10, b=10, t=20))
            st.plotly_chart(fig_g2, use_container_width=True)
    overview_summary()

elif page == "Segments":
    st.markdown('<div class="main-title">CUSTOMER SEGMENTS BREAKDOWN</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    top_segment = df.groupby('Cluster')['monetary'].mean().idxmax()
    render_kpi(k1, "users", "ACTIVE CLUSTERS", "4 Segments", "K-Means, k=4", True, "#00f0ff", "rgba(0, 240, 255, 0.35)")
    render_kpi(k2, "dollar", "TOP SPENDING SEGMENT", top_segment, "Highest avg monetary", True, "#ff007f", "rgba(255, 0, 127, 0.35)")
    render_kpi(k3, "trending-up", "AVG RECENCY", f"{df['recency'].mean():.0f} Days", "Overall Average", False, "#00ff9f", "rgba(0, 255, 159, 0.35)")
    render_kpi(k4, "shopping-bag", "REPEAT CUSTOMERS", f"{repeat_customers_pct:.1f}%", "Ordered more than once", True, "#a855f7", "rgba(168, 85, 247, 0.35)")
    st.markdown("<br>", unsafe_allow_html=True)
    seg_rev = df.groupby('Cluster')['monetary'].mean().reset_index()
    fig_bar = px.bar(seg_rev, x='Cluster', y='monetary', color='Cluster', color_discrete_map=neon_colors)
    fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#fff", showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)


    st.markdown('<div class="section-header">SEGMENT SUMMARY TABLE</div>', unsafe_allow_html=True)
    seg_summary = df.groupby('Cluster').agg(
        customers=('Cluster', 'count'),
        avg_recency=('recency', 'mean'),
        avg_monetary=('monetary', 'mean'),
        avg_frequency=('frequency', 'mean')
    ).reset_index().round(2)
    st.dataframe(seg_summary, use_container_width=True)
    segments_summary()

elif page == "Reports":
    st.markdown('<div class="main-title">EXECUTIVE REPORTS & DATA EXPORTS</div>', unsafe_allow_html=True)
    st.download_button("📥 Download Dataset (CSV)", data=df.to_csv(index=False), file_name="segmentation_report.csv", mime="text/csv")
    reports_summary()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">KEY INSIGHTS</div>', unsafe_allow_html=True)
    st.markdown("- **At Risk / Dissatisfied** segment has the lowest review score, correlated with slowest delivery time.")
    st.markdown("- **High Value** segment drives the most revenue despite being the smallest group.")
    st.markdown(f"- Only **{repeat_customers_pct:.1f}%** of customers are repeat buyers — retention is a major growth opportunity.")

elif page == "Insights":
    st.markdown('<div class="main-title">BUSINESS INSIGHTS & METRICS</div>', unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    render_kpi(b1, "dollar", "TOTAL REVENUE", f"₹{total_revenue/100000:.1f}L", "All-time customer spend", True, "#00f0ff", "rgba(0, 240, 255, 0.35)")
    render_kpi(b2, "activity", "AVG DELIVERY TIME", f"{df_uncapped['avg_delivery_Days'].mean():.1f} Days" if 'avg_delivery_Days' in df_uncapped.columns else "N/A", "Across all orders", False, "#ff007f", "rgba(255, 0, 127, 0.35)")
    render_kpi(b3, "trending-up", "AVG REVIEW SCORE", f"{df_uncapped['avg_review_score'].mean():.2f} / 5" if 'avg_review_score' in df_uncapped.columns else "N/A", "Customer satisfaction", True, "#00ff9f", "rgba(0, 255, 159, 0.35)")
    render_kpi(b4, "repeat", "REPEAT RATE", f"{repeat_customers_pct:.1f}%", "Customers ordering again", False, "#a855f7", "rgba(168, 85, 247, 0.35)")

    st.markdown("<br>", unsafe_allow_html=True)

    col_x, col_y = st.columns(2)

    with col_x:
        st.markdown('<div class="section-header">DELIVERY TIME vs REVIEW SCORE</div>', unsafe_allow_html=True)
        if 'avg_delivery_Days' in df_uncapped.columns and 'avg_review_score' in df_uncapped.columns:
            sample_df = df_uncapped.sample(min(3000, len(df_uncapped)), random_state=42)
            fig_scatter = px.scatter(
                sample_df, x='avg_delivery_Days', y='avg_review_score',
                opacity=0.4, color_discrete_sequence=['#00f0ff']
            )
            fig_scatter.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color="#e2e8f0", height=350,
                xaxis_title="Avg Delivery Days", yaxis_title="Avg Review Score"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Delivery/review columns not found in dataset.")

    with col_y:
        st.markdown('<div class="section-header">SEGMENT-WISE AVG MONETARY</div>', unsafe_allow_html=True)
        seg_monetary = df.groupby('Cluster')['monetary'].mean().reset_index()
        fig_seg = px.bar(
            seg_monetary, x='Cluster', y='monetary',
            color='Cluster', color_discrete_map=neon_colors
        )
        fig_seg.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color="#e2e8f0", height=350, showlegend=False
        )
        st.plotly_chart(fig_seg, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">KEY BUSINESS TAKEAWAYS</div>', unsafe_allow_html=True)

    insight_cols = st.columns(4)
    insights = [
        ("💎", "High Value", "Drives most revenue despite being the smallest segment. Priority for retention.", "#00f0ff"),
        ("⚠️", "At Risk", "Lowest review scores, linked directly to slower delivery times. Needs urgent logistics fix.", "#ff007f"),
        ("🔁", "Repeat Buyers", f"Only {repeat_customers_pct:.1f}% of customers order more than once — biggest growth lever.", "#00ff9f"),
        ("📦", "Delivery Impact", "Delivery speed is the strongest driver of customer satisfaction across all segments.", "#a855f7"),
    ]
    for col, (icon, title, text, color) in zip(insight_cols, insights):
        with col:
            st.markdown(f"""
                <div style="background: rgba(18, 20, 45, 0.65); backdrop-filter: blur(12px); border: 1px solid {color}88; border-radius: 14px; padding: 16px; box-shadow: 0 0 15px {color}33; height: 180px;">
                    <div style="font-size: 22px;">{icon}</div>
                    <div style="color: {color}; font-weight: 700; font-size: 13px; margin-top: 6px; text-shadow: 0 0 8px {color};">{title}</div>
                    <div style="color: #cbd5e1; font-size: 12px; margin-top: 8px; line-height: 1.4;">{text}</div>
                </div>
            """, unsafe_allow_html=True)
            insights_summary()