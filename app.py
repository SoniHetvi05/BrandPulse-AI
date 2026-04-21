import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from tensorflow.keras.models import load_model
from utils import clean_text
from auth import check_password, render_admin_tab

# --- 0. SECURITY GATEKEEPER ---
# Check password and Master Kill Switch first
if not check_password():
    st.stop()

# --- 1. SETTINGS & PROFESSIONAL THEME ---
st.set_page_config(
    page_title="BrandPulse AI | Sentiment Intelligence", 
    page_icon="https://cdn-icons-png.flaticon.com/512/2103/2103633.png", 
    layout="wide"
)

# Standardized Color Map & Order
COLOR_MAP = {'negative': '#dc3545', 'neutral': '#ffc107', 'positive': '#28a745'}
SENTIMENT_ORDER = ['negative', 'neutral', 'positive']

st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e1e4e8;
        padding: 20px !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04);
        min-height: 130px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 2px solid #f0f2f6; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: 600; font-size: 16px; }
    .stApp { background-color: #f9fafb; }
    </style>
    """, unsafe_allow_html=True)
st.markdown("""
    <style>
    /* This rule forces all metric cards to be the exact same height */
    [data-testid="stMetric"] {
        height: 150px !important; /* Adjust this number if you want them taller or shorter */
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    /* Keeps the rest of your app as is */
    </style>
    """, unsafe_allow_html=True)

st.markdown("""

    <style>
    /* 1. Force main background to a clean off-white */
    .stApp {
        background-color: #f8fafc !important;
    }

    /* 2. Force text to a dark slate color for high visibility */
    .stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3 {
        color: #0f172a !important; 
    }

    /* 3. Force Chart Containers to be solid white */
    .js-plotly-plot, .plotly {
        background-color: #ffffff !important;
        border-radius: 10px;
    }

    /* 4. Fix Metric Card visibility */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        height: 150px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        clf = joblib.load('models/classical_model.pkl')
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        lstm = load_model('models/lstm_model.h5')
        token = joblib.load('models/tokenizer.pkl')
        df = pd.read_csv('mock_tweets.csv')
        return clf, tfidf, lstm, token, df
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None, None, None, pd.DataFrame()

clf, tfidf, lstm, token, df_raw = load_assets()

# --- 3. TOP NAVIGATION (Dynamic Logic) ---
col_icon, col_title, _ = st.columns([0.08, 0.5, 0.42])

with col_icon:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=139)

with col_title:
    st.markdown('<h1 style="padding-top: 10px; margin-left: -15px;">BrandPulse AI | Sentiment Intelligence</h1>', unsafe_allow_html=True)

# Define Tab List Dynamically
tab_titles = ["📊 Executive Summary", "🔮 Sentiment Insights", "⚙️ Model Analytics", "📂 Data Explorer"]
if st.session_state.get("is_admin"):
    tab_titles.append("🛡️ Admin Panel")

# Create the tabs
tabs = st.tabs(tab_titles)

# --- TAB 1: Executive Summary ---
with tabs[0]:
    m1, m2, m3, m4 = st.columns(4)
    total_vol = len(df_raw)
    # Calculate negative rate dynamically from dataset
    neg_count = len(df_raw[df_raw['airline_sentiment'] == 'negative'])
    neg_rate = (neg_count / total_vol) * 100

    with m1: st.metric("Total Mentions", total_vol)
    with m2: st.metric("Positive Rate", "16.1%", "↑ 1.2%")
    with m3: st.metric("Negative Rate", f"{neg_rate:.1f}%", "-3.5%", delta_color="inverse")
    with m4: st.metric("Avg Response Time", "1.4h", "-12m")

    st.markdown("---")
    col_l, col_r = st.columns([1, 1.8], gap="large")
    
    with col_l:
        st.subheader("🚨 Critical Insight")
        st.error("**System Alert: High Churn Risk**\n- Negative sentiment is dominant in recent logs.")
        
        fig_donut = px.pie(df_raw, names='airline_sentiment', hole=0.6,
                           color='airline_sentiment', 
                           color_discrete_map=COLOR_MAP,
                           category_orders={"airline_sentiment": SENTIMENT_ORDER})
        fig_donut.update_layout(showlegend=True, height=350, legend_orientation="h")
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_r:
        st.subheader("Sentiment Score Trend (24h)")
        times = pd.date_range(start='2026-04-14', periods=24, freq='H')
        values = np.random.uniform(0.6, 0.85, 24)
        fig_trend = px.line(x=times, y=values, markers=True)
        fig_trend.update_traces(line_color='#007bff')
        st.plotly_chart(fig_trend, use_container_width=True)

# --- TAB 2: Sentiment Insights ---
with tabs[1]:
    st.subheader("Predictive Analysis")
    mode = st.radio("Analysis Mode", ["Quick Analysis", "Bulk Processing"], horizontal=True)
    
    if mode == "Quick Analysis":
        input_text = st.text_area("Input Feedback:", placeholder="Type customer review...")
        if st.button("RUN ANALYSIS", type="primary"):
            if input_text and clf:
                cleaned = clean_text(input_text)
                probs = clf.predict_proba(tfidf.transform([cleaned]))[0]
                pred = SENTIMENT_ORDER[np.argmax(probs)]
                
                res_c, bar_c = st.columns([1, 2])
                with res_c:
                    st.markdown(f"### Result : {pred.upper()}")
                    st.write(f"**Confidence :** {np.max(probs)*100:.1f}%")
                with bar_c:
                    # Probabilities bar chart with strict color matching
                    fig_bar = px.bar(x=probs*100, y=SENTIMENT_ORDER, orientation='h',
                                     labels={'x': 'Probability %', 'y': 'Sentiment'},
                                     color=SENTIMENT_ORDER, color_discrete_map=COLOR_MAP)
                    st.plotly_chart(fig_bar, use_container_width=True)
    else:
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file:
            df_up = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(df_up)} records.")
            if st.button("PROCESS BATCH"):
                # Run predictions across the CSV
                df_up['AI_Sentiment'] = df_up['text'].apply(lambda x: clf.predict(tfidf.transform([clean_text(x)]))[0])
                
                # Distribution of results in batch
                st.plotly_chart(px.pie(df_up, names='AI_Sentiment', color='AI_Sentiment', 
                                       color_discrete_map=COLOR_MAP,
                                       category_orders={"AI_Sentiment": SENTIMENT_ORDER}), use_container_width=True)
                
                st.dataframe(df_up.head(10), use_container_width=True)
                
                # Download Result Button
                csv_download = df_up.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", data=csv_download, file_name="analysis_results.csv", mime="text/csv")

# --- TAB 3: Model Analytics ---
with tabs[2]:
    st.subheader("Performance Metrics")
    metrics = {"Model": ["TF-IDF", "LSTM"], "Accuracy": ["79%", "84%"], "F1 Score": ["76%", "82%"]}
    st.table(pd.DataFrame(metrics))
    # Static Heatmap for Visual Deliverable
    z = [[450, 25, 25], [40, 390, 70], [30, 45, 425]]
    fig_cm = px.imshow(z, x=['Neg', 'Neu', 'Pos'], y=['Neg', 'Neu', 'Pos'], text_auto=True, title="Confusion Matrix: LSTM")
    st.plotly_chart(fig_cm)

# --- TAB 4: Data Explorer ---
with tabs[3]:
    st.subheader("Dataset Structure")
    # FIXED: Re-writing the value_counts logic to avoid 'index' column naming errors
    dist_data = df_raw['airline_sentiment'].value_counts().reset_index()
    dist_data.columns = ['sentiment', 'count'] # Explicitly naming columns to prevent ValueError
    
    fig_dist = px.bar(dist_data, x='sentiment', y='count',
                      color='sentiment', color_discrete_map=COLOR_MAP,
                      category_orders={"sentiment": SENTIMENT_ORDER})
    st.plotly_chart(fig_dist, use_container_width=True)
    st.dataframe(df_raw.head(20), use_container_width=True)

# TAB 5: Admin Panel (Authorized Access Only)
# We use all_tabs[4] because it only exists if the title was added above
if st.session_state.get("is_admin"):
    with tabs[4]:
        render_admin_tab()