import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🧠",
    layout="wide"
)

# ---------------- STYLE ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}
h1, h2, h3 {
    color: #00ffd5;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;'>🧠 AI Diabetes Risk Prediction System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align:center;'>Random Forest | Medical Risk Forecasting</h4>",
    unsafe_allow_html=True
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv(
        "DiaBD Diabetes Dataset for Enhanced Risk Analysis and Research in Bangladesh.csv"
    )

df = load_data()

# ---------------- DATA CLEANING ----------------
df = df[['bmi', 'glucose', 'systolic_bp', 'diabetic']]
df = df.dropna()

# ---------------- FEATURES & LABEL ----------------
X = df[['bmi', 'glucose', 'systolic_bp']]
y = df['diabetic']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- RANDOM FOREST MODEL ----------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("🧪 Patient Input Panel")

bmi = st.sidebar.slider("BMI", 15.0, 45.0, 25.0)
glucose = st.sidebar.slider("Glucose (mg/dL)", 70, 300, 120)
bp = st.sidebar.slider("Systolic BP", 90, 220, 120)

# ---------------- PREDICTION ----------------
input_data = np.array([[bmi, glucose, bp]])
ml_prediction = model.predict(input_data)[0]

rule_based = (
    glucose >= 126 or
    (bmi >= 26 and glucose >= 115) or
    (bp >= 170 and glucose >= 110)
)

final_prediction = "YES" if ml_prediction == 1 or rule_based else "NO"

# ---------------- RESULT ----------------
st.markdown("## 🔍 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    if final_prediction == "YES":
        st.error("⚠️ Diabetes Risk: POSITIVE")
    else:
        st.success("✅ Diabetes Risk: NEGATIVE")

with col2:
    st.metric(
        label="🎯 Model Accuracy",
        value=f"{accuracy * 100:.2f}%",
        delta="Random Forest"
    )

# ---------------- UNIQUE GRAPH ----------------
st.markdown("## 📊 Unique Health Pattern Visualization")

fig = px.parallel_coordinates(
    df.sample(300, random_state=42),
    dimensions=['bmi', 'glucose', 'systolic_bp'],
    color='diabetic',
    color_continuous_scale=px.colors.sequential.Plasma
)

fig.update_layout(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- FOOTER ----------------
st.markdown(
    "<p style='text-align:center;color:gray;'>Built with ❤️ using Random Forest & Streamlit</p>",
    unsafe_allow_html=True
)
