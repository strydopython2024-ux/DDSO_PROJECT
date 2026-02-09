import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import time

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI DDoS Detection",
    page_icon="🛡️",
    layout="wide"
)

# ==================== CYBER ANIMATED CSS ====================
st.markdown("""
<style>

/* -------- BACKGROUND ANIMATION -------- */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.stApp {
    background: linear-gradient(-45deg, #020024, #050a30, #000000, #020024);
    background-size: 400% 400%;
    animation: gradientBG 18s ease infinite;
    color: white;
}

/* -------- FLOAT ANIMATION -------- */
@keyframes float {
    0% {transform: translateY(0px);}
    50% {transform: translateY(-12px);}
    100% {transform: translateY(0px);}
}

/* -------- SCAN LINE -------- */
@keyframes scan {
    0% {top: -10%;}
    100% {top: 110%;}
}

.scan-line {
    position: fixed;
    width: 100%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00f5d4, transparent);
    animation: scan 6s linear infinite;
    opacity: 0.6;
    z-index: 0;
}

/* -------- TITLES -------- */
.main-title {
    font-size: 50px;
    font-weight: 900;
    color: #00f5d4;
    text-align: center;
    animation: float 4s ease-in-out infinite;
}

.sub-title {
    font-size: 20px;
    color: #9ca3af;
    text-align: center;
}

/* -------- CARDS -------- */
.card {
    background: rgba(22, 27, 34, 0.9);
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 0 25px rgba(0,245,212,0.25);
    transition: all 0.4s ease;
    animation: float 6s ease-in-out infinite;
}

.card:hover {
    transform: translateY(-10px) scale(1.05);
    box-shadow: 0 0 40px rgba(0,245,212,0.7);
}

/* -------- TABLE GLOW -------- */
[data-testid="stDataFrame"] {
    background: rgba(0,0,0,0.4);
    border-radius: 12px;
    box-shadow: 0 0 15px rgba(0,245,212,0.2);
}

/* -------- FOOTER -------- */
.footer {
    text-align: center;
    color: #9ca3af;
    margin-top: 50px;
    animation: float 5s ease-in-out infinite;
}

</style>

<div class="scan-line"></div>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("efficientstnet_best_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ==================== LABEL MAP (SAFE) ====================
label_map = {
    0: "BENIGN",
    1: "DNS ATTACK",
    2: "NetBIOS ATTACK",
    3: "SYN FLOOD",
    4: "UDP FLOOD"
}

# ==================== HEADER ====================
st.markdown('<div class="main-title">🛡️ AI-Powered DDoS Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">EfficientSTNet | CNN + LSTM | Cyber Threat Intelligence</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Control Panel")
show_raw = st.sidebar.checkbox("Show raw uploaded data")
confidence_view = st.sidebar.checkbox("Show prediction confidence")

# ==================== FILE UPLOADER ====================
uploaded_file = st.file_uploader("📂 Upload Network Traffic CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if show_raw:
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # ==================== PROCESS ====================
    with st.spinner("🔍 Analyzing network traffic patterns..."):
        time.sleep(1.2)

        X = scaler.transform(df)
        X = X.reshape(X.shape[0], 7, 11)

        preds = model.predict(X)
        classes = np.argmax(preds, axis=1)

    st.success("✅ Analysis completed successfully")

    # ==================== RESULTS ====================
    results = pd.DataFrame()
    results["Prediction"] = [
        label_map.get(int(c), f"UNKNOWN ({int(c)})")
        for c in classes
    ]

    st.markdown("## 🚨 Detection Results")

    col1, col2, col3 = st.columns(3)

    total = len(results)
    attacks = results[results["Prediction"] != "BENIGN"].shape[0]
    benign = total - attacks

    col1.markdown(f"<div class='card'>📊 <b>Total Flows</b><br><h2>{total}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'>🚨 <b>Attacks Detected</b><br><h2>{attacks}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'>🟢 <b>Benign Traffic</b><br><h2>{benign}</h2></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.dataframe(results, use_container_width=True)

    # ==================== CONFIDENCE VIEW ====================
    if confidence_view:
        st.subheader("📈 Prediction Confidence")
        confidence_df = pd.DataFrame(
            preds,
            columns=[f"Class_{i}" for i in range(preds.shape[1])]
        )
        st.dataframe(confidence_df.head(), use_container_width=True)

# ==================== FOOTER ====================
st.markdown("""
<div class="footer">
Built with ❤️ using Deep Learning & Streamlit<br>
EfficientSTNet – Multi-Class DDoS Detection
</div>
""", unsafe_allow_html=True)
