# ==============================================================================
# SNAP ENGINE FULL DASHBOARD (UPLOAD + GRAPH + REAL DATA)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io

# ==============================================================================
# 🔥 SNAP ENGINE FINAL
# ==============================================================================

class SnapEngineFinal:

    def normalize(self, x):
        x = np.array(x, dtype=float)
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)

    def compute_field(self, B, T):
        B = self.normalize(B)
        T = self.normalize(T)
        PI_I = B - T
        PI_I = pd.Series(PI_I).rolling(5, center=True).mean()
        PI_I = PI_I.bfill().ffill().values
        return PI_I

    def safe_gradient(self, x):
        g = np.gradient(x)
        return np.clip(g, -1, 1)

    def probability(self, PI_I, S):
        return 1 / (1 + np.exp(-8 * (PI_I + 0.5*S)))

    def detect(self, P, threshold=0.9):
        idx = np.where(P > threshold)[0]
        clean = []
        for i in idx:
            if i > 2 and P[i] > P[i-1] > P[i-2]:
                clean.append(i)
        return np.array(clean)

    def confidence(self, P):
        return float((1 / (1 + np.std(P[-20:]))) * np.max(P))


# ==============================================================================
# 📂 FILE LOADER
# ==============================================================================

def load_file(uploaded_file):

    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                df = pd.read_csv(io.BytesIO(f.read()))
    else:
        df = pd.read_csv(uploaded_file)

    df.columns = [c.strip().lower() for c in df.columns]

    jd_col = next((c for c in df.columns if "jd" in c), None)
    mag_col = next((c for c in df.columns if "mag" in c), None)

    jd = pd.to_numeric(df[jd_col], errors="coerce")
    mag = pd.to_numeric(df[mag_col], errors="coerce")

    mask = (~np.isnan(jd)) & (~np.isnan(mag))

    return jd[mask].values, mag[mask].values


# ==============================================================================
# 🎨 STREAMLIT UI
# ==============================================================================

st.set_page_config(page_title="SNAP Engine", layout="wide")

st.title("🌌 SNAP ENGINE (Upload + Real Data + Graph)")

uploaded_file = st.file_uploader("📂 Upload CSV or ZIP file", type=["csv", "zip"])

if uploaded_file:

    jd, mag = load_file(uploaded_file)

    # convert to dimensionless
    B = 10 ** (-0.4 * mag)
    T = np.full_like(B, np.mean(B))

    engine = SnapEngineFinal()

    PI_I = engine.compute_field(B, T)
    S = engine.safe_gradient(PI_I)
    P = engine.probability(PI_I, S)
    events = engine.detect(P)
    conf = engine.confidence(P)

    # ==============================================================================
    # 📊 CHART (USING GENUI)
    # ==============================================================================

    data_chart = []
    for i in range(len(jd)):
        data_chart.append({
            "jd": float(jd[i]),
            "probability": float(P[i])
        })

    st.subheader("📈 SNAP Probability Curve")

    

    # ==============================================================================
    # 📊 LIGHT CURVE
    # ==============================================================================

    st.subheader("🌟 Light Curve")

    fig, ax = plt.subplots()
    ax.plot(jd, mag)
    ax.invert_yaxis()
    st.pyplot(fig)

    # ==============================================================================
    # 📊 RESULTS
    # ==============================================================================

    st.subheader("🚨 RESULTS")

    st.write("Total Data Points:", len(jd))
    st.write("Detected Events:", len(events))
    st.write("Max Probability:", float(np.max(P)))
    st.write("Confidence:", conf)

    if len(events) > 0:
        st.error("🔥 SNAP-like instability detected")
    else:
        st.success("✅ System stable")