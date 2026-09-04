# ==============================================================================
# 🔥 SNAP ENGINE vULTIMATE (CLEAN + REAL DATA + STABLE)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io

# ==============================================================================
# 🧠 ENGINE
# ==============================================================================

class SnapEngineFinal:

    def normalize(self, x):
        x = np.array(x, dtype=float)
        return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-9)

    def compute_field(self, B, T):
        B = self.normalize(B)
        T = self.normalize(T)

        PI = B - T

        # smoothing (core)
        PI = pd.Series(PI).rolling(7, center=True).mean()
        PI = PI.bfill().ffill().values

        return PI

    def gradient(self, x):
        g = np.gradient(x)
        return np.clip(g, -1, 1)

    def probability(self, PI, S):
        return 1 / (1 + np.exp(-6 * (PI + 0.5*S)))

    def detect(self, P, threshold=0.9):
        idx = np.where(P > threshold)[0]

        clean = []
        for i in idx:
            if i > 3:
                if P[i] > P[i-1] > P[i-2]:
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
# 🧹 DATA CLEANING (REAL FIX)
# ==============================================================================

def remove_outliers(jd, mag):

    m = np.mean(mag)
    s = np.std(mag)

    mask = (mag > m - 2*s) & (mag < m + 2*s)

    return jd[mask], mag[mask]


def smooth_data(jd, mag):

    df = pd.DataFrame({"jd": jd, "mag": mag})

    # group by day (important for astronomy)
    df["jd_round"] = df["jd"].round(0)

    df = df.groupby("jd_round")["mag"].mean().reset_index()

    # rolling smoothing
    df["mag"] = df["mag"].rolling(5).mean()

    df = df.dropna()

    return df["jd_round"].values, df["mag"].values


# ==============================================================================
# 🎨 UI
# ==============================================================================

st.set_page_config(page_title="SNAP Engine Ultimate", layout="wide")

st.title("🌌 SNAP ENGINE vULTIMATE (Real + Clean + Stable)")

uploaded_file = st.file_uploader("📂 Upload CSV or ZIP", type=["csv", "zip"])

if uploaded_file:

    # ---------------- LOAD ----------------
    jd, mag = load_file(uploaded_file)

    # ---------------- CLEAN ----------------
    jd, mag = remove_outliers(jd, mag)
    jd, mag = smooth_data(jd, mag)

    # ---------------- CONVERT ----------------
    B = 10 ** (-0.4 * mag)
    T = np.full_like(B, np.mean(B))

    # ---------------- ENGINE ----------------
    engine = SnapEngineFinal()

    PI = engine.compute_field(B, T)
    S = engine.gradient(PI)
    P = engine.probability(PI, S)

    events = engine.detect(P)
    conf = engine.confidence(P)

    # ==============================================================================
    # 📊 GRAPH 1: PROBABILITY
    # ==============================================================================

    st.subheader("📈 SNAP Probability")

    fig1, ax1 = plt.subplots()
    ax1.plot(jd, P)
    ax1.axhline(0.9, linestyle="--")
    ax1.set_xlabel("JD")
    ax1.set_ylabel("Probability")
    ax1.grid(True)

    if len(events) > 0:
        ax1.scatter(jd[events], P[events])

    st.pyplot(fig1)

    # ==============================================================================
    # 📊 GRAPH 2: LIGHT CURVE (CLEAN)
    # ==============================================================================

    st.subheader("🌟 Clean Light Curve")

    fig2, ax2 = plt.subplots()
    ax2.plot(jd, mag)
    ax2.invert_yaxis()
    ax2.set_xlabel("JD")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True)

    st.pyplot(fig2)

    # ==============================================================================
    # 📊 RESULTS
    # ==============================================================================

    st.subheader("🚨 RESULTS")

    st.write("Data Points:", len(jd))
    st.write("Detected Events:", len(events))
    st.write("Max Probability:", float(np.max(P)))
    st.write("Confidence:", conf)

    if len(events) > 0:
        st.error("🔥 REAL SNAP SIGNAL DETECTED")
    else:
        st.success("✅ SYSTEM STABLE")