# ==============================================================================
# 🔥 GADE SNAP DASHBOARD v6 (STABLE + NO FAKE EVENTS)
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io

from engine import SNAPBayesianEngine

st.set_page_config(page_title="SNAP Dashboard v6", layout="wide")
st.title("🔭 SNAP Astro Dashboard (Stable Scientific Version)")

engine = SNAPBayesianEngine()

# ==============================================================================
# 📂 FILE LOADER
# ==============================================================================

def read_zip_multi_files(uploaded_file):
    star_datasets = {}
    with zipfile.ZipFile(uploaded_file) as z:
        for f_path in z.namelist():
            if f_path.endswith((".csv", ".txt")):
                with z.open(f_path) as f:
                    df = pd.read_csv(io.BytesIO(f.read()))
                    star_datasets[f_path] = df
    return star_datasets


def load_aavso(df):
    df.columns = [c.strip().lower() for c in df.columns]

    jd_c = next((c for c in df.columns if "jd" in c), None)
    mag_c = next((c for c in df.columns if "mag" in c), None)

    df["jd"] = pd.to_numeric(df[jd_c], errors="coerce")
    df["mag"] = pd.to_numeric(df[mag_c], errors="coerce")

    df = df.dropna().sort_values("jd")

    return df["jd"].values, df["mag"].values


# ==============================================================================
# 🧹 CLEANING (VERY IMPORTANT)
# ==============================================================================

def clean_data(jd, mag):

    # remove outliers
    m, s = np.mean(mag), np.std(mag)
    mask = (mag > m - 2*s) & (mag < m + 2*s)

    jd, mag = jd[mask], mag[mask]

    # group by day
    df = pd.DataFrame({"jd": jd, "mag": mag})
    df["jd_round"] = df["jd"].round(0)
    df = df.groupby("jd_round")["mag"].mean().reset_index()

    # smooth
    df["mag"] = df["mag"].rolling(7).mean()
    df = df.dropna()

    return df["jd_round"].values, df["mag"].values


# ==============================================================================
# ⚙️ CONTROL PANEL
# ==============================================================================

st.sidebar.header("⚙️ Controls")
threshold = st.sidebar.slider("SNAP Threshold", 0.5, 1.0, 0.9)
alpha = st.sidebar.slider("Alpha", 0.0, 2.0, 0.3)
engine.alpha = alpha

uploaded_files = st.file_uploader(
    "Upload CSV or ZIP", type=["csv", "zip"], accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

# ==============================================================================
# 🔄 LOAD DATA
# ==============================================================================

all_data = {}

for uf in uploaded_files:
    if uf.name.endswith(".zip"):
        all_data.update(read_zip_multi_files(uf))
    else:
        all_data[uf.name] = pd.read_csv(uf)

sel_star = st.selectbox("Select Star", list(all_data.keys()))

JD, mag = load_aavso(all_data[sel_star])

# 🔥 CLEAN DATA
JD, mag = clean_data(JD, mag)

# ==============================================================================
# 🧠 ENGINE FIX (IMPORTANT)
# ==============================================================================

# convert magnitude → brightness
B = 10 ** (-0.4 * mag)

# 🔥 dynamic baseline (FIX)
T = pd.Series(B).rolling(30).mean().bfill().ffill().values

# core engine
pi_B, pi_T, I, _ = engine.score(mag)

# 🔥 recompute stable PI
PI = B - T
PI = pd.Series(PI).rolling(7).mean().bfill().ffill().values

# gradient
S = np.gradient(PI)
S = np.clip(S, -1, 1)

# 🔥 soft probability (FIX)
P = 1 / (1 + np.exp(-3 * (PI + 0.3*S)))

# ==============================================================================
# 🚨 SMART DETECTION (NO SPAM)
# ==============================================================================

events = []
for i in range(5, len(P)):
    if (
        P[i] > threshold and
        P[i] > P[i-1] > P[i-2] and
        P[i] - P[i-3] > 0.05
    ):
        events.append(i)

events = np.array(events)

# ==============================================================================
# 📊 METRICS
# ==============================================================================

c1, c2, c3 = st.columns(3)
c1.metric("Max P", f"{np.max(P):.3f}")
c2.metric("Mean P", f"{np.mean(P):.3f}")
c3.metric("Events", len(events))

# ==============================================================================
# 📈 GRAPH 1
# ==============================================================================

st.subheader("📈 SNAP Probability")

fig, ax = plt.subplots(figsize=(12,4))
ax.plot(JD, P)
ax.axhline(threshold, linestyle="--")

if len(events) > 0:
    ax.scatter(JD[events], P[events])

ax.grid(True)
st.pyplot(fig)

# ==============================================================================
# 🌟 GRAPH 2
# ==============================================================================

st.subheader("🌟 Clean Light Curve")

fig2, ax2 = plt.subplots(figsize=(12,4))
ax2.plot(JD, mag)
ax2.invert_yaxis()
ax2.grid(True)
st.pyplot(fig2)

# ==============================================================================
# 🚨 RESULT
# ==============================================================================

st.subheader("🚨 RESULT")

st.write("Data Points:", len(JD))
st.write("Detected Events:", len(events))

if len(events) > 0:
    st.error("🔥 TRUE INSTABILITY DETECTED")
else:
    st.success("✅ STABLE")