import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from engine import SNAPBayesianEngine

st.set_page_config(page_title="SNAP Astro Dashboard", layout="wide")
st.title("🔭 SNAP Astro Dashboard")
st.caption("AAVSO data → preprocessing → SNAP engine → alerts")

engine = SNAPBayesianEngine()

def read_uploaded_file(uploaded_file):
    try:
        uploaded_file.seek(0)
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, delim_whitespace=True)
    except Exception as e:
        st.error(f"File read failed: {e}")
        return None

def load_aavso(df):
    if df is None or df.empty:
        raise ValueError("Empty dataset")

    cols = {c.lower().strip(): c for c in df.columns}
    jd_col = cols.get("jd")
    mag_col = cols.get("mag")

    if jd_col is None or mag_col is None:
        raise ValueError("File must contain JD and mag columns")

    out = df[[jd_col, mag_col]].copy()
    out.columns = ["JD", "mag"]
    out["JD"] = pd.to_numeric(out["JD"], errors="coerce")
    out["mag"] = pd.to_numeric(out["mag"], errors="coerce")
    out = out.dropna(subset=["JD", "mag"]).sort_values("JD")
    return out["JD"].to_numpy(), out["mag"].to_numpy(), out

def detect_events(P, threshold):
    P = np.asarray(P, dtype=float)
    return np.where(P > threshold)

st.sidebar.header("Controls")
threshold = st.sidebar.slider("SNAP threshold", 0.0, 1.0, 0.8, 0.01)
alpha = st.sidebar.slider("Engine alpha", 0.0, 2.0, 0.3, 0.01)
engine.alpha = alpha

uploaded_file = st.file_uploader("Upload AAVSO CSV/TXT", type=["csv", "txt"])

if uploaded_file is None:
    st.info("Please upload a file to continue.")
    st.stop()

raw_df = read_uploaded_file(uploaded_file)
if raw_df is None:
    st.stop()

try:
    JD, mag, cleaned_df = load_aavso(raw_df)
except Exception as e:
    st.error(str(e))
    st.stop()

if len(mag) < 3:
    st.error("Not enough valid rows after cleaning.")
    st.stop()

pi_B, pi_T, I, P = engine.score(mag)
events = detect_events(P, threshold)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Max P", f"{np.max(P):.3f}")
c2.metric("Mean P", f"{np.mean(P):.3f}")
c3.metric("Detected Events", str(len(events)))
c4.metric("Threshold", f"{threshold:.2f}")

st.subheader("📈 Instability Probability")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(JD, P, color="purple", linewidth=1.5, label="P(snap)")
ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
if len(events) > 0:
    ax.scatter(JD[events], P[events], color="orange", s=30, label="Detected")
ax.set_xlabel("JD")
ax.set_ylabel("Probability")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

st.subheader("🌟 Light Curve")
fig2, ax2 = plt.subplots(figsize=(12, 4))
ax2.plot(JD, mag, color="black", linewidth=1.2, label="Magnitude")
ax2.invert_yaxis()
ax2.set_xlabel("JD")
ax2.set_ylabel("Magnitude")
ax2.grid(True, alpha=0.3)
ax2.legend()
st.pyplot(fig2)

st.subheader("🧠 Engine Terms")
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(JD, pi_B, label=r"$\Pi_B$", color="steelblue")
ax3.plot(JD, pi_T, label=r"$\Pi_T$", color="darkgreen")
ax3.plot(JD, I, label=r"$\Pi_I$", color="crimson")
ax3.set_xlabel("JD")
ax3.grid(True, alpha=0.3)
ax3.legend()
st.pyplot(fig3)

st.subheader("🚨 Alerts")
if len(events) > 0:
    st.error("SNAP-like alerts detected")
    st.write(events.tolist())
else:
    st.success("No SNAP-like event detected")

with st.expander("Show cleaned data"):
    st.dataframe(cleaned_df)

csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download cleaned data as CSV",
    data=csv_data,
    file_name="cleaned_aavso.csv",
    mime="text/csv",
)
