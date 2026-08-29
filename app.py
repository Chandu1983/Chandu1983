# ==============================================================================
# GADE UNIVERSAL DIMENSIONLESS BOUNDARY METRIC (FRONTEND VISUAL DASHBOARD)
# Framework Construction: Interface module for engine.py Core
# Lead Inventor: Chandrakant Shivram Gade, Nashik, Maharashtra, India.
# Version: 5.0.0 | Multi-Star Production Blueprint
# ==============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io

# Import the core Gade Engine components directly from your engine.py file
from engine import SNAPBayesianEngine, filter_gade_events

# --- Streamlit Frontend Page Layout Config ---
st.set_page_config(page_title="SNAP Multi-Star Dashboard", layout="wide")
st.title("🔭 SNAP Astro Dashboard (Multi-Star Version v5)")
st.caption("Advanced Preprocessing -> Gade Dimensionless Engine -> Automated Core Selection")

# Initialize Gade Core Engine
engine = SNAPBayesianEngine()

def read_zip_multi_files(uploaded_file):
    """Extracts and parses multi-star CSV or TXT datasets from a ZIP bundle."""
    star_datasets = {}
    with zipfile.ZipFile(uploaded_file) as z:
        for f_path in [f for f in z.namelist() if f.lower().endswith((".csv", ".txt"))]:
            s_name = f_path.split("/")[-1].replace(".csv","").replace(".txt","").strip()
            with z.open(f_path) as f:
                try:
                    df = pd.read_csv(io.BytesIO(f.read()))
                except:
                    df = pd.read_csv(io.BytesIO(f.read()), delim_whitespace=True)
                if not df.empty:
                    star_datasets[s_name] = df
    return star_datasets

def load_aavso(df):
    """Discovers, parses, and cleans AAVSO magnitude columns automatically."""
    df.columns = [str(c).strip().lower() for c in df.columns]
    jd_c = next((c for c in df.columns if "jd" in c or "julian" in c), None)
    mag_c = next((c for c in df.columns if "mag" in c or "magnitude" in c or "flux" in c), None)
    
    if not jd_c or not mag_c:
        raise ValueError("Critical Column Mismatch. Columns must contain JD and Magnitude identifiers.")
    
    # Auto-detect target star name inside file matrix matrix
    star_col = next((c for c in df.columns if "star" in c or "name" in c or "object" in c), None)
    det_star = "Unknown"
    if star_col and not df[star_col].empty:
        det_star = str(df[star_col].iloc[0]).strip()
    
    df["clean_jd"] = pd.to_numeric(df[jd_c], errors="coerce")
    df["clean_mag"] = pd.to_numeric(df[mag_c], errors="coerce")
    
    out = df.dropna(subset=["clean_jd", "clean_mag"]).sort_values("clean_jd")
    out = out.groupby("clean_jd", as_index=False)["clean_mag"].mean()
    return out["clean_jd"].to_numpy(), out["clean_mag"].to_numpy(), det_star

# Sidebar Configuration Control Panel
st.sidebar.header("⚙️ Gade Engine Control Panel")
threshold = st.sidebar.slider("SNAP Probability Threshold", 0.0, 1.0, 0.98, 0.01)
alpha = st.sidebar.slider("Engine Alpha (Tension Weight)", 0.0, 2.0, 0.3, 0.01)
engine.alpha = alpha

all_stars_data = {}

# Simulation vs Upload File Routing Management
if st.sidebar.checkbox("Load Historical T CrB Dataset Simulation", value=False):
    np.random.seed(42)
    s_jd = np.linspace(2431000, 2431200, 150)
    s_mag = 10.0 - 4.5 * np.exp(-((s_jd - 2431185) / 12)**2) + np.random.normal(0, 0.1, 150)
    all_stars_data["T_CrB_1946_Simulated"] = pd.DataFrame({"jd": s_jd, "mag": s_mag, "star": "T CrB Simulated"})
else:
    u_files = st.file_uploader("Upload AAVSO Files (Multiple CSVs or ZIP Archive)", type=["csv", "txt", "zip"], accept_multiple_files=True)
    if u_files:
        for uf in u_files:
            if uf.name.lower().endswith(".zip"):
                zd = read_zip_multi_files(uf)
                if zd:
                    all_stars_data.update(zd)
            else:
                try:
                    df_r = pd.read_csv(uf)
                except:
                    df_r = pd.read_csv(uf, delim_whitespace=True)
                all_stars_data[uf.name.replace(".csv","").replace(".txt","").strip()] = df_r

if not all_stars_data:
    st.info("📌 System Idle. Drop your AAVSO files or check Simulation Mode in the sidebar to fire up the engine.")
    st.stop()

# Interactive Target Selection Interface
sel_star = st.selectbox("Select Target Star From Ingested Database:", list(all_stars_data.keys()))
JD, mag, det_star_internal = load_aavso(all_stars_data[sel_star])

# Gade Automated File Metadata Discovery Banner Display
st.info(f"✨ **Gade Automated File Metadata Discovery Engine**\n\n"
        f"• 🌟 **तारे का नाम (Identified Target):** `{det_star_internal if det_star_internal != 'Unknown' and det_star_internal != 'nan' else sel_star}`\n\n"
        f"• 📅 **डेटा अवधि (Observations Epoch):** From JD `{JD[0]:.4f}` to `{JD[-1]:.4f}` (कुल `{len(JD)}` अनूठे दिन दर्ज)")

# Compute Core Analytics via engine.py backend
pi_B, pi_T, I, P = engine.score(mag)
events = filter_gade_events(P, threshold, window=1200)

# Real-Time Metrics Grid Display
c1, c2, c3, c4 = st.columns(4)
c1.metric("Max P", f"{np.max(P):.3f}")
c2.metric("Mean P", f"{np.mean(P):.3f}")
c3.metric("Detected Core Events", str(len(events)))
c4.metric("Applied Threshold", f"{threshold:.2f}")

# Plot 1: Instability Probability Curve Track
st.subheader("📈 Instability Probability Track")
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(JD, P, color="purple", linewidth=1.2, label="P(snap)")
ax.axhline(threshold, color="red", linestyle="--", label="Threshold Boundary")
if events:
    ax.scatter(JD[events], P[events], color="orange", s=98, label="True Dynamic Peak", zorder=5)
ax.set_xlabel("Julian Date (JD)")
ax.set_ylabel("Probability Matrix")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)

# Plot 2: Processed Light Curve Astronomical Layout
st.subheader("🌟 Processed Light Curve")
fig2, ax2 = plt.subplots(figsize=(12, 3.5))
ax2.plot(JD, mag, color="black", linewidth=1.0, label="Observed Magnitude")
ax2.invert_yaxis()  # Standard astronomical scale convention
ax2.set_xlabel("Julian Date (JD)")
ax2.set_ylabel("Magnitude (m)")
ax2.grid(True, alpha=0.3)
st.pyplot(fig2)

# Plot 3: Dimensionless Boundary Index Vector Breakdown
st.subheader("🧠 Dimensionless Boundary Index Terms Evaluation")
fig3, ax3 = plt.subplots(figsize=(12, 3.5))
ax3.plot(JD, pi_B, label=r"Driving Term ($\Pi_B$)", color="steelblue")
ax3.plot(JD, pi_T, label=r"Resistance Term ($\Pi_T$)", color="darkgreen")
ax3.plot(JD, I, label=r"Gade Index ($\Pi_I$)", color="crimson")
ax3.set_xlabel("Julian Date (JD)")
ax3.set_ylabel("Dimensionless Scalar Value")
ax3.grid(True, alpha=0.3)
ax3.legend()
st.pyplot(fig3)

# Core Real-Time System Alerts Banner
st.subheader("🚨 Real-Time Core System Alerts")
if events:
    st.error(f"🚨 ALERT: SNAP-like burst detected at {len(events)} core critical points.")
    st.write("Indexed core dataframe timestamps (Data Row Offsets):", events)
else:
    st.success("✅ SYSTEM STABLE: No anomalous SNAP-like core events detected within threshold bounds.")

with st.expander("Expand Normalized Dataset Ledger"):
    st.dataframe(pd.DataFrame({"Julian Date (JD)": JD, "Normalized Magnitude": mag}))

st.markdown("---")
st.subheader("⚙️ Gade Engine Integrity & Mathematical Verification")
if st.button("🧪 Execute Automated Mathematical Audit"):
    test_engine = SNAPBayesianEngine(alpha=alpha)
    
    # Audit 1: Perfect Static Balance Test Execution
    _, _, t1_I, _ = test_engine.score([10.0, 10.0, 10.0])
    test_1_passed = np.allclose(t1_I, 0.0, atol=1e-7)
    if test_1_passed:
        st.success("✅ **Test 1: Perfect Equilibrium (B = T) — PASSED**")
        st.write(f"• Measured Gade Index Residual: `{float(t1_I) if not isinstance(t1_I, np.ndarray) else float(t1_I[0]):.12f}`")
    else:
        st.error("❌ Test 1: Equilibrium Validation Failed")
        
    st.markdown(" ")
    
    # Audit 2: Dynamic Trigger Test Execution
    _, _, t2_I, t2_P = test_engine.score([6.0, 5.0, 7.0])
    test_2_passed = np.max(t2_I) > 0.5 and np.max(t2_P) > 0.85
    if test_2_passed:
        st.success("✅ **Test 2: SNAP Peak-to-Decline Trigger — PASSED**")
        st.write(f"• Measured Outburst Peak Probability: `{np.max(t2_P)*100:.2f}%`")
    else:
        st.error("❌ Test 2: Dynamic Outburst Verification Failed")
        
    if test_1_passed and test_2_passed:
        st.balloons()
        st.toast("Gade Core Mathematical Audit Successful!", icon="🔬")
