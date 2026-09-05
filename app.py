# ==============================================================================
# CMBT LIVE OBSERVATORY CYCLONE AI (VERSION 1.1.0)
# Multi-Column Plotting + Threshold Slider Controls
# Lead Inventor: Chandrakant Shivram Gade, Nashik, Maharashtra, India.
# Official Identity Verification Lock: DOI: 10.5281/zenodo.22160704
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- FRONTEND PAGE CONFIGURATION ---
st.set_page_config(page_title="Live Cyclone AI Observatory", layout="wide")
st.title("🌪️ LIVE OBSERVATORY CYCLONE AI v1.1")
st.caption("Sovereign CMBT Vortex Mechanics -> Hyperbolic Tangent Bound Analytics Engine")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Analysis Controls")
high_threshold = st.sidebar.slider("Extreme Danger Threshold", 0.0, 1.0, 0.85, 0.01)
instability_threshold = st.sidebar.slider("Instability Threshold", 0.0, 1.0, 0.60, 0.01)
show_raw_signal = st.sidebar.checkbox("Show Raw Signal Plot", value=True)
show_phi_plot = st.sidebar.checkbox("Show Normalized Signal Plot", value=True)
show_dphi_plot = st.sidebar.checkbox("Show Gradient Plot", value=True)
show_vortex_plot = st.sidebar.checkbox("Show Vortex Index Plot", value=True)

# --- DIMENSIONLESS VORTEX CORE ENGINE ---
def normalize_signal(x):
    """
    Standard score normalization framework with dynamic protection
    against zero-standard-deviation crashes.
    """
    x = np.array(x, dtype=float)
    std_dev = np.std(x)
    return (x - np.mean(x)) / (std_dev if std_dev > 0 else 1e-9)

def compute_vortex_core(x):
    """
    Executes the universal boundary gradient and binds the systemic
    state variables using the rigid Hyperbolic Tangent (tanh) matrix.
    """
    phi_state = normalize_signal(x)
    velocity_gradient_dphi = np.gradient(phi_state)
    vortex_index_V = np.tanh(1.2 * phi_state + 0.8 * velocity_gradient_dphi)
    return phi_state, velocity_gradient_dphi, vortex_index_V

# --- CYCLONE METRIC CLASSIFIER ---
def classify_vortex_instability(v_index, high_thr=0.85, instab_thr=0.60):
    """
    Maps absolute velocity thresholds into standardized dynamic
    classification regimes with scalar boundary protection.
    """
    absolute_risk = np.abs(v_index)
    risk_labels = []

    for score in absolute_risk:
        if score > high_thr:
            risk_labels.append("🔥 HIGH CYCLONE / EXTREME VORTEX")
        elif score > instab_thr:
            risk_labels.append("⚠️ INSTABILITY ZONE")
        else:
            risk_labels.append("✅ STABLE")

    return absolute_risk, risk_labels

# --- APPLICATION INPUT HUB ---
uploaded_data_buffer = st.file_uploader(
    "Upload Time-Series Data (Wind / Barometric Pressure / Core Signal CSV)",
    type=["csv"]
)

if uploaded_data_buffer is None:
    st.info("📌 System Idle. Drop your live cyclone time-series or station logs to activate the AI observatory.")
    st.stop()

# --- MAIN PIPELINE EXECUTION CONDUIT ---
try:
    df_raw = pd.read_csv(uploaded_data_buffer)

    numeric_columns = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) == 0:
        st.error("❌ CRITICAL ERROR: No numeric target datasets discovered in the uploaded frame.")
        st.stop()

    st.sidebar.subheader("📊 Data Selection")
    selected_columns = st.sidebar.multiselect(
        "Choose numeric columns for analysis",
        options=numeric_columns,
        default=[numeric_columns[0]]
    )

    if len(selected_columns) == 0:
        st.warning("Select at least one numeric column to continue.")
        st.stop()

    st.success(f"✅ Selected columns: {', '.join(selected_columns)}")

    # --- MULTI-COLUMN METRICS LAYOUT ---
    metric_cols = st.columns(min(len(selected_columns), 3))
    for i, col_name in enumerate(selected_columns[:3]):
        series = df_raw[col_name].dropna().to_numpy()
        if len(series) >= 1:
            metric_cols[i].metric(f"Mean: {col_name}", f"{np.mean(series):.3f}")
        else:
            metric_cols[i].metric(f"Mean: {col_name}", "N/A")

    # --- MULTI-COLUMN PLOTTING SECTION ---
    st.markdown("---")
    plot_cols = st.columns(2)

    with plot_cols[0]:
        if show_raw_signal:
            st.subheader("📈 Raw Signal Tracks")
            fig_raw, ax_raw = plt.subplots(figsize=(12, 4))
            for col_name in selected_columns:
                series = df_raw[col_name].dropna().to_numpy()
                ax_raw.plot(series, linewidth=1.0, label=col_name)
            ax_raw.set_xlabel("Data Row Offsets")
            ax_raw.set_ylabel("Raw Value")
            ax_raw.grid(True, alpha=0.2)
            ax_raw.legend()
            st.pyplot(fig_raw)

    with plot_cols[1]:
        if show_phi_plot:
            st.subheader("🌐 Normalized Signal Tracks")
            fig_phi, ax_phi = plt.subplots(figsize=(12, 4))
            for col_name in selected_columns:
                series = df_raw[col_name].dropna().to_numpy()
                if len(series) >= 5:
                    phi = normalize_signal(series)
                    ax_phi.plot(phi, linewidth=1.0, label=f"{col_name} (phi)")
            ax_phi.set_xlabel("Data Row Offsets")
            ax_phi.set_ylabel("Normalized Value")
            ax_phi.grid(True, alpha=0.2)
            ax_phi.legend()
            st.pyplot(fig_phi)

    st.markdown("---")
    plot_cols2 = st.columns(2)

    with plot_cols2[0]:
        if show_dphi_plot and len(selected_columns) > 0:
            st.subheader("📉 Gradient Tracks")
            fig_dphi, ax_dphi = plt.subplots(figsize=(12, 4))
            for col_name in selected_columns:
                series = df_raw[col_name].dropna().to_numpy()
                if len(series) >= 5:
                    phi = normalize_signal(series)
                    dphi = np.gradient(phi)
                    ax_dphi.plot(dphi, linewidth=1.0, label=f"{col_name} (dphi)")
            ax_dphi.set_xlabel("Data Row Offsets")
            ax_dphi.set_ylabel("Gradient Value")
            ax_dphi.grid(True, alpha=0.2)
            ax_dphi.legend()
            st.pyplot(fig_dphi)

    with plot_cols2[1]:
        if show_vortex_plot and len(selected_columns) > 0:
            st.subheader("🌪️ Vortex Index Tracks")
            fig_vortex, ax_vortex = plt.subplots(figsize=(12, 4))
            for col_name in selected_columns:
                series = df_raw[col_name].dropna().to_numpy()
                if len(series) >= 5:
                    phi, dphi, vortex_V = compute_vortex_core(series)
                    ax_vortex.plot(vortex_V, linewidth=1.2, label=f"{col_name} (V)")
            ax_vortex.axhline(high_threshold, linestyle="--", color="red", alpha=0.7, label=f"Extreme Threshold ({high_threshold:.2f})")
            ax_vortex.axhline(instability_threshold, linestyle=":", color="orange", alpha=0.7, label=f"Instability Threshold ({instability_threshold:.2f})")
            ax_vortex.set_xlabel("Data Row Offsets")
            ax_vortex.set_ylabel("Dimensionless Boundary Index")
            ax_vortex.grid(True, alpha=0.3)
            ax_vortex.legend()
            st.pyplot(fig_vortex)

    # --- ANALYSIS OF FIRST SELECTED COLUMN FOR ALERT LEDGER ---
    st.markdown("---")
    st.subheader("🚨 Automated Real-Time Anomalies Ledger")

    target_column_name = selected_columns[0]
    raw_signal_vector = df_raw[target_column_name].dropna().to_numpy()

    if len(raw_signal_vector) < 5:
        st.error("❌ CRITICAL ERROR: Insufficient array size for computing system gradients.")
        st.stop()

    phi, dphi, vortex_V = compute_vortex_core(raw_signal_vector)
    risk_scores, state_labels = classify_vortex_instability(
        vortex_V,
        high_thr=high_threshold,
        instab_thr=instability_threshold
    )

    maximum_risk_score = np.max(risk_scores)
    if maximum_risk_score > high_threshold:
        global_alert_banner = "🚨 CRITICAL VORTEX ALERT"
    elif maximum_risk_score > instability_threshold:
        global_alert_banner = "⚠️ MODERATE INSTABILITY ALERT"
    else:
        global_alert_banner = "✅ SYSTEM REGIME STABLE"

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Max Vortex Risk Index", f"{maximum_risk_score:.3f}")
    metric_col2.metric("Global Alert Status", global_alert_banner)

    alert_indices = np.where(risk_scores > instability_threshold)[0]

    if len(alert_indices) > 0:
        if maximum_risk_score > high_threshold:
            st.error(f"🚨 CRITICAL: Vortex instability triggers breached across `{len(alert_indices)}` sequence offsets.")
        else:
            st.warning(f"⚠️ NOTICE: Moderate fluid turbulence mapped across `{len(alert_indices)}` sequence offsets.")

        alert_records = {
            "Sequence Offset Index": alert_indices,
            "Calculated Risk Score": np.round(risk_scores[alert_indices], 4),
            "CMBT System Status Label": [state_labels[idx] for idx in alert_indices]
        }

        df_alerts_compressed = pd.DataFrame(alert_records)
        st.dataframe(df_alerts_compressed, use_container_width=True, hide_index=True)
    else:
        st.success("✅ SYSTEM STABLE: Vortex analytics grid records zero multi-domain anomalies.")

except Exception as e:
    st.error(f"❌ COMPILATION CRASH SHIELD ACTIVE: Processing script blocked an array shape disruption. Details: {str(e)}")