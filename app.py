import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import io

# ==============================================================================
# GADE CORE ENGINE WITH REAL NOISE FILTERING
# ==============================================================================
class SNAPBayesianEngine:
    def __init__(self, alpha=0.3, eps=1e-12):
        self.alpha = alpha
        self.eps = eps
        self.c, self.G, self.hbar = 299792458.0, 6.67430e-11, 1.0545718e-34
        self.l_p_sq = (self.hbar * self.G) / (self.c**3)

    def flux(self, mag):
        return 10 ** (-0.4 * np.asarray(mag, dtype=float))

    def normalize(self, x):
        return np.asarray(x, dtype=float) / (np.mean(x) + self.eps)

    def compute_state(self, mag):
        F = self.flux(mag)
        # वास्तविक AAVSO डेटा के शोर (Noise) को साफ करने के लिए रोलिंग फ़िल्टर
        if len(F) > 10:
            F = pd.Series(F).rolling(window=11, center=True, min_periods=1).mean().to_numpy()
        S = self.normalize(F)
        dS = np.gradient(S) if len(S) >= 2 else np.zeros_like(S)
        return S, 1.0 + self.alpha * dS

    def index(self, pi_B, pi_T):
        return np.asarray(pi_B, dtype=float) - np.asarray(pi_T, dtype=float)

    def probability(self, I):
        I = np.asarray(I, dtype=float)
        if len(I) < 2: return np.full_like(I, 0.5)
        dI = np.gradient(I)
        vol = np.array([np.std(I[max(0, i - 15):i + 1]) for i in range(len(I))])
        z = 2.0 * I + 1.2 * dI + 0.4 * vol # पीक को और तीखा करने के लिए वेटेज बढ़ाया
        return 1.0 / (1.0 + np.exp(-z))

    def score(self, mag):
        pi_B, pi_T = self.compute_state(mag)
        I = self.index(pi_B, pi_T)
        return pi_B, pi_T, I, self.probability(I)

def filter_gade_events(P, threshold, window=50):
    P = np.asarray(P, dtype=float)
    raw_indices = np.where(P > threshold)[0]
    filtered = []
    for idx in raw_indices:
        start, end = max(0, idx - window), min(len(P), idx + window + 1)
        if P[idx] == np.max(P[start:end]) and int(idx) not in filtered:
            filtered.append(int(idx))
    return filtered

# ==============================================================================
# STREAMLIT UI ARCHITECTURE
# ==============================================================================
st.set_page_config(page_title="SNAP Multi-Star Dashboard", layout="wide")
st.title("🔭 SNAP Astro Dashboard (Multi-Star Version)")
st.caption("Advanced Multi-Star Parallel Preprocessing -> SNAP Engine -> Automated Core Selection")

engine = SNAPBayesianEngine()

def read_zip_multi_files(uploaded_file):
    star_datasets = {}
    try:
        uploaded_file.seek(0)
        with zipfile.ZipFile(uploaded_file) as z:
            file_list = [f for f in z.namelist() if f.lower().endswith((".csv", ".txt"))]
            for file_path in file_list:
                star_name = file_path.split("/")[-1].replace(".csv", "").replace(".txt", "").strip()
                with z.open(file_path) as f:
                    content = f.read()
                    try: df = pd.read_csv(io.BytesIO(content))
                    except: df = pd.read_csv(io.BytesIO(content), delim_whitespace=True)
                    if not df.empty: star_datasets[star_name] = df
        return star_datasets
    except Exception as e:
        st.error(f"ZIP parse failed: {e}")
        return None

def load_aavso(df):
    if df is None or df.empty: raise ValueError("Empty matrix.")
    df.columns = [str(c).strip().lower() for c in df.columns]
    jd_col = next((c for c in df.columns if "jd" in c or "julian" in c), None)
    mag_col = next((c for c in df.columns if "mag" in c or "magnitude" in c or "flux" in c), None)
    if not jd_col or not mag_col: raise ValueError("Columns mismatch.")
    out = df[[jd_col, mag_col]].copy()
    out.columns = ["JD", "mag"]
    out["JD"] = pd.to_numeric(out["JD"], errors="coerce")
    out["mag"] = pd.to_numeric(out["mag"], errors="coerce")
    # ओवरलैपिंग (एक ही समय के कई डेटा पॉइंट्स) को साफ करना
    out = out.dropna().sort_values("JD").groupby("JD", as_index=False).mean()
    return out["JD"].to_numpy(), out["mag"].to_numpy()

st.sidebar.header("Controls")
threshold = st.sidebar.slider("SNAP threshold", 0.0, 1.0, 0.98, 0.01) # डिफ़ॉल्ट 0.98 सेट किया शोर रोकने के लिए
alpha = st.sidebar.slider("Engine alpha", 0.0, 2.0, 0.3, 0.01)
engine.alpha = alpha

run_sim = st.sidebar.checkbox("Load Historical T CrB Dataset", value=False)
all_stars_data = {}

if run_sim:
    np.random.seed(42)
    sim_jd = np.linspace(2431000, 2431200, 150)
    sim_mag = 10.0 - 4.5 * np.exp(-((sim_jd - 2431185) / 12)**2) + np.random.normal(0, 0.1, 150)
    all_stars_data["T_CrB_1946_Simulated"] = pd.DataFrame({"jd": sim_jd, "mag": sim_mag})
    st.sidebar.success("Default T CrB Dataset Loaded.")
else:
    uploaded_files = st.file_uploader("Upload AAVSO Files (Multiple CSVs or ZIP)", type=["csv", "txt", "zip"], accept_multiple_files=True)
    if uploaded_files:
        for u_file in uploaded_files:
            if u_file.name.lower().endswith(".zip"):
                zip_data = read_zip_multi_files(u_file)
                if zip_data: all_stars_data.update(zip_data)
            else:
                df_raw = pd.read_csv(u_file) if u_file.name.lower().endswith(".csv") else pd.read_csv(u_file, delim_whitespace=True)
                all_stars_data[u_file.name.replace(".csv","").replace(".txt","").strip()] = df_raw

if not all_stars_data:
    st.info("System Idle. Upload files or select Simulation Mode.")
    st.stop()

selected_star = st.selectbox("Available Stars in System:", list(all_stars_data.keys()))
try:
    JD, mag = load_aavso(all_stars_data[selected_star])
    cleaned_df = pd.DataFrame({"JD": JD, "mag": mag})
except Exception as e:
    st.error(f"Error processing: {e}")
    st.stop()

pi_B, pi_T, I, P = engine.score(mag)
events = filter_gade_events(P, threshold, window=50) # विंडो बढ़ाकर 50 की ताकि केवल क्लस्टर के मुख्य पीक दिखें

c1, c2, c3, c4 = st.columns(4)
c2.metric("Mean P", f"{np.mean(P):.3f}")
c1.metric("Max P", f"{np.max(P):.3f}")
c3.metric("Detected Events", str(len(events)))
c4.metric("Threshold", f"{threshold:.2f}")

st.subheader(f"📈 Instability Probability — {selected_star}")
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(JD, P, color="purple", linewidth=1.2, label="P(snap)")
ax.axhline(threshold, color="red", linestyle="--", label="Threshold")
if events: ax.scatter(JD[events], P[events], color="orange", s=70, label="True Dynamic Peak", zorder=5)
ax.set_xlabel("JD"); ax.set_ylabel("Probability"); ax.grid(True, alpha=0.3); ax.legend()
st.pyplot(fig)

st.subheader(f"🌟 Light Curve — {selected_star}")
fig2, ax2 = plt.subplots(figsize=(12, 3.5))
ax2.plot(JD, mag, color="black", linewidth=1.0, label="Magnitude")
ax2.invert_yaxis()
ax2.set_xlabel("JD"); ax2.set_ylabel("Magnitude"); ax2.grid(True, alpha=0.3); ax2.legend()
st.pyplot(fig2)

st.subheader(f"🧠 Engine Terms — {selected_star}")
fig3, ax3 = plt.subplots(figsize=(12, 3.5))
ax3.plot(JD, pi_B, label=r"$\Pi_B$", color="steelblue")
ax3.plot(JD, pi_T, label=r"$\Pi_T$", color="darkgreen")
ax3.plot(JD, I, label=r"$\Pi_I$", color="crimson")
ax3.set_xlabel("JD"); ax3.grid(True, alpha=0.3); ax3.legend()
st.pyplot(fig3)

st.subheader("🚨 Alerts")
if events:
    st.error(f"SNAP-like alerts detected at {len(events)} core critical points.")
    st.write(events)
else:
    st.success("No SNAP-like event detected.")

with st.expander("Show cleaned data"): st.dataframe(cleaned_df)

# ==============================================================================
# INTEGRITY AUDIT TEST MODULE
# ==============================================================================
st.markdown("---")
st.subheader("⚙️ Gade Engine Integrity & Math Validation")

if st.button("🧪 Run Automated Mathematical Audit"):
    with st.spinner("Processing Audit..."):
        test_engine = SNAPBayesianEngine(alpha=alpha)
        t1_pi_B, t1_pi_T, t1_I, t1_P = test_engine.score([10.0, 10.0, 10.0])
        test_1_passed = np.allclose(t1_I, 0.0, atol=1e-7)
        if test_1_passed:
            st.success("✅ **Test 1: Perfect Equilibrium (B = T) — PASSED**")
            st.write(f"• Measured Gade Index: `{float(t1_I[0]):.12f}`") # एरे इंडेक्सिंग एरर फिक्स यहाँ है
        else: st.error("❌ Test 1 Failed")

        st.markdown(" ")
        t2_pi_B, t2_pi_T, t2_I, t2_P = test_engine.score([6.0, 5.0, 7.0])
        test_2_passed = np.max(t2_I) > 0.5 and np.max(t2_P) > 0.85
        if test_2_passed:
            st.success("✅ **Test 2: SNAP Peak-to-Decline Trigger — PASSED**")
            st.write(f"• Measured Probability: `{np.max(t2_P)*100:.2f}%`")
        else: st.error("❌ Test 2 Failed")
        
        if test_1_passed and test_2_passed:
            st.balloons(); st.toast("Gade Engine Audit Successful!", icon="🔬")
