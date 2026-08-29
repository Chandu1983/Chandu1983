import streamlit as st, pandas as pd, numpy as np, matplotlib.pyplot as plt, zipfile, io

class SNAPBayesianEngine:
    def __init__(self, alpha=0.3):
        self.alpha, self.eps = alpha, 1e-12
    def flux(self, m): return 10 ** (-0.4 * np.asarray(m, dtype=float))
    def normalize(self, x): return np.asarray(x, dtype=float) / (np.mean(x) + self.eps)
    def compute_state(self, m):
        F = self.flux(m)
        if len(F) > 50: F = pd.Series(F).rolling(window=101, center=True, min_periods=1).mean().to_numpy()
        S = self.normalize(F)
        return S, 1.0 + self.alpha * (np.gradient(S) if len(S) >= 2 else np.zeros_like(S))
    def score(self, m):
        B, T = self.compute_state(m)
        I = B - T
        if len(I) < 2: return B, T, I, np.full_like(I, 0.5)
        z = 2.5 * I + 1.5 * np.gradient(I) + 0.2 * np.array([np.std(I[max(0, i-50):i+1]) for i in range(len(I))])
        return B, T, I, 1.0 / (1.0 + np.exp(-z))

def filter_gade_events(P, threshold, window=1200): # हाइपर मैक्रो विंडो 1200 सेट
    P, raw, filtered = np.asarray(P, dtype=float), np.where(P > threshold), []
    for idx in raw[0]:
        if P[idx] == np.max(P[max(0, idx-window):min(len(P), idx+window+1)]) and int(idx) not in filtered:
            filtered.append(int(idx))
    return filtered

st.set_page_config(page_title="SNAP Multi-Star Dashboard", layout="wide")
st.title("🔭 SNAP Astro Dashboard (Multi-Star Version)")
st.caption("Advanced Preprocessing -> SNAP Engine -> Automated Core Selection")
engine = SNAPBayesianEngine()

def read_zip_multi_files(uploaded_file):
    star_datasets = {}
    with zipfile.ZipFile(uploaded_file) as z:
        for f_path in [f for f in z.namelist() if f.lower().endswith((".csv", ".txt"))]:
            s_name = f_path.split("/")[-1].replace(".csv","").replace(".txt","").strip()
            with z.open(f_path) as f:
                try: df = pd.read_csv(io.BytesIO(f.read()))
                except: df = pd.read_csv(io.BytesIO(f.read()), delim_whitespace=True)
                if not df.empty: star_datasets[s_name] = df
    return star_datasets

def load_aavso(df):
    df.columns = [str(c).strip().lower() for c in df.columns]
    jd_c = next((c for c in df.columns if "jd" in c or "julian" in c), None)
    mag_c = next((c for c in df.columns if "mag" in c or "magnitude" in c or "flux" in c), None)
    if not jd_c or not mag_c: raise ValueError("Columns mismatch.")
    det_star = str(df[next((c for c in df.columns if "star" in c or "name" in c), df.columns[0])].iloc[0]).strip()
    out = df[[jd_c, mag_c]].copy()
    out.columns = ["JD", "mag"]
    out = out.dropna().astype(float).sort_values("JD").groupby("JD", as_index=False).mean()
    return out["JD"].to_numpy(), out["mag"].to_numpy(), det_star

threshold = st.sidebar.slider("SNAP threshold", 0.0, 1.0, 0.98, 0.01)
alpha = st.sidebar.slider("Engine alpha", 0.0, 2.0, 0.3, 0.01)
engine.alpha = alpha
all_stars_data = {}

if st.sidebar.checkbox("Load Historical T CrB Dataset", value=False):
    np.random.seed(42)
    s_jd = np.linspace(2431000, 2431200, 150)
    s_mag = 10.0 - 4.5 * np.exp(-((s_jd - 2431185) / 12)**2) + np.random.normal(0, 0.1, 150)
    all_stars_data["T_CrB_1946_Simulated"] = pd.DataFrame({"jd": s_jd, "mag": s_mag, "star": "T CrB Simulated"})
else:
    u_files = st.file_uploader("Upload AAVSO Files (Multiple CSVs or ZIP)", type=["csv", "txt", "zip"], accept_multiple_files=True)
    if u_files:
        for uf in u_files:
            if uf.name.lower().endswith(".zip"):
                zd = read_zip_multi_files(uf)
                if zd: all_stars_data.update(zd)
            else:
                try: df_r = pd.read_csv(uf)
                except: df_r = pd.read_csv(uf, delim_whitespace=True)
                all_stars_data[uf.name.replace(".csv","").replace(".txt","").strip()] = df_r

if not all_stars_data:
    st.info("System Idle. Upload files or select Simulation Mode.")
    st.stop()

sel_star = st.selectbox("Available Stars:", list(all_stars_data.keys()))
JD, mag, det_star_internal = load_aavso(all_stars_data[sel_star])

# तारे का नाम, शुरुआती और अंतिम तारीख दिखाने वाला ऑटोमैटिक बैनर
st.info(f"✨ **Gade Automated File Metadata Discovery Engine**\n\n• 🌟 **तारे का नाम (Identified Target):** `{det_star_internal if det_star_internal != 'nan' else sel_star}`\n\n• 📅 **डेटा अवधि (Observations Epoch):** From JD `{JD[0]:.4f}` to `{JD[-1]:.4f}` (कुल `{len(JD)}` अनूठे दिन दर्ज)")

pi_B, pi_T, I, P = engine.score(mag)
events = filter_gade_events(P, threshold, window=1200) # 1200 की विंडो

c1, c2, c3, c4 = st.columns(4)
c1.metric("Max P", f"{np.max(P):.3f}"); c2.metric("Mean P", f"{np.mean(P):.3f}"); c3.metric("Detected Events", str(len(events))); c4.metric("Threshold", f"{threshold:.2f}")

st.subheader("📈 Instability Probability")
fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(JD, P, color="purple", linewidth=1.2, label="P(snap)")
ax.axhline(threshold, color="red", linestyle="--")
if events: ax.scatter(JD[events], P[events], color="orange", s=80, label="True Dynamic Peak", zorder=5)
ax.set_xlabel("JD"); ax.set_ylabel("Probability"); ax.grid(True, alpha=0.3); ax.legend(); st.pyplot(fig)

st.subheader("🌟 Light Curve")
fig2, ax2 = plt.subplots(figsize=(12, 3.5))
ax2.plot(JD, mag, color="black", linewidth=1.0)
ax2.invert_yaxis(); ax2.set_xlabel("JD"); ax2.set_ylabel("Magnitude"); ax2.grid(True, alpha=0.3); st.pyplot(fig2)

st.subheader("🧠 Engine Terms")
fig3, ax3 = plt.subplots(figsize=(12, 3.5))
ax3.plot(JD, pi_B, label=r"$\Pi_B$", color="steelblue"); ax3.plot(JD, pi_T, label=r"$\Pi_T$", color="darkgreen"); ax3.plot(JD, I, label=r"$\Pi_I$", color="crimson")
ax3.set_xlabel("JD"); ax3.grid(True, alpha=0.3); ax3.legend(); st.pyplot(fig3)

st.subheader("🚨 Alerts")
if events: st.error(f"SNAP-like alerts detected at {len(events)} core critical points."); st.write(events)
else: st.success("No SNAP-like event detected.")

with st.expander("Show cleaned data"): st.dataframe(pd.DataFrame({"JD": JD, "mag": mag}))

st.markdown("---")
st.subheader("⚙️ Gade Engine Integrity & Math Validation")
if st.button("🧪 Run Automated Mathematical Audit"):
    test_engine = SNAPBayesianEngine(alpha=alpha)
    _, _, t1_I, _ = test_engine.score([10.0, 10.0, 10.0])
    test_1_passed = np.allclose(t1_I, 0.0, atol=1e-7)
    if test_1_passed:
        st.success("✅ **Test 1: Perfect Equilibrium (B = T) — PASSED**")
        st.write(f"• Measured Gade Index: `{float(t1_I[0]) if isinstance(t1_I, np.ndarray) else float(t1_I):.12f}`")
    else: st.error("❌ Test 1 Failed")
    st.markdown(" ")
    _, _, t2_I, t2_P = test_engine.score([6.0, 5.0, 7.0])
    test_2_passed = np.max(t2_I) > 0.5 and np.max(t2_P) > 0.85
    if test_2_passed: st.success("✅ **Test 2: SNAP Peak-to-Decline Trigger — PASSED**\n• Measured Probability: " + f"`{np.max(t2_P)*100:.2f}%`")
    else: st.error("❌ Test 2 Failed")
    if test_1_passed and test_2_passed: st.balloons(); st.toast("Audit Successful!", icon="🔬")
