import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="CCPP Hybrid Intelligent System",
    page_icon="⚡",
    layout="wide"
)

# 1. Load Artefak Model
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler_X.pkl')
    config = joblib.load('model_config.pkl')
    model = load_model('model_D_pe.keras')
    return scaler, config, model

scaler_X, model_config, model_D = load_artifacts()

# 2. Setup Fuzzy Inference System (Mamdani)
def build_fuzzy_system():
    pe_input = ctrl.Antecedent(np.arange(420, 497, 0.5), 'pe_prediction')
    operating_status = ctrl.Consequent(np.arange(0, 101, 1), 'operating_status')

    pe_input['Low'] = fuzz.trapmf(pe_input.universe, [420, 420, 432, 442])
    pe_input['Normal'] = fuzz.trimf(pe_input.universe, [438, 450, 460])
    pe_input['Good'] = fuzz.trimf(pe_input.universe, [455, 468, 478])
    pe_input['Optimal'] = fuzz.trapmf(pe_input.universe, [472, 482, 496, 496])

    operating_status['Low'] = fuzz.trimf(operating_status.universe, [0, 0, 35])
    operating_status['Normal'] = fuzz.trimf(operating_status.universe, [25, 50, 70])
    operating_status['Good'] = fuzz.trimf(operating_status.universe, [60, 75, 88])
    operating_status['Optimal'] = fuzz.trimf(operating_status.universe, [80, 100, 100])

    r1 = ctrl.Rule(pe_input['Low'], operating_status['Low'])
    r2 = ctrl.Rule(pe_input['Normal'], operating_status['Normal'])
    r3 = ctrl.Rule(pe_input['Good'], operating_status['Good'])
    r4 = ctrl.Rule(pe_input['Optimal'], operating_status['Optimal'])

    system = ctrl.ControlSystem([r1, r2, r3, r4])
    return ctrl.ControlSystemSimulation(system)

fuzzy_sim = build_fuzzy_system()

def get_fuzzy_decision(pe_val):
    fuzzy_sim.input['pe_prediction'] = float(np.clip(pe_val, 420, 496))
    fuzzy_sim.compute()
    score = fuzzy_sim.output['operating_status']
    if score < 35:
        category = "Low"
    elif score < 65:
        category = "Normal"
    elif score < 82:
        category = "Good"
    else:
        category = "Optimal"
    return score, category

# Sidebar Menu (8 Area Wajib Sesuai Panduan Tubes)
st.sidebar.title("⚡ Navigasi Dashboard")
menu = st.sidebar.radio(
    "Pilih Menu:",
    [
        "1. Overview",
        "2. Dataset Explorer",
        "3. GA Optimization",
        "4. PSO Optimization",
        "5. Neural Network",
        "6. Fuzzy Inference",
        "7. Prediction & Simulation",
        "8. Model Comparison"
    ]
)

# Konten Menu
if menu == "1. Overview":
    st.title("⚡ Combined Cycle Power Plant Output Prediction")
    st.subheader("Hybrid Intelligent System: GA - PSO - MLP - Fuzzy Logic")
    st.markdown("""
    * **Mata Kuliah**: Kecerdasan Buatan (TE2514031) - Kelas A
    * **Kelompok A3**: Rizky Jems G. S., Muhammad Mulia Haziq, I Made Ari Anggara, M. Raihan
    * **Dataset**: Combined Cycle Power Plant (UCI ID: 294)
    * **Pipeline Alur**:
      `Dataset UCI` ➔ `Preprocessing (MinMax)` ➔ `GA (Feature Selection)` ➔ `PSO (Hyperparameter Tuning)` ➔ `MLP Regression` ➔ `Fuzzy Decision Support` ➔ `Streamlit`
    """)

elif menu == "2. Dataset Explorer":
    st.title("📊 Dataset Explorer")
    df = pd.read_excel('Folds5x2_pp.xlsx', sheet_name=0)
    st.write("Cuplikan Data (5 baris pertama):", df.head())
    col1, col2 = st.columns(2)
    with col1:
        st.write("Statistik Deskriptif:", df.describe())
    with col2:
        st.write("Missing Values:", df.isnull().sum())

elif menu == "3. GA Optimization":
    st.title("🧬 Genetic Algorithm (Feature Selection)")
    st.write(f"**Kromosom Terbaik**: `{model_config['best_chromosome']}`")
    st.write(f"**Fitur Terpilih**: `{model_config['selected_feature_names']}`")
    st.info("Fitur 'AP' (Ambient Pressure) dieliminasi oleh GA karena memberikan kontribusi paling redundan terhadap penurunan error.")

elif menu == "4. PSO Optimization":
    st.title("🐝 Particle Swarm Optimization (Hyperparameters)")
    st.write(f"**Hidden Units Layer 1**: `{model_config['pso_best_units1']}`")
    st.write(f"**Hidden Units Layer 2**: `{model_config['pso_best_units2']}`")
    st.write(f"**Learning Rate**: `{model_config['pso_best_lr']:.6f}`")

elif menu == "5. Neural Network":
    st.title("🧠 Neural Network (MLP Regression)")
    st.markdown("""
    * **Arsitektur**: Input Layer (3 unit hasil GA) ➔ Hidden 1 (73 ReLU) ➔ Hidden 2 (33 ReLU) ➔ Output (1 Linear)
    * **Optimizer**: Adam dengan learning rate optimal hasil PSO
    * **Loss Function**: Mean Squared Error (MSE)
    """)

elif menu == "6. Fuzzy Inference":
    st.title("🌫️ Fuzzy Logic Decision-Support Layer")
    st.markdown("""
    * **Tipe FIS**: Mamdani Centroid Defuzzification
    * **Input (Antecedent)**: Prediksi Daya Bersih (PE) [420 - 496 MW]
    * **Output (Consequent)**: Operating Status Index [0 - 100]
    * **Status Kategori**: Low, Normal, Good, Optimal
    """)

elif menu == "7. Prediction & Simulation":
    st.title("🔮 Prediksi & Simulasi Real-Time")
    st.write("Masukkan parameter sensor operasional pembangkit:")
    c1, c2 = st.columns(2)
    with c1:
        at_val = st.number_input("Ambient Temperature (AT) [°C]", value=15.0, min_value=1.0, max_value=40.0)
        v_val = st.number_input("Exhaust Vacuum (V) [cm Hg]", value=40.0, min_value=25.0, max_value=85.0)
    with c2:
        ap_val = st.number_input("Ambient Pressure (AP) [hPa]", value=1013.0, min_value=990.0, max_value=1035.0)
        rh_val = st.number_input("Relative Humidity (RH) [%]", value=75.0, min_value=25.0, max_value=100.0)

    if st.button("Jalankan Prediksi & Evaluasi Fuzzy"):
        # Scale semua 4 fitur
        raw_input = np.array([[at_val, v_val, ap_val, rh_val]])
        scaled_input = scaler_X.transform(raw_input)
        # Ambil subset fitur hasil GA: index [0, 1, 3] -> AT, V, RH
        ga_input = scaled_input[:, model_config['selected_features_idx']]
        
        # Prediksi via Model D
        pred_pe = float(model_D.predict(ga_input, verbose=0)[0][0])
        score, decision = get_fuzzy_decision(pred_pe)

        st.success(f"**Prediksi Net Electrical Power Output (PE)**: {pred_pe:.2f} MW")
        st.metric(label="Status Operasi Pembangkit (Fuzzy)", value=decision, delta=f"Score: {score:.1f}/100")

elif menu == "8. Model Comparison":
    st.title("📈 Model Comparison (Ablation Study)")
    comp_df = pd.DataFrame(model_config['metrics_comparison']).T
    st.table(comp_df)
