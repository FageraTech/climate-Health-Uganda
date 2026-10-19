import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime

# 1. Enforce Premium Screen Layout Settings
st.set_page_config(
    page_title="Uganda Climate-Health Risk Dashboard", 
    page_icon="🏥", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject Enterprise CSS Injection Styles
st.markdown("""
    <style>
        /* Base Container Background & Typography */
        .stApp {
            background-color: #F8F9FA;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Premium Header Banner */
        .main-header {
            background: linear-gradient(135deg, #0A4D45 0%, #147A6E 100%);
            padding: 2.5rem;
            border-radius: 12px;
            color: #FFFFFF;
            margin-bottom: 2rem;
            box-shadow: 0 4px 15px rgba(13, 92, 83, 0.15);
        }
        .main-header h1 {
            color: #FFFFFF !important;
            font-weight: 700;
            margin-bottom: 0.5rem;
            font-size: 2.5rem;
        }
        .main-header p {
            color: #D1E7E4 !important;
            font-size: 1.1rem;
            margin: 0;
        }
        
        /* Structured Data Entry Cards */
        .data-card {
            background-color: #FFFFFF;
            padding: 2rem;
            border-radius: 10px;
            border-left: 5px solid #0D5C53;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            margin-bottom: 1.5rem;
        }
        .card-title {
            color: #0D5C53;
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        /* Clean Up Sidebar Elements */
        .css-164789v {
            background-color: #E8F1F0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- CUSTOM CACHED ASSETS LOADING ---
@st.cache_resource
def load_production_pipeline():
    meta = joblib.load("models/pipeline_metadata.joblib")
    model = lgb.Booster(model_file="models/production_lgbm_model.txt")
    return meta, model

try:
    meta, model = load_production_pipeline()
    st.sidebar.markdown("### 🎛️ System Status")
    st.sidebar.success("HEALTH PREDICTION ENGINE:")
except Exception as e:
    st.error(f"🔴 Critical Initialization Error: Models missing from disk. Run serialization script. Detail: {e}")
    st.stop()

# --- BRANDED APPLICATION HEADER ---
st.markdown("""
    <div class="main-header">
        <h1>🏥 Health Prediction in Uganda & Climate Changes</h1>
        <p>Operational Epidemiological Risk Intelligence Engine Engine Framework — Powered by FageraTech</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("### 🔬 Patient & Environmental Intake Matrix")

# --- UI SPLIT LAYOUT FOR INTENDED PARAMETERS ---
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
        <div class="data-card">
            <div class="card-title">📋 Demographic Profile & Case Records</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Render input forms natively underneath the container headers
    age = st.number_input("Individual Age at Record Time", min_value=0, max_value=120, value=45)
    gender = st.selectbox("Recorded Gender", options=["Male", "Female", "Unknown"])
    zone = st.selectbox("Geographic Regional Zone", options=["Rural", "Peri-urban", "Urban"])
    location = st.text_input("Uganda District / Village / Area Name", value="Kampala")
    death_date = st.date_input("Date of Case Analysis / Diagnostic Entry", value=datetime.today())

with col2:
    st.markdown("""
        <div class="data-card">
            <div class="card-title">🌤️ Micro-Climate Threshold & Lag Matrix</div>
        </div>
    """, unsafe_allow_html=True)
    
    avg_temp = st.slider("Average Local Temperature on Record Day (°C)", -5.0, 50.0, 24.0)
    max_temp = st.slider("Maximum Local Temperature on Record Day (°C)", -5.0, 55.0, 29.0)
    min_temp = st.slider("Minimum Local Temperature on Record Day (°C)", -10.0, 40.0, 16.0)
    precipitation = st.number_input("Precipitation Accumulated Level (mm)", min_value=0.0, value=12.5)
    
    st.markdown("---")
    st.caption("**📊 Multi-Scale Satellite & Environmental Rolling Historical Metrics**")
    
    tmax_30d = st.number_input("Highest Maximum 2m Temp over Trailing 30 Days (°C)", value=27.6)
    tmin_30d = st.number_input("Lowest Minimum 2m Temp over Trailing 30 Days (°C)", value=16.1)
    temp_range_mean_30d = st.number_input("Mean Daily Temperature Range over Trailing 30 Days (°C)", value=7.9)
    
    tavg_7d = st.number_input("Mean Air Temperature over Trailing 7 Days (°C)", value=22.0)
    tavg_30d = st.number_input("Mean Air Temperature over Trailing 30 Days (°C)", value=21.4)
    tavg_90d = st.number_input("Mean Air Temperature over Trailing 90 Days (°C)", value=21.0)
    
    rain_sum_7d = st.number_input("Total Rainfall Accumulated over Trailing 7 Days (mm)", value=28.4)
    rain_sum_30d = st.number_input("Total Rainfall Accumulated over Trailing 30 Days (mm)", value=175.2)
    rain_sum_90d = st.number_input("Total Rainfall Accumulated over Trailing 90 Days (mm)", value=421.9)
    rain_days_30d = st.slider("Count of Significant Rain Days (>1mm) over Trailing 30 Days", 0, 30, 18)
    max_daily_rain_30d = st.number_input("Highest Single-Day Rainfall Event over Trailing 30 Days (mm)", value=20.4)
    hot_days_30d = st.slider("Count of Extreme Heatwave Days (>35°C) over Trailing 30 Days", 0, 30, 0)
    
    ndvi_30d = st.slider("Median Vegetation Greenness Index (NDVI) over Trailing 30 Days", -1.0, 1.0, 0.72)
    ndvi_90d = st.slider("Median Vegetation Greenness Index (NDVI) over Trailing 90 Days", -1.0, 1.0, 0.73)
    elevation = st.number_input("Topographic Ground Elevation Feature (m)", value=1114)
    slope = st.number_input("Topographic Terrain Slope Angle (Degrees)", value=0.16)

st.markdown("<br>", unsafe_allow_html=True)

# --- INFERENCE ENGINE ROUTINE ---
if st.button("🚀 Run Diagnostic Risk Inference Engine", type="primary", use_container_width=True):
    # 1. Structure form parameters safely back to matching dictionary keys
    raw_data = {
        "age": age, "avg_temperature": avg_temp, "max_temperature": max_temp, "min_temperature": min_temp,
        "precipitation": precipitation, "elevation": elevation, "slope": slope, "hot_days_30d": hot_days_30d,
        "max_daily_rain_30d": max_daily_rain_30d, "ndvi_30d": ndvi_30d, "ndvi_90d": ndvi_90d,
        "rain_days_30d": rain_days_30d, "rain_sum_30d": rain_sum_30d, "rain_sum_7d": rain_sum_7d,
        "rain_sum_90d": rain_sum_90d, "tavg_30d": tavg_30d, "tavg_7d": tavg_7d, "tavg_90d": tavg_90d,
        "tmax_30d": tmax_30d, "tmin_30d": tmin_30d, "temp_range_mean_30d": temp_range_mean_30d,
        "latitude": -1.2921, "longitude": 36.8219 # Coordinates baseline defaults
    }
    
    # 2. Extract Temporal Features
    dt = pd.to_datetime(death_date)
    raw_data["year"] = dt.year
    raw_data["month"] = dt.month
    raw_data["day"] = dt.day
    raw_data["dayofyear"] = dt.timetuple().tm_yday
    raw_data["weekofyear"] = dt.isocalendar().week
    raw_data["quarter"] = (dt.month - 1) // 3 + 1
    raw_data["month_sin"] = np.sin(2 * np.pi * dt.month / 12)
    raw_data["month_cos"] = np.cos(2 * np.pi * dt.month / 12)
    raw_data["doy_sin"] = np.sin(2 * np.pi * raw_data["dayofyear"] / 365)
    raw_data["doy_cos"] = np.cos(2 * np.pi * raw_data["dayofyear"] / 365)
    
    # 3. Dynamic Mathematical Interactions (Matches training feature maps precisely)
    raw_data["temp_range"] = max_temp - min_temp
    raw_data["tavg_diff_7d_30d"] = tavg_7d - tavg_30d
    raw_data["tavg_diff_30d_90d"] = tavg_30d - tavg_90d
    raw_data["tmax_anomaly"] = max_temp - tavg_30d
    raw_data["tmin_anomaly"] = min_temp - tavg_30d
    raw_data["rain_intensity"] = rain_sum_30d / (rain_days_30d + 1)
    raw_data["rain_7d_ratio"] = rain_sum_7d / (rain_sum_30d + 1)
    raw_data["rain_30d_ratio"] = rain_sum_30d / (rain_sum_90d + 1)
    raw_data["heavy_rain_flag"] = int(max_daily_rain_30d > 50)
    raw_data["ndvi_change"] = ndvi_30d - ndvi_90d
    raw_data["ndvi_ratio"] = ndvi_30d / (ndvi_90d + 1e-6)
    raw_data["hot_day_ratio"] = hot_days_30d / 30.0
    raw_data["is_infant"] = int(age <= 1)
    raw_data["is_child"] = int(1 < age <= 5)
    raw_data["is_elderly"] = int(age >= 65)
    raw_data["age_squared"] = age ** 2
    raw_data["age_x_temp"] = age * avg_temp
    
    # 4. Extract Pipeline Mappings and Encodings
    raw_data["location_freq"] = meta["loc_freq_map"].get(location, 0.0)
    raw_data["location_te"] = meta["loc_te_map"].get(location, meta["global_te_mean"])
    
    try:
        raw_data["gender_enc"] = meta["le_gender"].transform([gender])[0]
    except:
        raw_data["gender_enc"] = meta["le_gender"].transform(["Unknown"])[0]
        
    try:
        raw_data["zone_enc"] = meta["le_zone"].transform([zone])[0]
    except:
        raw_data["zone_enc"] = 0
        
    # 5. Build matrix in correct alignment indices
    input_df = pd.DataFrame([raw_data])
    inference_matrix = input_df[meta["features"]].values
    
    # 6. Apply simple imputation and extract predictions
    inference_matrix = meta["imputer"].transform(inference_matrix)
    risk_probability = model.predict(inference_matrix)[0]
    
    # 7. Apply the rigid 0.5 classification threshold boundary
    binary_label = 1 if risk_probability >= 0.5 else 0
    
        # --- OUTPUT RESULTS LAYER ---
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Corrected closed container layout card block
    st.markdown("""
        <div style="background-color: #FFFFFF; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); border-top: 6px solid #0D5C53; margin-bottom: 2rem;">
            <h3 style="margin: 0; color: #333333; font-weight: 600;">📊 Analytical Outcome: FageraTech Risk Report</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Split metrics viewport out natively without nested formatting blocks
    m1, m2 = st.columns(2)
    
    with m1:
        if binary_label == 1:
            st.error("🚨 **Classification Label: CLIMATE-SENSITIVE MORTALITY ACCELERATION RISK**\n\nThe environmental factors combined with patient demographics suggest high historical correlation with climate stress anomalies.")
        else:
            st.success("🟢 **Classification Label: BASELINE CLINICAL BIOLOGICAL ATTRIBUTES**\n\nThe environmental risk trends are within regular variances for this demographic window.")
            
    with m2:
        st.metric(label="Continuous Probability Index (TargetRAUC Metrics)", value=f"{risk_probability:.4f}")
        st.progress(float(risk_probability))
