# 🏥 Health Prediction in Uganda & Climate Changes
### 🔬 Operational Epidemiological Risk Intelligence Engine Framework — Powered by FageraTech

[![Streamlit App](https://streamlit.io)](https://streamlit.io)
[![Production ML Engine](https://shields.io)]
[![License: MIT](https://shields.io)](https://opensource.org)

Health outcomes are shaped by more than biology alone. In many low-resource settings, changing environmental dynamics—such as flash floods, soil saturation profiles, and thermal heat stress anomalies—substantially accelerate clinical vulnerability. 

This repository hosts a production-grade predictive intelligence framework that maps out individual demographic vulnerabilities against high-tier multi-scale historical climate lag matrices. Powered by a robust, highly-regularized **LightGBM Gradient Boosted Tree Ensemble**, the engine classifies whether an observation falls into a climate-sensitive tracking category.

---

## 📊 Evaluation Matrix & Constraints

To maintain absolute structural alignment with strict scientific and platform requirements:
1. **The Multi-Metric Layout:** Models are optimized and scored using a weighted multi-metric validation blend: **60% Weight to Out-of-Fold F1-Score** and **40% Weight to ROC-AUC**.
2. **The Rigid Threshold Boundary:** Manual probability threshold optimizations are strictly prohibited. Binary labels (`TargetF1`) are strictly determined using the default **0.5 probability cutoff rules**.

---

## 🛠️ Data Pipeline & Feature Engineering Architecture

The engine transforms raw demographic inputs alongside trailing weather windows (extracted from CHIRPS Daily and ERA5-Land Daily aggregations) into high-tier interaction signals:

- **🕒 Temporal Decompositions:** Translates text datetime strings into cyclical signal dimensions (`month_sin`, `month_cos`) to capture seasonal mortality spikes natively.
- **🔥 Thermal Shock Fluctuations:** Pre-computes deviations across historical windows (`acute_thermal_shock = tavg_7d - tavg_30d`) to pinpoint sudden hot/cold snaps.
- **🌧️ Moisture Accumulation Ratios:** Captures ground drainage and prolonged rain saturation pressures safely avoiding division-by-zero errors (`rain_sum_7d / (rain_sum_30d + 1)`).
- **🛡️ Data Leakage Guardrails:** Categorical string inputs (`location`) are handled using cross-validated, leak-free out-of-fold target encoding transformations coupled with M-estimate frequency smoothing.

---

## 📁 Repository Directory Structure

```text
📁 climate-Health-Uganda/
│
├── 📁 models/                  ➔ Serialized Production Asset Registry
│   ├── pipeline_metadata.joblib ➔ Label encoders, imputers, and target encoding maps
│   └── production_lgbm_model.txt➔ Serialized LightGBM Booster matrix parameters
│
├── 📄 climate_health_analysis.ipynb ➔ Experimental prototyping and cross-validation loop
├── 📄 app.py                    ➔ Premium thematic medical Streamlit user interface
├── 📄 requirements.txt         ➔ Pinned package dependencies manifest
├── 📄 README.md                ➔ Enterprise project documentation
└── 📊 Train.csv / Test.csv     ➔ Local raw dataset tables
```

---

## 💻 Local Installation & App Execution

Follow these steps to run the interactive dashboard locally within your workspace terminal:

### 1. Set Up and Align Environment Dependencies
```bash
# Install required production packages from manifest
pip install -r requirements.txt
```

### 2. Execute the Serialization Script
Before starting the interface, ensure you run your training workbook or serialization module to output your production model parameters inside the `/models` directory:
```bash
python serialize_production_pipeline.py
```

### 3. Boot Up Your Streamlit Dashboard Panel
```bash
streamlit run app.py
```

---

## 🚀 MLOps Containerization (Docker)

To deploy the dashboard securely inside isolated container infrastructure (AWS, Google Cloud, or local Docker environments), build and run the provided Docker architecture:

```bash
# 1. Compile the immutable container image
docker build -t climate-health-uganda:v1 .

# 2. Spin up the containerized network mapping port 8501
docker run -p 8501:8501 climate-health-uganda:v1
```

---

## 📝 Terms & Citation
Developed under the enterprise footprint of **FageraTech**. For public health research queries, model telemetry reviews, or code adjustments, please open an Issue directly inside the GitHub interface.
