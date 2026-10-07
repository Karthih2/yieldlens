<img src="https://img.shields.io/badge/YieldLens-Stability--Aware%20Sensor%20Reduction-1F6FEB?style=for-the-badge" width="100%">

<h1 align="center">🔬 Semiconductor Yield Intelligence</h1>

<p align="center"><b>Sensor Stability • Yield Prediction • Explainable ML</b></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/SECOM-Semiconductor%20Dataset-orange">
  <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-Prediction-red?logo=xgboost&logoColor=white">
  <img src="https://img.shields.io/badge/SHAP-Explainability-purple">
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white">
</p>

<p align="center">
  A stability-aware semiconductor yield intelligence system that reduces high-dimensional sensor data, predicts wafer failure risk, and provides interpretable model explanations.
</p>

---

## 📋 Project Overview

**Key Features:**

* **Stability-Aware Sensor Reduction** - Reduces 590 sensor features to a stable shortlist of 17 sensors
* **Chronological Analysis** - Evaluates sensor behavior across five manufacturing batches
* **Yield Prediction** - Predicts wafer failure probability using machine learning
* **Explainable ML** - Uses SHAP to identify important sensor contributions
* **Drift Validation** - Evaluates sensor stability under controlled drift
* **Interactive Dashboard** - Provides prediction, stability, and explainability views

---

## 🏗️ System Architecture

<p align="center">
  <img src="YieldLens%20System%20Architecture%202.drawio.png" alt="YieldLens System Architecture" width="95%">
</p>

YieldLens combines sensor preprocessing, stability analysis, feature reduction, machine learning, explainability, and a FastAPI-based monitoring application into a single workflow.

---

## 🔄 Processing Pipeline

### 1️⃣ Data Preparation

* Load the SECOM semiconductor dataset
* Combine sensor measurements with wafer labels
* Handle missing values using median imputation
* Preserve chronological manufacturing order

### 2️⃣ Sensor Quality & Stability Analysis

* Audit sensor quality using tripwire checks
* Divide the data into five chronological batches
* Perform batch-wise feature selection
* Calculate sensor stability scores
* Validate stability using controlled drift injection

### 3️⃣ Sensor Reduction

```text
590 Sensors
     │
     ▼
Quality Assessment
     │
     ▼
Chronological Batches
     │
     ▼
Batch-wise Selection
     │
     ▼
Stability Analysis
     │
     ▼
17 Shortlisted Sensors
```

### 4️⃣ Yield Prediction

* Train machine learning models using the selected sensors
* Use SMOTE to address class imbalance
* Evaluate models across chronological batches
* Deploy a reduced-feature Random Forest model

### 5️⃣ Explainability

* Calculate global SHAP importance
* Generate per-wafer SHAP explanations
* Identify sensors contributing to prediction risk

---

## 📊 Dataset

| Property              | Value |
| --------------------- | ----: |
| Dataset               | SECOM |
| Wafer Records         | 1,567 |
| Sensor Features       |   590 |
| Failure Rate          | ~6.6% |
| Chronological Batches |     5 |
| Final Sensors         |    17 |

### Batch Distribution

```text
314 → 313 → 313 → 313 → 314
```

---

## 📈 Key Results

| Metric                    | Result |
| ------------------------- | -----: |
| Original Sensors          |    590 |
| Final Shortlisted Sensors |     17 |
| Chronological Batches     |      5 |
| Wafer Records             |  1,567 |
| Failure Rate              |  ~6.6% |

The stability-aware reduction provides a compact sensor representation for the deployed yield prediction model while retaining stability and feature importance as selection criteria.

---

## 🔍 Explainability

YieldLens uses **SHAP** to explain both overall model behavior and individual wafer predictions.

The system provides:

* Global sensor importance
* Per-prediction sensor contributions
* Stability score versus feature importance analysis

This allows users to understand not only the predicted wafer risk, but also the sensor measurements influencing the prediction.

---

## 🛠️ Technologies Used

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/XGBoost-EC0000?style=flat-square">
  <img src="https://img.shields.io/badge/SHAP-purple?style=flat-square">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&logo=sqlite&logoColor=white">
  <img src="https://img.shields.io/badge/Chart.js-FF6384?style=flat-square&logo=chart.js&logoColor=white">
</p>

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Node.js and npm

### Installation

**1. Install dependencies**

```bash
pip install -r backend/requirements.txt
```

**2. Install frontend dependencies**

```bash
cd frontend
npm install
cd ..
```

**3. Start YieldLens**

```bash
run_app.bat
```

The application runs at:

```text
http://127.0.0.1:8000/ui/
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 📚 Key Learnings

* **Sensor stability matters** when working with high-dimensional manufacturing data.
* **Chronological validation** provides a more realistic assessment of changing process conditions.
* **Feature reduction** simplifies model deployment and interpretation.
* **Explainability** helps connect model predictions with individual sensor behavior.


---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👤 Author

**Karthick S**

---

<div align="center">

✨ <i>A stability-aware approach to sensor reduction, semiconductor yield prediction, and explainable manufacturing analytics.</i> ✨

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1F6FEB,100:6B7280&height=120&section=footer" width="100%"/>

</div>
