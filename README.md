# 🩺 Healthcare Survival Prediction System

[![Streamlit App](https://static.streamlit.io/badge-streamlit.svg)](https://healthcare-survival-prediction-system.streamlit.app/)
[![GitHub license](https://img.shields.io/github/license/Yogesh-DataScientist/Healthcare-Survival-Prediction-System)](https://github.com/Yogesh-DataScientist/Healthcare-Survival-Prediction-System/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)

An interactive, premium-grade Clinical Decision Support System (CDSS) built with **Streamlit**, **scikit-learn**, and **XGBoost**. This system predicts patient survival probability in Intensive Care Units (ICUs) using vital clinical signals, demographic information, and pre-existing comorbidities.

🔗 **Live Demo:** [healthcare-survival-prediction-system.streamlit.app](https://healthcare-survival-prediction-system.streamlit.app/)

---

## 🎨 Premium User Experience & Theme

Designed with advanced UI/UX considerations, this application includes:
- **Adaptable Dark/Light Theme**: A custom design language featuring custom-curated, highly legible emerald/neutral-slate color palettes.
- **Dynamic Particle System**: Interactive background canvas particles toggled via the settings panel.
- **Micro-Animations**: Custom CSS button ripple effects, smooth progress bar load transitions, and animated results rendering.
- **Form Helpers**: Single-click "Generate Healthy Patient" or "Generate Critical Patient" mock data buttons to quickly test the model prediction pipeline.

---

## 🚀 Key Modules

### 1. 🏠 Home Page
- Conceptual overview of the system.
- Live key metrics showcase showing XGBoost's hero accuracy (e.g. `87.5%`).
- Quick-start navigation pathing.

### 2. 📈 EDA Dashboard (Exploratory Data Analysis)
- Dynamic visualization of patient distributions (vitals, demographics, comorbidities).
- Interactive histograms, scatter plots, and correlation heatmaps powered by Plotly.
- Responsive data filtering.

### 3. 🩺 Survival Prediction Engine
- Real-time diagnostic evaluation of a patient based on 18 critical parameters (e.g., GCS Scores, Heart Rate, Respiration Rate, Mean Arterial Pressure, Glucose, BMI, Diabetes, Cirrhosis).
- **Dual-Model Predictors**: Choose dynamically between a high-accuracy **XGBoost Classifier** and a robust **AdaBoost Classifier**.
- Visual confidence breakdown displaying survival vs mortality probabilities.

### 4. ⚙️ Settings & Model Comparison Evaluator
- Customize system state (Theme toggle, animation configurations, particle system controls).
- **Model Performance Metrics**: Detailed side-by-side comparison of **Accuracy, Precision, Recall, and F1-Score**.
- **Interactive Confusion Matrix Visualizer**: Directly compare how XGBoost and AdaBoost perform on patient test slices.

---

## 🛠️ Tech Stack & Dependencies

- **Core Framework**: [Streamlit](https://streamlit.io/) (v1.20.0+)
- **Data Wrangling**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Interactive Charts**: [Plotly Express & Plotly Graph Objects](https://plotly.com/python/)
- **Machine Learning**: [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- **Model Storage**: Pickle serialized pipelines

---

## 📦 Installation & Local Setup

Run the application locally by following these steps:

### Prerequisites
Make sure you have **Python 3.8+** installed.

### 1. Clone the Repository
```bash
git clone https://github.com/Yogesh-DataScientist/Healthcare-Survival-Prediction-System.git
cd Healthcare-Survival-Prediction-System
```

### 2. Install Dependencies
Install all package requirements specified in the manifest:
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
Launch the web server locally:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🧬 Machine Learning Pipeline

The predictors are trained on clinical records using **18 features**:
- **Demographics**: `Age`, `BMI`, `Gender`, `Ethnicity`
- **ICU Admission Info**: `ICU Admission Source`, `ICU Stay Type`
- **Physiological / Vital Signals**: `Heart Rate`, `Respiration Rate`, `MAP (Mean Arterial Pressure)`, `Min d1 SpO2`, `Max d1 Temperature`, `Max d1 Glucose`
- **Neurological Assessment**: Glasgow Coma Scale (`GCS Eyes`, `GCS Motor`, `GCS Verbal`)
- **Comorbidities**: `Diabetes Mellitus`, `Cirrhosis`

The models evaluate these attributes to calculate survival probabilities.

---

## 📜 License

Distributed under the MIT License. See [LICENSE](https://github.com/Yogesh-DataScientist/Healthcare-Survival-Prediction-System/blob/main/LICENSE) for more information.

---

**Developed with ❤️ by [Yogesh-DataScientist](https://github.com/Yogesh-DataScientist)**
