# 🛩️ Predictive Maintenance & Remaining Useful Life (RUL) Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Objective
Develop a machine learning approach to estimate the **Remaining Useful Life (RUL)** of industrial aircraft engines using historical sensor and operating-condition data. 

Unexpected equipment failure can result in significant maintenance costs and downtime. Predictive maintenance attempts to estimate equipment degradation before failure, allowing for timely interventions and minimizing disruptions.

## 📖 Table of Contents
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Models & Performance](#-models--performance)
- [Key Findings](#-key-findings)
- [Technologies](#-technologies)

## 📊 Dataset
The dataset used is the **NASA Turbofan Engine Degradation Simulation Data Set (FD001)**.
- **Time-series data**: Recorded per operating cycle for various engines.
- **Features**: Includes 3 operational settings and 21 sensor measurements.
- **Scenario**: The engine operates normally at the start and develops a fault over time until system failure.

## 📂 Project Structure
```text
predictive-maintenance/
│
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for exploration and modeling
├── src/                # Source code for data processing
├── figures/            # Generated plots and visualizations
├── Project_Report.md   # Detailed project report
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/predictive-maintenance.git
   cd predictive-maintenance
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

The project analysis is primarily driven through Jupyter Notebooks. You can explore the data and run the models by starting the Jupyter environment:

```bash
jupyter notebook
```
Navigate to the `notebooks/` directory and run the notebooks in sequential order to replicate the preprocessing, exploratory data analysis, and modeling steps.

## 🔬 Methodology

1. **Data Understanding & EDA**: Explored time-series trends and removed constant sensor readings with no predictive power.
2. **Target Variable Creation (RUL)**: Formulated the continuous target variable `RUL` based on the remaining cycles until failure.
3. **Feature Engineering**: Generated moving averages and rolling standard deviations (window size = 5) to capture long-term trends and volatility.
4. **Time-Aware Validation**: Split data by `engine_id` to prevent data leakage and simulate real-world evaluation.
5. **Modeling**: Trained and evaluated regression models to predict the RUL.

## 📈 Models & Performance

We trained three regression models to predict the RUL, evaluating them based on **Root Mean Squared Error (RMSE)**:
- **Linear Regression** (Baseline)
- **Random Forest Regressor**
- **XGBoost Regressor**

*(Check the notebooks for exact performance metrics and comparisons).*

## 💡 Key Findings
- **Non-Linear Relationships**: Tree-based models (XGBoost and Random Forest) significantly outperformed the Linear Regression baseline, suggesting the relationship between sensor measurements and engine degradation is highly nonlinear.
- **Noise Reduction**: Features generated from rolling statistics smoothed out sensor noise and provided clearer degradation signals, heavily improving model performance.
- **Crucial Sensors**: Sensors 11, 4, and 7 showed the highest feature importance across tree-based models, highlighting their utility in signaling late-stage engine wear.

## 🛠️ Technologies
- **Python**
- **Pandas** & **NumPy** for data manipulation
- **Matplotlib** & **Seaborn** for data visualization
- **Scikit-learn** for machine learning pipelines and metrics
- **XGBoost** for advanced gradient boosting regression

---
*This project demonstrates how multivariate sensor data can effectively estimate equipment remaining useful life and support data-driven predictive maintenance decisions.*
