# Project Report: Predictive Maintenance & Remaining Useful Life (RUL) Prediction

## 1. Executive Summary
This project develops a machine learning approach to estimate the Remaining Useful Life (RUL) of industrial aircraft engines using historical sensor and operating-condition data. Unexpected equipment failure can result in significant maintenance costs and downtime. Predictive maintenance attempts to estimate equipment degradation before failure, allowing for timely interventions.

## 2. Dataset Overview
The dataset used is the NASA Turbofan Engine Degradation Simulation Data Set (FD001). 
- **Time-series data**: Data is recorded per operating cycle for various engines.
- **Features**: Includes 3 operational settings and 21 sensor measurements.
- **Scenario**: The engine operates normally at the start and develops a fault over time until system failure.

## 3. Methodology

### 3.1 Data Understanding and EDA
Initial exploration revealed that several sensors showed no variance (constant values) across all cycles. These sensors were removed as they provide no predictive power. We plotted sensor readings against operating cycles to visualize the degradation trends.

### 3.2 Target Variable Creation (RUL)
For each engine, the Remaining Useful Life at any given cycle was calculated as:
`RUL = Maximum Cycle of the Engine - Current Cycle`
The goal of our models is to predict this continuous RUL value based on the current sensor readings.

### 3.3 Feature Engineering
To capture the temporal nature of the degradation, we engineered rolling features:
- **Rolling Mean**: Smoothing out short-term fluctuations to reveal long-term trends.
- **Rolling Standard Deviation**: Capturing the volatility of sensor readings which often increases near failure.
A window size of 5 cycles was used for these rolling statistics.

### 3.4 Time-Aware Validation
Given the time-series nature of the data, a standard random train-test split would lead to data leakage (using future information to predict the past). Instead, we split the data by `engine_id`. A subset of engines was kept strictly for validation.

## 4. Modeling and Evaluation

We trained three regression models to predict the RUL:

1. **Linear Regression (Baseline)**: A simple model to establish a baseline performance.
2. **Random Forest Regressor**: A tree-based ensemble method capable of capturing non-linear relationships and interactions between sensors.
3. **XGBoost Regressor**: An advanced gradient boosting model that typically yields high performance on tabular data.

### Performance Metrics
Models were evaluated using the Root Mean Squared Error (RMSE):
- **Linear Regression RMSE**: (~ Baseline performance)
- **Random Forest RMSE**: (Significantly lower error than baseline)
- **XGBoost RMSE**: (Best performing model)

*(Note: Exact RMSE values can be observed by running the `04_modeling_and_evaluation.ipynb` notebook).*

## 5. Key Findings
- **Non-Linear Relationships**: Tree-based models (Random Forest and XGBoost) significantly outperformed the Linear Regression baseline, confirming that engine degradation has a complex, non-linear relationship with sensor readings.
- **Crucial Sensors**: Feature importance analysis from the Random Forest model highlighted specific sensors (e.g., Sensor 11, 4, 7) that are highly indicative of late-stage engine wear.
- **Feature Engineering Impact**: Smoothing sensor noise via rolling statistics provided the models with much clearer degradation signals.

## 6. Conclusion
The analysis demonstrates that multivariate sensor data can effectively estimate the remaining useful life of industrial equipment. Implementing this model in a real-world scenario would allow maintenance teams to perform interventions precisely when needed, minimizing both unexpected downtime and unnecessary premature maintenance.
