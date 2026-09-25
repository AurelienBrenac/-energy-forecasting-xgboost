# ⚡ Hourly Energy Consumption Forecasting using XGBoost & LightGBM

## 📌 Project Overview
We developed a robust, production-grade Machine Learning pipeline to forecast hourly electricity demand (in Megawatts) for the **PJM Interconnection (PJME)** utility grid [0.1]. Using historical load data spanning over a decade, we benchmarked and optimized two powerful gradient boosting frameworks—**XGBoost** and **LightGBM**—while enforcing strict time-series validation workflows to eliminate look-ahead biases and combat concept drift [0.1].

## 🛠️ Data, Tech Stack & Installation
* **Dataset**: PJM Hourly Energy Consumption (Kaggle) - 145,366 records.
* **Core Stack**: Python 3.12, XGBoost, LightGBM, Scikit-Learn, Pandas, Numpy, Holidays.
* **Visualization**: Matplotlib, Seaborn.

To replicate this environment locally, clone the repository and run:
```bash
pip install -r requirements.txt
```

---

## 🔬 Methodology & Validation Strategies
In time-series forecasting, standard random K-Fold cross-validation introduces fatal **data leakage** by using future data to predict past horizons. To ensure real-world operational reliability, **we benchmarked three distinct chronological validation frameworks**:

1. **TimeSeriesSplit (Standard Cumulative)**: Training window expands sequentially over time.
2. **Walk-Forward (Expanding Window)**: Simulates sequential, chronological production re-training.
3. **Walk-Forward (Rolling Window)**: A rolling window cross-validation framework designed to evaluate model resistance to **concept drift** by discarding obsolete historical consumer behavior [0.1].

---

## 📈 Experimental Results & Architectural Comparison

### 1. Iterative Feature Engineering & Model Cross-Validation
We conducted our experiments to measure the explicit value of our **Advanced Feature Engineering** (incorporating US holiday effects, meteorological season proxies, and weekend shifts) combined with a target **Log Transformation** (\(y_{log} = \ln(y + 1)\)) to stabilize error variance:

| Validation Strategy | Model Architecture | Feature Engineering & Transform | Mean RMSE | Standard Deviation (std) | Variance |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **TimeSeriesSplit** | XGBoost Regressor | Baseline (6 features) | 4148.8880 | 237.0124 | 56174.89 |
| **TimeSeriesSplit** | XGBoost Regressor | Advanced + Log Transform | 3609.2941 | 227.2302 | 51633.56 |
| **TimeSeriesSplit** | LightGBM Regressor | Advanced + Log Transform | 3600.5162 | 232.1780 | 53906.63 |
| **Walk-Forward (Expanding)** | XGBoost Regressor | Baseline (6 features) | 4268.2208 | 193.8109 | 37562.65 |
| **Walk-Forward (Expanding)** | XGBoost Regressor | Advanced + Log Transform | 3720.0830 | 194.7584 | 37930.82 |
| **Walk-Forward (Expanding)** | LightGBM Regressor | Advanced + Log Transform | 3718.8382 | 193.4372 | 37417.95 |
| **Walk-Forward (Rolling)** | XGBoost Regressor | Baseline (6 features) | 4289.2250 | 196.6303 | 38663.48 |
| **Walk-Forward (Rolling)** | XGBoost Regressor | Advanced + Log Transform | 3737.3182 | 190.7419 | 36382.48 |
| **Walk-Forward (Rolling)** | LightGBM Regressor | Advanced + Log Transform | 3731.9049 | 194.7935 | 37944.51 |
| **Walk-Forward (Rolling)** | **XGBoost (Hyperparameter Tuned)** | **Advanced + Log Transform** | **3734.3830** | **188.4210** | **35502.53** |
| **Walk-Forward (Rolling)** | **LightGBM (Hyperparameter Tuned)** | **Advanced + Log Transform** | **3851.5402** | **192.1140** | **36907.82** |

### 2. Hyperparameter Optimization Takeaways
We implemented separate, automated Random Search optimization loops tailored to the rolling-window validation scheme.
* **XGBoost** won the predictive benchmark (`Mean RMSE: 3,734.38 MW`), using a structural `max_depth=5` approach that seamlessly generalized long-term cyclical trends.
* **LightGBM** executed significantly faster but proved more sensitive to hourly noise (`Mean RMSE: 3,851.54 MW`), with its hyper-parameters settling at a restricted `num_leaves=15` layout to prevent severe overfitting.

---

## 🔍 Residual Analysis & Error Diagnostics
* **Heteroscedasticity Analysis**: Our sequential residual plots confirmed that introducing the target log transformation successfully flattened and compressed error variance across standard grid operating frames.
* **The Residual Funnel Reality**: However, both diagnostics exposed a clear shared limitation: a residual funnel expands during peak intervals exceeding 40,000 MW. This visual proof systematically demonstrates that non-linear tree ensembles require explicit exogenous weather variables (e.g., hourly temperature metrics) rather than purely historical calendar filters to fully stabilize tail-end variance during extreme seasonal climate shifts.

---

## 🚀 Next Steps & Future Operational Perspectives
1. **Continuous Data Ingestion & API Integration**: Automating data streaming via the **PJM Open Data Miner API** up to 2026.
2. **Mitigating Post-2020 Concept Drift (The COVID-19 Shock)**: Testing the optimized pipelines on post-2020 windows to map the permanent macroeconomic shifts caused by remote-work profiles.
3. **Exogenous Feature Integration**: Extracting historical NOAA weather station records to deploy explicit Heating/Cooling Degree Days (HDD/CDD) indicators.