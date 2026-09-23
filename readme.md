# ⚡ Hourly Energy Consumption Forecasting using XGBoost

## 📌 Project Overview
We developed a robust, production-grade Machine Learning pipeline to forecast hourly electricity demand (in Megawatts) for the **PJME (PJM Interconnection)** grid. Using historical load data spanning over a decade, we built an optimized **XGBoost Regressor** capable of capturing complex daily, weekly, and seasonal patterns while ensuring strict validation methodologies to prevent temporal data leakage.

## 🛠️ Data & Tech Stack
* **Dataset**: PJM Hourly Energy Consumption (Kaggle) - 145,366 records.
* **Core Stack**: Python 3.12, XGBoost, Scikit-Learn, Pandas, Numpy.
* **Visualization**: Matplotlib, Seaborn.

---

## 🔬 Methodology & Validation Strategies
In time-series forecasting, standard K-Fold cross-validation introduces severe **data leakage** by using future data to predict the past. To address this challenge and ensure real-world reliability, **we benchmarked three distinct validation frameworks**:

1. **TimeSeriesSplit (Standard Cumulative)**: Training window expands sequentially over time.
2. **Walk-Forward (Expanding Window)**: Our custom split simulating sequential, chronological production re-training without global leakage.
3. **Walk-Forward (Rolling Window)**: A rolling cross-validation framework designed to evaluate model resistance to **concept drift** by discarding obsolete historical data.

---

## 📈 Experimental Results & Feature Engineering

### 1. Iterative Feature Engineering Performance
We conducted our experiments in two separate phases to measure the explicit value of our **Feature Engineering Stage** combined with a target **Log Transformation** (\(y_{log} = \ln(y + 1)\)) to stabilize error variance:

| Validation Strategy | Feature Engineering Stage | Mean RMSE | Standard Deviation (std) | Variance |
| :--- | :--- | :---: | :---: | :---: |
| **TimeSeriesSplit** | Baseline (6 features) | 4148.8880 | 237.0124 | 56174.89 |
| **TimeSeriesSplit** | Advanced + Log Transform | 4085.2611 | 242.0673 | 58596.58 |
| **Walk-Forward (Expanding)** | Baseline (6 features) | 4268.2208 | 193.8109 | 37562.65 |
| **Walk-Forward (Expanding)** | Advanced + Log Transform | 4206.0066 | 211.7308 | 44829.92 |
| **Walk-Forward (Rolling)** | Baseline (6 features) | 4289.2250 | 196.6303 | 38663.48 |
| **Walk-Forward (Rolling)** | Advanced + Log Transform | 4231.4814 | 181.8153 | 33056.80 |
| **Walk-Forward (Rolling)** | **Optimized Model (Hyperparameter Tuned)** | **3795.2978** | **156.4210** | **24467.53** |

### 2. Hyperparameter Optimization Tuning
After mapping our cross-validation strategies, **we implemented an automated Random Search framework** integrated within our strict rolling-window validation. 
* Our pipeline evaluated multiple parameters and found the optimal configuration: **Learning Rate: 0.01, Max Depth: 5, Estimators: 1000, Subsample: 0.8, Colsample_bytree: 0.9**.
* This fine-tuning reduced our final Mean RMSE to **3795.29**—a massive **11.5% overall increase in model accuracy** compared to the rolling baseline.

---

## 📊 Feature Importance Insights
According to our champion model's weight attribution:
* **Hourly Profile (`hour` - 33.67%)**: Remains the most crucial driver, capturing intra-day peak electricity usage (morning routines vs. evening spikes).
* **Weekly/Seasonal Shifts (`dayofweek`, `dayofyear`, `month`, `season` - over 58% combined)**: Strongly dominate predictions, mapping winter heating and summer air-conditioning loads.
* **Holiday Impacts (`is_holiday` - 3.29%)**: Successfully isolates commercial/industrial shutdowns on major calendar events.

---

## 🔍 Residual Analysis & Error Diagnostics
To complete our technical validation, we evaluated our configurations sequentially:
* **Heteroscedasticity Analysis**: Our baseline scatter plot clearly exposed a classic megaphone-shaped heteroscedasticity pattern. By introducing the logarithmic transformation, **we successfully compressed this variance expansion**, resulting in a much more symmetrical distribution.
* **The Residual Funnel Reality**: While the log transformation optimized our relative error metrics, a minor residual funnel remains visible at extreme peaks (exceeding 40,000 MW). This proves that non-linear tree ensembles like XGBoost require explicit exogenous variables, such as hourly temperature logs, to fully stabilize predictions during massive climate-driven grid loads.

---

## 🚀 Next Steps & Future Operational Perspectives
Scaling this architecture for real-world utility grid deployment would involve several logical expansions:
1. **Continuous Data Ingestion & API Integration**: Connecting our data layer directly to the **PJM Open Data Miner API** (`://pjm.com`) to automate hourly data streaming from 2019 up to 2026.
2. **Mitigating Post-2020 Concept Drift (The COVID-19 Shock)**: Testing our optimized XGBoost model on the post-2020 window to evaluate how well our feature matrix adapts to the heavy structural remote-work trend break.
3. **Exogenous Feature Integration (Weather Metrics)**: Pulling historical hourly weather station data (NOAA) to replace our proxy `season` feature with exact heating/cooling degree days (HDD/CDD) to unlock the ultimate layer of forecasting precision.
