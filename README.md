# Forecasting Paddy Production in Sri Lanka

A comparative study of time series and machine learning models for forecasting paddy production, average yield and sown extent in Sri Lanka, combined with an analysis of long-term farming practice trends.

> B.Sc. (Honours) in Applied Sciences - Department of Statistics, University of Sri Jayewardenepura, Sri Lanka
> Carried out in collaboration with the Department of Census and Statistics, Sri Lanka

---

## Overview

Rice is Sri Lanka's most important food crop, and accurate forecasts of paddy production are essential for food security planning. This project has two main goals:

1. **Forecast** paddy production, average yield and sown extent for both the Maha and Yala growing seasons, using seasonal data from 1992 to 2025.
2. **Analyze** how key farming practices — sowing methods, fertilizer application and weeding methods — have changed over time and how they relate to one another.

Six forecasting models (ARIMA, SARIMA, Holt's Linear Trend, Holt-Winters Exponential Smoothing, LSTM and Random Forest) were built and compared. A Vector Autoregression (VAR) model with Granger causality testing was used to study relationships between farming practice trends.

## Data

- **Source:** Department of Census and Statistics, Sri Lanka
- **Period:** 1992–2025 (67 seasonal observations, Maha and Yala seasons)
- **Forecasting variables:** paddy production ('000 metric tons), average yield (kg/hectare), sown extent (hectares)
- **Farming practice variables:** sowing method, fertilizer application method, weeding method (recorded as % share of farmers using each method)

## Methodology

**Pre-processing:** Data cleaned and indexed by date using `pandas`. Stationarity checked with the Augmented Dickey-Fuller (ADF) test; ACF/PACF plots used to support model order selection.

**Train-test split:** 80:20 for ARIMA, SARIMA, Holt's, Holt-Winters and Random Forest. A 70:10:20 train-validation-test split for LSTM.

**Forecasting models:**
| Model | Approach |
|---|---|
| ARIMA / SARIMA | Automatic order selection via `auto_arima` (AIC-based), seasonal period = 2 |
| Holt's Linear Trend | Exponential smoothing with damped trend |
| Holt-Winters | Multiplicative seasonal exponential smoothing |
| Random Forest | Lag-feature regression (t-1, t-2, t-3), tuned with `GridSearchCV` + `TimeSeriesSplit` |
| LSTM | PyTorch sequence model, hyperparameters tuned with Optuna (50 trials) |

**Evaluation metrics:** RMSE, MAE, and MAPE on held-out test data.

**Farming practice analysis:** Temporal trend plots for each practice category, followed by VAR modelling and Granger causality tests to identify predictive relationships between practices.

## Results

### Model comparison (lower is better)

**Paddy Production**
| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| **SARIMA (Best)** | 295.97 | 377.59 | **12.96** |
| Holt-Winters | 325.45 | 384.64 | 14.47 |
| LSTM | 433.89 | 516.76 | 18.27 |
| Random Forest | 586.50 | 696.14 | 24.91 |

**Average Yield**
| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| ARIMA | 544.57 | 739.64 | 15.08 |
| Holt's Method | 504.23 | 662.15 | 13.79 |
| **LSTM (Best)** | 487.94 | 587.15 | **12.77** |
| Random Forest | 596.91 | 722.14 | 15.30 |

**Sown Extent**
| Model | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| SARIMA | 116,163.01 | 123,103.06 | 19.24 |
| Holt-Winters | 93,998.41 | 98,317.19 | 15.29 |
| **LSTM (Best)** | 78,773.66 | 92,114.12 | **14.29** |
| Random Forest | 112,155.71 | 131,921.85 | 17.30 |

**Key finding:** No single model won for every variable. SARIMA performed best for paddy production because of its strong, explicit seasonal pattern. LSTM performed best for average yield and sown extent, both of which show non-linear trends and irregular shocks (e.g. the 2021–2022 fertilizer ban) that classical models struggle to capture.

### Three-year forecast (2026–2028)

- **Paddy production:** stable, with Maha season output around 2,900–2,937 thousand metric tons and Yala season output around 2,034–2,113 thousand metric tons.
- **Average yield:** gradual recovery, reaching approximately 4,350–4,375 kg/hectare by 2028.
- **Sown extent:** relatively stable, with Maha season extent around 747,000–779,000 hectares and Yala season extent around 427,000–462,000 hectares.

### Farming practice trends

- **Fertilizer application:** chemical-only use dominated until around 2010, after which combined chemical-and-organic use grew steadily. The 2021–2022 chemical fertilizer ban caused a sharp, temporary surge in organic-only use and a collapse in chemical-only use.
- **Sowing methods:** broadcasting remains dominant (over 90% of usage), though transplanting methods are slowly gaining ground.
- **Weeding methods:** weedicide use rose from about 60% in the early 1990s to 80–90% in recent years, largely replacing hand weeding.
- **VAR / Granger causality:** confirmed statistically significant predictive relationships between competing practices - for example, bidirectional causality between hand weeding and weedicide use, and broadcasting's changes helping predict future changes in transplanting.

## Tech Stack

- **Language:** Python
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Statistical modelling:** `statsmodels` (ARIMA, SARIMA, Holt's, Holt-Winters, ADF test, VAR, Granger causality), `pmdarima` (`auto_arima`)
- **Machine learning:** `scikit-learn` (Random Forest, Grid Search, evaluation metrics)
- **Deep learning:** `PyTorch` (LSTM), `Optuna` (hyperparameter tuning)


## Recommendations

- Use a variable-specific model selection approach rather than one model for all targets — seasonal variables favor SARIMA, non-linear variables favor LSTM.
- Incorporate climate variables (seasonal rainfall, temperature) as additional predictors in future work.
- Introduce future fertilizer policy changes gradually, given the sharp yield impact observed from the 2021–2022 ban.
- Extend the analysis to district-level data to capture regional variation.
- Retrain and update the models each season as new data becomes available.

## Author

**I. A. A. Wathsala,**
Department of Statistics, University of Sri Jayewardenepura, Sri Lanka

Supervised by Mr. P. Dias (University of Sri Jayewardenepura) and Mr. S. D. S. Nimesha (Department of Census and Statistics)

## Acknowledgements

This research was carried out in collaboration with the Department of Census and Statistics, Sri Lanka, which provided the dataset used in this study.

