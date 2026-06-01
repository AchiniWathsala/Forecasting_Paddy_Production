# =========================================================
# FERTILIZER USAGE FORECASTING — FIXED VERSION
#
# KEY FIXES APPLIED:
#   1. Mixed integration orders handled correctly:
#        - Organic_only (I(0))  → individual ARIMA
#        - Chemical_only, Both  (I(2)) → VAR on 2nd differences
#   2. Lag capped at max=4 and constrained by sample size so
#      we don't burn all degrees of freedom (n=67 is small)
#   3. Inverse differencing done carefully per-series
#   4. Future forecast clipped + renormalized so rows sum to 100
#   5. Evaluation uses aligned actual vs forecast indices
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_excel("df.xlsx")

df = data.iloc[27:, :]
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

df = df[['Date', 'Chemical_only', 'Organic_only', 'Both', 'None']]
df.set_index('Date', inplace=True)

print("Original Data Shape:", df.shape)
print(df.head())

# =========================================================
# ROW SUM CHECK
# =========================================================

df['Row_Sum'] = df.sum(axis=1)
print("\nRow Sum Check (should be ~100):")
print(df['Row_Sum'].describe())
df.drop(columns='Row_Sum', inplace=True)

# =========================================================
# PLOT ORIGINAL DATA
# =========================================================

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['steelblue', 'seagreen', 'darkorange', 'tomato']
labels_map = {
    'Chemical_only': 'Chemical Fertilizer Only',
    'Organic_only':  'Organic Fertilizer Only',
    'Both':          'Both Chemical and Organic',
    'None':          'None'
}

for col, color in zip(df.columns, colors):
    ax.plot(df.index, df[col], marker='o', markersize=3, label=labels_map[col], color=color)

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.title("Year-wise Distribution of Fertilizer Usage")
plt.legend(title="Fertilizer Type", loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# ADF TEST HELPER
# =========================================================

def adf_check(series):
    result = adfuller(series.dropna(), autolag='AIC')
    return result[0], result[1]

def find_integration_order(series, max_d=3):
    """Return the number of differences needed to achieve stationarity."""
    d = 0
    temp = series.dropna().copy()
    while adf_check(temp)[1] > 0.05 and d < max_d:
        temp = temp.diff().dropna()
        d += 1
    return d

# =========================================================
# CHECK STATIONARITY PER SERIES
# =========================================================

print("\n" + "="*55)
print("ADF TEST — ORIGINAL SERIES")
print("="*55)

integration_orders = {}

for col in df.columns:
    adf_stat, p_val = adf_check(df[col])
    d = find_integration_order(df[col])
    integration_orders[col] = d
    status = "Already Stationary (I(0))" if d == 0 else f"I({d}) — needs {d} differencing"
    print(f"\n{col}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_val:.4f}")
    print(f"  Integration   : {status}")

print("\nIntegration Orders:", integration_orders)

# =========================================================
# STRATEGY DECISION
# =========================================================
#
# From output:
#   Chemical_only → I(2)
#   Organic_only  → I(0)   ← must NOT be over-differenced
#   Both          → I(2)
#   None          → recovered as 100 - others (not modelled)
#
# Plan:
#   A) VAR on {Chemical_only, Both} after 2nd differencing
#   B) ARIMA on Organic_only independently
#   C) None = 100 - (Chemical_only + Organic_only + Both)
#
# =========================================================

VAR_COLS   = ['Chemical_only', 'Both']      # I(2) series → VAR
ARIMA_COL  = 'Organic_only'                 # I(0) series → ARIMA
VAR_D      = 2                              # differencing for VAR series

# =========================================================
# TRAIN / TEST SPLIT  (80 / 20 on original data)
# =========================================================

TRAIN_RATIO = 0.80
n = len(df)
train_end_idx = int(n * TRAIN_RATIO)        # index in original df

train_orig = df.iloc[:train_end_idx]
test_orig  = df.iloc[train_end_idx:]

print(f"\nTrain observations (original) : {len(train_orig)}")
print(f"Test  observations (original) : {len(test_orig)}")

# =========================================================
# PART A — VAR MODEL FOR Chemical_only AND Both
# =========================================================

print("\n" + "="*55)
print("PART A — VAR MODEL  (Chemical_only  &  Both)")
print("="*55)

# --- Difference the VAR series (train only) ---
var_train_orig = train_orig[VAR_COLS].copy()

var_train_diff = var_train_orig.copy()
for _ in range(VAR_D):
    var_train_diff = var_train_diff.diff()
var_train_diff.dropna(inplace=True)

print(f"\nVAR training rows after {VAR_D} differencing: {len(var_train_diff)}")

# --- Verify stationarity of differenced train series ---
print("\nADF after differencing (train):")
for col in VAR_COLS:
    stat, pv = adf_check(var_train_diff[col])
    status = "Stationary ✓" if pv <= 0.05 else "NOT Stationary ✗"
    print(f"  {col}: p={pv:.4f}  {status}")

# --- Lag selection ---
# Cap maxlags so we keep enough degrees of freedom:
#   rule: maxlags <= (nobs - 1) // (2 * k)  where k = number of series
k = len(VAR_COLS)
safe_maxlags = max(1, (len(var_train_diff) - 1) // (2 * k))
safe_maxlags = min(safe_maxlags, 6)          # hard cap at 6

print(f"\nSafe maxlags for lag selection: {safe_maxlags}")

var_model_selector = VAR(var_train_diff)
lag_result = var_model_selector.select_order(maxlags=safe_maxlags)
print("\nLAG ORDER SELECTION:")
print(lag_result.summary())

optimal_lag = lag_result.aic
if optimal_lag is None or optimal_lag == 0:
    optimal_lag = 1
optimal_lag = min(optimal_lag, safe_maxlags)

print(f"Optimal Lag (AIC): {optimal_lag}")

# --- Fit VAR ---
var_fitted = VAR(var_train_diff).fit(optimal_lag)
print(var_fitted.summary())

is_stable = var_fitted.is_stable()
print(f"\nIs VAR Model Stable? {is_stable}")

if not is_stable:
    print("Reducing lag to 1 for stability...")
    optimal_lag = 1
    var_fitted = VAR(var_train_diff).fit(optimal_lag)
    print(f"Stable with lag=1? {var_fitted.is_stable()}")

# --- Forecast differenced values for test period ---
forecast_input_var = var_train_diff.values[-optimal_lag:]
n_test = len(test_orig)

fc_diff = var_fitted.forecast(y=forecast_input_var, steps=n_test)
fc_diff_df = pd.DataFrame(fc_diff, index=test_orig.index, columns=VAR_COLS)

# --- Inverse 2nd differencing to get back original scale ---
#
# For d=2 we need:
#   last 1st-difference value (from training original data)
#   last original value (from training original data)
#
# Step 1 (undo 2nd diff): cumsum of fc_diff + last 1st-diff value of train
last_first_diff = var_train_orig.diff().dropna().iloc[-1]   # last 1st-diff in train
step1 = fc_diff_df.cumsum() + last_first_diff

# Step 2 (undo 1st diff): cumsum of step1 + last original value of train
last_original_var = var_train_orig.iloc[-1]
fc_var_original = step1.cumsum() + last_original_var

print("\nVAR forecast (original scale) — first few rows:")
print(fc_var_original.head())

# =========================================================
# PART B — ARIMA MODEL FOR Organic_only
# =========================================================

print("\n" + "="*55)
print(f"PART B — ARIMA MODEL  ({ARIMA_COL})")
print("="*55)

organic_train = train_orig[ARIMA_COL]
organic_test  = test_orig[ARIMA_COL]

# --- Auto-select best ARIMA (p, d, q) by AIC over a small grid ---
# Organic_only is I(0), so d=0 or d=1 at most

best_aic   = np.inf
best_order = (1, 0, 1)

for p, d, q in itertools.product(range(4), range(2), range(4)):
    try:
        m = ARIMA(organic_train, order=(p, d, q)).fit()
        if m.aic < best_aic:
            best_aic   = m.aic
            best_order = (p, d, q)
    except Exception:
        continue

print(f"Best ARIMA order for {ARIMA_COL}: {best_order}  (AIC={best_aic:.2f})")

arima_fitted = ARIMA(organic_train, order=best_order).fit()
print(arima_fitted.summary())

fc_organic = arima_fitted.forecast(steps=n_test)
fc_organic = pd.Series(fc_organic.values, index=test_orig.index, name=ARIMA_COL)

print("\nARIMA forecast (original scale):")
print(fc_organic)

# =========================================================
# COMBINE FORECASTS
# =========================================================

forecast_combined = fc_var_original.copy()
forecast_combined[ARIMA_COL] = fc_organic.values

# Recover None
forecast_combined['None'] = 100 - forecast_combined.sum(axis=1)

# Clip and renormalize
for col in forecast_combined.columns:
    forecast_combined[col] = forecast_combined[col].clip(lower=0, upper=100)

row_sums = forecast_combined.sum(axis=1)
for col in forecast_combined.columns:
    forecast_combined[col] = (forecast_combined[col] / row_sums) * 100

print("\nForecast Row Sum Check (should be 100):")
print(forecast_combined.sum(axis=1))

# =========================================================
# ALIGN ACTUAL VS FORECAST
# =========================================================

actual = test_orig.copy()
forecast_combined = forecast_combined.loc[actual.index]

# =========================================================
# MODEL EVALUATION
# =========================================================

print("\n" + "="*55)
print("MODEL EVALUATION")
print("="*55)

results = []

for col in df.columns:
    if col not in forecast_combined.columns:
        continue

    act  = actual[col].values
    pred = forecast_combined[col].values

    mae  = mean_absolute_error(act, pred)
    rmse = np.sqrt(mean_squared_error(act, pred))

    nonzero = act != 0
    if nonzero.sum() > 0:
        mape = np.mean(np.abs((act[nonzero] - pred[nonzero]) / act[nonzero])) * 100
    else:
        mape = np.nan

    results.append({
        'Category': col,
        'MAE':      round(mae, 4),
        'RMSE':     round(rmse, 4),
        'MAPE (%)': round(mape, 2) if not np.isnan(mape) else 'N/A'
    })

    print(f"\n{col}")
    print(f"  MAE      : {mae:.4f}")
    print(f"  RMSE     : {rmse:.4f}")
    if not np.isnan(mape):
        print(f"  MAPE (%) : {mape:.2f}%")
    else:
        print(f"  MAPE (%) : N/A")

results_df = pd.DataFrame(results)
print("\nSummary Table:")
print(results_df.to_string(index=False))

# =========================================================
# ACTUAL VS FORECAST PLOTS
# =========================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(df.columns):
    if col not in forecast_combined.columns:
        continue
    ax = axes[i]
    ax.plot(actual.index, actual[col],
            label='Actual', color='steelblue', marker='o', markersize=4)
    ax.plot(forecast_combined.index, forecast_combined[col],
            label='Forecast', color='tomato', marker='s', markersize=4, linestyle='--')
    ax.set_title(labels_map.get(col, col), fontsize=12, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Percentage (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.suptitle("Fertilizer Application — Actual vs Forecast", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# =========================================================
# FUTURE FORECAST
# =========================================================

print("\n" + "="*55)
print("FUTURE FORECAST")
print("="*55)

FUTURE_STEPS = 10

# --- VAR future forecast ---
# Re-fit VAR on full differenced series (not just train)
var_full_orig = df[VAR_COLS].copy()
var_full_diff = var_full_orig.copy()
for _ in range(VAR_D):
    var_full_diff = var_full_diff.diff()
var_full_diff.dropna(inplace=True)

var_future_fitted = VAR(var_full_diff).fit(optimal_lag)
future_input_var  = var_full_diff.values[-optimal_lag:]

future_diff_raw = var_future_fitted.forecast(y=future_input_var, steps=FUTURE_STEPS)
future_diff_df  = pd.DataFrame(future_diff_raw, columns=VAR_COLS)

# Inverse 2nd differencing on full series
last_first_diff_full = var_full_orig.diff().dropna().iloc[-1]
step1_future = future_diff_df.cumsum() + last_first_diff_full

last_original_full = var_full_orig.iloc[-1]
future_var_original = step1_future.cumsum() + last_original_full

# --- ARIMA future forecast (re-fit on full Organic_only series) ---
arima_full_fitted = ARIMA(df[ARIMA_COL], order=best_order).fit()
future_organic    = arima_full_fitted.forecast(steps=FUTURE_STEPS)

# --- Combine ---
future_combined = future_var_original.copy()
future_combined[ARIMA_COL] = future_organic.values
future_combined['None'] = 100 - future_combined.sum(axis=1)

# Clip and renormalize
for col in future_combined.columns:
    future_combined[col] = future_combined[col].clip(lower=0, upper=100)

row_sums_f = future_combined.sum(axis=1)
for col in future_combined.columns:
    future_combined[col] = (future_combined[col] / row_sums_f) * 100

# Assign future date index
last_date = df.index[-1]
freq = pd.infer_freq(df.index)
if freq is None:
    freq = '6MS'

future_dates = pd.date_range(start=last_date, periods=FUTURE_STEPS + 1, freq=freq)[1:]
future_combined.index = future_dates

print("\nFuture Forecast Values:")
print(future_combined.round(2).to_string())

# =========================================================
# PLOT FUTURE FORECAST
# =========================================================

fig, ax = plt.subplots(figsize=(14, 6))

for col, color in zip(df.columns, colors):
    ax.plot(df.index, df[col],
            color=color, label=f'{labels_map[col]} (Historical)')
    ax.plot(future_combined.index, future_combined[col],
            color=color, linestyle='--', marker='o', markersize=4,
            label=f'{labels_map[col]} (Forecast)')

ax.axvline(x=df.index[-1], color='black', linestyle=':', linewidth=1.5, label='Forecast Start')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.title("Fertilizer Application — Historical and Future Forecast")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# =========================================================
# STACKED AREA PLOT — FUTURE FORECAST
# =========================================================

fig, ax = plt.subplots(figsize=(14, 6))

all_dates = df.index.tolist() + future_combined.index.tolist()

full_series = {}
for col in df.columns:
    hist   = df[col].values
    future = future_combined[col].values
    full_series[col] = np.concatenate([hist, future])

ax.stackplot(
    all_dates,
    [full_series[c] for c in df.columns],
    labels=[labels_map[c] for c in df.columns],
    colors=colors,
    alpha=0.75
)

ax.axvline(x=df.index[-1], color='black', linestyle=':', linewidth=1.5, label='Forecast Start')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.xlabel("Year")
plt.ylabel("Percentage (%)")
plt.title("Fertilizer Application — Stacked Area (Historical + Forecast)")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print("\nDone.") 