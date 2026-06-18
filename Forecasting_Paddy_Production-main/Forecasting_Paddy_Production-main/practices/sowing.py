# =========================================================
# IMPORT LIBRARIES
# =========================================================
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_absolute_error, mean_squared_error

mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False

# =========================================================
# LOAD DATA
# =========================================================
data = pd.read_excel("df.xlsx")
df = data.iloc[27:, :]
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df = df[['Date', 'Broadcasting', 'Transplant_Rows',
         'Transplant_NotRows', 'Parachute', 'Machine', 'Other(sowing)']]
df.set_index('Date', inplace=True)

# =========================================================
# FIX DATE FREQUENCY
# =========================================================
df.index = pd.DatetimeIndex(df.index)
inferred_freq = pd.infer_freq(df.index)

if inferred_freq is not None:
    df.index.freq = inferred_freq
    print(f"Inferred frequency : {inferred_freq}")
else:
    df.index = pd.date_range(
        start=df.index[0],
        periods=len(df),
        freq='YS'
    )
    print("Frequency set to Annual Start (YS).")

print(df.head())
print(df.info())

# =========================================================
# PLOT ORIGINAL DATA
# =========================================================
ax = df.plot(figsize=(12, 6))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.set_ylim(0, 100)
plt.xticks(rotation=45)
plt.xlabel("Year", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)
plt.legend(title="Sowing Method", loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Year-wise Distribution of Sowing Methods", fontsize=14)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# =========================================================
# ADF TEST FUNCTION
# =========================================================
def adf_check(series):
    result = adfuller(series.dropna())
    return result[0], result[1]

# =========================================================
# CHECK ORIGINAL STATIONARITY
# =========================================================
print("\nADF TEST RESULTS (ORIGINAL SERIES)")
diff_count = {}

for column in df.columns:
    adf_stat, p_value = adf_check(df[column])
    print(f"\n{column}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")

    d = 0
    temp_series = df[column].copy()
    while adf_check(temp_series)[1] > 0.05:
        temp_series = temp_series.diff().dropna()
        d += 1
        if d == 3:
            break

    diff_count[column] = d
    print(f"  Differencing Needed : {d}")

# =========================================================
# USE SAME DIFFERENCING ORDER
# =========================================================
max_diff = max(diff_count.values())
print(f"\nMaximum Differencing Order : {max_diff}")

# =========================================================
# DIFFERENCE ALL SERIES
# =========================================================
df_diff = df.copy()
for i in range(max_diff):
    df_diff = df_diff.diff()
df_diff = df_diff.dropna()

print(f"Shape after differencing : {df_diff.shape}")

# =========================================================
# FINAL STATIONARITY CHECK
# =========================================================
print("\nFINAL STATIONARITY RESULTS")
for column in df_diff.columns:
    adf_stat, p_value = adf_check(df_diff[column])
    print(f"\n{column}")
    print(f"  ADF Statistic : {adf_stat:.4f}")
    print(f"  p-value       : {p_value:.4f}")
    print("  Stationary" if p_value <= 0.05 else "  STILL NOT Stationary")

# =========================================================
# TRAIN TEST SPLIT (80 / 20)
# =========================================================
train_size = int(len(df_diff) * 0.8)
train = df_diff.iloc[:train_size]
test  = df_diff.iloc[train_size:]

print(f"\nTrain Shape : {train.shape}")
print(f"Test Shape  : {test.shape}")

# =========================================================
# ADD SMALL JITTER TO AVOID SINGULAR COVARIANCE MATRIX
# This is needed when variables are near-collinear
# (shares sum to ~100%, so one is linearly dependent)
# =========================================================
np.random.seed(42)
jitter = np.random.normal(0, 1e-6, train.shape)
train_jittered = train + jitter

# =========================================================
# SAFE LAG SELECTION
# =========================================================
n_obs        = len(train_jittered)
n_vars       = train_jittered.shape[1]
safe_maxlags = max(1, int((n_obs - 1) / (n_vars + 1)))
print(f"\nTrain observations    : {n_obs}")
print(f"Number of variables   : {n_vars}")
print(f"Safe max lags allowed : {safe_maxlags}")

selected_lag = 1  # default fallback
for try_lag in range(safe_maxlags, 0, -1):
    try:
        model_sel    = VAR(train_jittered)
        lag_results  = model_sel.select_order(maxlags=try_lag)
        selected_lag = lag_results.selected_orders['aic']
        if selected_lag == 0:
            selected_lag = 1
        print(f"Optimal Lag (AIC) : {selected_lag}")
        print(lag_results.summary())
        break
    except Exception as e:
        print(f"  maxlags={try_lag} failed: {e}")
        continue

print(f"\nFinal selected lag : {selected_lag}")

# =========================================================
# FIT VAR MODEL
# =========================================================
var_model  = VAR(train_jittered)
var_result = var_model.fit(selected_lag)

print("\nVAR MODEL SUMMARY")
print(var_result.summary())

# =========================================================
# FORECASTING
# =========================================================
forecast_steps = len(test)
lag_input      = train_jittered.values[-selected_lag:]

forecast_values = var_result.forecast(y=lag_input, steps=forecast_steps)

forecast_df = pd.DataFrame(
    forecast_values,
    index=test.index,
    columns=test.columns
)

print("\nForecasted Differenced Values")
print(forecast_df.head())

# =========================================================
# INVERSE DIFFERENCING (generalized for any max_diff)
# =========================================================
forecast_restored = forecast_df.copy()

for level in range(max_diff):
    ref = df.copy()
    diff_steps_needed = max_diff - 1 - level
    for _ in range(diff_steps_needed):
        ref = ref.diff().dropna()

    ref_train_end = ref.iloc[: train_size + max_diff - 1 - level]
    last_val      = ref_train_end.iloc[-1]

    forecast_restored = forecast_restored.cumsum() + last_val

print("\nForecast Values (Original Scale)")
print(forecast_restored.head())

# =========================================================
# CLIP & NORMALIZE (0-100 percentage constraint)
# =========================================================
forecast_restored = forecast_restored.clip(lower=0, upper=100)

row_sums = forecast_restored.sum(axis=1)
forecast_restored = forecast_restored.div(row_sums, axis=0) * 100

print("\nForecasts After Normalization")
print(forecast_restored.head())

# =========================================================
# ACTUAL TEST VALUES (ORIGINAL SCALE)
# =========================================================
actual_original = df.iloc[train_size + max_diff:]
actual_original = actual_original.loc[forecast_restored.index]

# =========================================================
# EVALUATION METRICS
# =========================================================
print("\nMODEL EVALUATION")
evaluation_results = []

for column in actual_original.columns:
    actual    = actual_original[column]
    predicted = forecast_restored[column]

    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs(
        (actual - predicted) / actual.replace(0, np.nan)
    )) * 100

    evaluation_results.append([column, mae, rmse, mape])

    print(f"\n{column}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%")

evaluation_df = pd.DataFrame(
    evaluation_results,
    columns=['Variable', 'MAE', 'RMSE', 'MAPE']
)
print("\nEvaluation Summary")
print(evaluation_df)

# =========================================================
# INDIVIDUAL ACTUAL VS FORECAST PLOTS
# =========================================================
for column in actual_original.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(actual_original.index, actual_original[column],
             marker='o', label='Actual')
    plt.plot(forecast_restored.index, forecast_restored[column],
             marker='o', label='Forecast', linestyle='--')
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)
    plt.title(f"Actual vs Forecast — {column}", fontsize=14)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# =========================================================
# COMBINED SUBPLOT (3 rows x 2 cols for 6 variables)
# =========================================================
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12))
axes = axes.flatten()

for i, column in enumerate(actual_original.columns):
    axes[i].plot(actual_original.index, actual_original[column],
                 marker='o', label='Actual')
    axes[i].plot(forecast_restored.index, forecast_restored[column],
                 marker='o', label='Forecast', linestyle='--')
    axes[i].set_title(column)
    axes[i].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_ylabel("Percentage (%)")
    axes[i].legend()
    axes[i].grid(axis='y')

for j in range(len(actual_original.columns), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("VAR Model — Actual vs Forecast (Original Scale)", fontsize=14)
plt.tight_layout()
plt.show()

# =========================================================
# SAVE FORECASTS
# =========================================================
forecast_restored.to_excel("VAR_Forecasts.xlsx")
print("\nForecasts saved to VAR_Forecasts.xlsx")
