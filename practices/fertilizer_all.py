# =========================================================
# IMPORT LIBRARIES
# =========================================================
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from statsmodels.tsa.stattools import adfuller, grangercausalitytests
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
df = df[['Date', 'Chemical_only', 'Organic_only', 'Both', 'None']]
df.set_index('Date', inplace=True)

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
plt.legend([
    "Chemical Fertilizer only",
    "Organic Fertilizer only",
    "Both Chemical and Organic Fertilizer",
    "None"
], title="Application of Fertilizer", loc="center left", bbox_to_anchor=(1, 0.5))
plt.title("Year-wise Distribution of Fertilizer Usage", fontsize=14)
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
# ADD JITTER TO AVOID SINGULAR COVARIANCE MATRIX
# =========================================================
np.random.seed(42)
train_jittered = train + np.random.normal(0, 1e-6, train.shape)

# =========================================================
# SAFE LAG SELECTION
# =========================================================
n_obs        = len(train_jittered)
n_vars       = train_jittered.shape[1]
safe_maxlags = max(1, int((n_obs - 1) / (n_vars + 1)))
print(f"\nTrain observations    : {n_obs}")
print(f"Number of variables   : {n_vars}")
print(f"Safe max lags allowed : {safe_maxlags}")

selected_lag = 1
for try_lag in range(safe_maxlags, 0, -1):
    try:
        lag_result   = VAR(train_jittered).select_order(maxlags=try_lag)
        selected_lag = lag_result.selected_orders['aic']
        selected_lag = max(1, selected_lag)
        print(f"\nLag Order Selection:")
        print(lag_result.summary())
        print(f"Selected Lag (AIC) : {selected_lag}")
        break
    except Exception as e:
        print(f"  maxlags={try_lag} failed, trying lower...")
        continue

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
forecast_steps  = len(test)
lag_input       = train_jittered.values[-selected_lag:]

forecast_values = var_result.forecast(y=lag_input, steps=forecast_steps)
forecast_df     = pd.DataFrame(
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
    for _ in range(max_diff - 1 - level):
        ref = ref.diff().dropna()
    last_val = ref.iloc[: train_size + max_diff - 1 - level].iloc[-1]
    forecast_restored = forecast_restored.cumsum() + last_val

# =========================================================
# CLIP & NORMALIZE
# =========================================================
forecast_restored = forecast_restored.clip(lower=0, upper=100)
forecast_restored = forecast_restored.div(
    forecast_restored.sum(axis=1), axis=0
) * 100

print("\nForecasts After Normalization")
print(forecast_restored.head())

# =========================================================
# ACTUAL TEST VALUES
# =========================================================
actual_original  = df.iloc[train_size + max_diff:]
actual_original  = actual_original.loc[forecast_restored.index]

# =========================================================
# 1. FORECASTING ACCURACY — MAE, RMSE, NRMSE
# =========================================================
print("\n" + "="*60)
print("1. FORECASTING ACCURACY")
print("="*60)

evaluation_results = []

for column in actual_original.columns:
    actual    = actual_original[column]
    predicted = forecast_restored[column]

    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    # NRMSE — normalized by range
    value_range = actual.max() - actual.min()
    nrmse = (rmse / value_range * 100) if value_range != 0 else np.nan

    # MAPE — skip zeros
    nonzero = actual != 0
    mape = np.mean(np.abs(
        (actual[nonzero] - predicted[nonzero]) / actual[nonzero]
    )) * 100 if nonzero.sum() > 0 else np.nan

    evaluation_results.append([column, mae, rmse, nrmse, mape])

    print(f"\n{column}")
    print(f"  MAE   : {mae:.4f}  (percentage points)")
    print(f"  RMSE  : {rmse:.4f}  (percentage points)")
    print(f"  NRMSE : {nrmse:.2f}%")
    print(f"  MAPE  : {mape:.2f}%" if not np.isnan(mape) else "  MAPE  : N/A")

evaluation_df = pd.DataFrame(
    evaluation_results,
    columns=['Variable', 'MAE', 'RMSE', 'NRMSE(%)', 'MAPE(%)']
)
print("\nAccuracy Summary Table:")
print(evaluation_df.to_string(index=False))

# =========================================================
# ACTUAL VS FORECAST PLOTS
# =========================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
axes = axes.flatten()

for i, column in enumerate(actual_original.columns):
    axes[i].plot(actual_original.index, actual_original[column],
                 marker='o', label='Actual', color='steelblue')
    axes[i].plot(forecast_restored.index, forecast_restored[column],
                 marker='s', label='Forecast', color='tomato', linestyle='--')
    axes[i].set_title(column, fontsize=12)
    axes[i].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].set_ylabel("Percentage (%)")
    axes[i].legend()
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle("Actual vs Forecast — Original Scale", fontsize=14)
plt.tight_layout()
plt.show()

# =========================================================
# 2. GRANGER CAUSALITY — FULL MATRIX
# =========================================================
print("\n" + "="*60)
print("2. GRANGER CAUSALITY ANALYSIS")
print("="*60)
print(f"Using lag = {selected_lag} (same as VAR model)\n")

columns      = train_jittered.columns.tolist()
gc_results   = []

for caused in columns:
    for cause in columns:
        if cause == caused:
            continue
        try:
            test_data = train_jittered[[caused, cause]]
            gc_test   = grangercausalitytests(
                test_data,
                maxlag=selected_lag,
                verbose=False
            )
            p_value = gc_test[selected_lag][0]['ssr_ftest'][1]
            f_stat  = gc_test[selected_lag][0]['ssr_ftest'][0]

            if p_value <= 0.01:
                sig = "***"
            elif p_value <= 0.05:
                sig = "**"
            elif p_value <= 0.10:
                sig = "*"
            else:
                sig = "ns"

            gc_results.append({
                'Cause':        cause,
                'Effect':       caused,
                'F-statistic':  round(f_stat, 4),
                'p-value':      round(p_value, 4),
                'Significance': sig
            })

            print(f"{cause:20s} → {caused:20s} | "
                  f"F={f_stat:.4f}  p={p_value:.4f}  {sig}")

        except Exception as e:
            print(f"  Skipped {cause} → {caused} : {e}")

gc_df = pd.DataFrame(gc_results)

print("\nGranger Causality Summary:")
print(gc_df.to_string(index=False))

# --- Heatmap ---
pivot_p = gc_df.pivot(index='Effect', columns='Cause', values='p-value')

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    pivot_p,
    annot=True,
    fmt='.3f',
    cmap='RdYlGn_r',
    vmin=0,
    vmax=0.10,
    linewidths=0.5,
    ax=ax,
    cbar_kws={'label': 'p-value'}
)
ax.set_title(
    "Granger Causality p-values\n"
    "(*** p<0.01  ** p<0.05  * p<0.10  ns=not significant)",
    fontsize=12
)
ax.set_xlabel("Cause →")
ax.set_ylabel("→ Effect")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# =========================================================
# 3. IMPULSE RESPONSE FUNCTIONS (IRF)
# =========================================================
print("\n" + "="*60)
print("3. IMPULSE RESPONSE FUNCTIONS (IRF)")
print("="*60)

irf_periods = 10

irf = var_result.irf(periods=irf_periods)

print("IRF plot: shows how a shock in one variable")
print("affects all other variables over time.\n")

# Plot IRF
irf.plot(
    orth=True,
    figsize=(14, 10)
)
plt.suptitle(
    "Orthogonalized Impulse Response Functions\n"
    "(Response to One Standard Deviation Shock)",
    fontsize=13
)
plt.tight_layout()
plt.show()

# Plot IRF with confidence intervals
irf.plot(
    orth=True,
    stderr_type='mc',
    repl=100,
    figsize=(14, 10)
)
plt.suptitle(
    "IRF with 95% Confidence Intervals",
    fontsize=13
)
plt.tight_layout()
plt.show()

# Cumulative IRF
irf.plot_cum_effects(
    orth=True,
    figsize=(14, 10)
)
plt.suptitle(
    "Cumulative Impulse Response Functions",
    fontsize=13
)
plt.tight_layout()
plt.show()

# =========================================================
# 4. FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)
# =========================================================
print("\n" + "="*60)
print("4. FORECAST ERROR VARIANCE DECOMPOSITION (FEVD)")
print("="*60)

fevd_periods = 10

fevd = var_result.fevd(periods=fevd_periods)

print("FEVD: shows what % of each variable's forecast")
print("error variance is explained by each variable.\n")

# Print FEVD tables
fevd_results = fevd.decomp  # shape: (periods, n_vars, n_vars)

for i, col in enumerate(df.columns):
    print(f"\nFEVD for {col}:")
    fevd_df = pd.DataFrame(
        fevd_results[:, i, :],
        columns=df.columns
    )
    fevd_df.index = fevd_df.index + 1
    fevd_df.index.name = 'Period'
    print(fevd_df.round(4).to_string())

# --- FEVD Plot (built-in) ---
fevd.plot(figsize=(14, 10))
plt.suptitle(
    "Forecast Error Variance Decomposition (FEVD)",
    fontsize=13
)
plt.tight_layout()
plt.show()

# --- Custom FEVD stacked bar at period 10 ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

colors_fevd = ['steelblue', 'seagreen', 'darkorange', 'tomato']

for i, col in enumerate(df.columns):
    fevd_at_10 = fevd_results[fevd_periods - 1, i, :]
    axes[i].bar(
        df.columns,
        fevd_at_10 * 100,
        color=colors_fevd,
        edgecolor='white',
        linewidth=0.5
    )
    axes[i].set_title(f"FEVD of {col} at Period {fevd_periods}", fontsize=11)
    axes[i].set_ylabel("Variance Explained (%)")
    axes[i].set_ylim(0, 100)
    axes[i].tick_params(axis='x', rotation=45)
    for j, val in enumerate(fevd_at_10 * 100):
        axes[i].text(j, val + 1, f"{val:.1f}%",
                     ha='center', fontsize=9)

plt.suptitle(
    f"Variance Decomposition at Period {fevd_periods}\n"
    "(How much each variable explains of the forecast error)",
    fontsize=13
)
plt.tight_layout()
plt.show()

# =========================================================
# SAVE ALL RESULTS
# =========================================================
forecast_restored.to_excel("VAR_Forecasts.xlsx")

with pd.ExcelWriter("VAR_Full_Analysis.xlsx") as writer:
    evaluation_df.to_excel(writer, sheet_name='Accuracy',          index=False)
    gc_df.to_excel(        writer, sheet_name='Granger_Causality', index=False)

    for i, col in enumerate(df.columns):
        fevd_df = pd.DataFrame(
            fevd_results[:, i, :],
            columns=df.columns
        )
        fevd_df.index = fevd_df.index + 1
        fevd_df.index.name = 'Period'
        fevd_df.to_excel(writer, sheet_name=f'FEVD_{col[:10]}')

print("\nAll results saved to VAR_Full_Analysis.xlsx")
print("Forecasts saved to VAR_Forecasts.xlsx")
print("\nDone.")