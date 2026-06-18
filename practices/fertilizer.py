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

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['axes.grid'] = False

# =========================================================
# LOAD DATA
# =========================================================
data = pd.read_excel("df.xlsx")

# Remove unwanted rows
df = data.iloc[27:, :]

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort by date
df = df.sort_values('Date')

# Keep required columns
df = df[['Date',
         'Chemical_only',
         'Organic_only',
         'Both',
         'None']]

# Set Date as index
df.set_index('Date', inplace=True)

print(df.head())
print(df.info())

# =========================================================
# PLOT ORIGINAL DATA
# =========================================================

ax = df.plot(figsize=(12,6))

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax.set_ylim(0,100)

plt.xticks(rotation=45)

plt.xlabel("Year", fontsize=12)
plt.ylabel("Percentage (%)", fontsize=12)

plt.legend([
    "Chemical Fertilizer only",
    "Organic Fertilizer only",
    "Both Chemical and Organic Fertilizer",
    "None"
],
title="Application of Fertilizer",
loc="center left",
bbox_to_anchor=(1, 0.5))

plt.title("Year-wise Distribution of Fertilizer Usage",
          fontsize=14)

plt.grid(axis='y')

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
    print(f"ADF Statistic : {adf_stat:.4f}")
    print(f"p-value       : {p_value:.4f}")

    d = 0
    temp_series = df[column].copy()

    while adf_check(temp_series)[1] > 0.05:

        temp_series = temp_series.diff().dropna()

        d += 1

        if d == 3:
            break

    diff_count[column] = d

    print(f"Differencing Needed : {d}")

# =========================================================
# USE SAME DIFFERENCING ORDER
# =========================================================

max_diff = max(diff_count.values())

print("\nMaximum Differencing Order :", max_diff)

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
    print(f"ADF Statistic : {adf_stat:.4f}")
    print(f"p-value       : {p_value:.4f}")

    if p_value <= 0.05:
        print("Series is Stationary")

    else:
        print("Series is STILL NOT Stationary")

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

train_size = int(len(df_diff) * 0.8)

train = df_diff.iloc[:train_size]
test = df_diff.iloc[train_size:]

print("\nTrain Shape :", train.shape)
print("Test Shape  :", test.shape)

# =========================================================
# LAG SELECTION
# =========================================================

model = VAR(train)

lag_results = model.select_order(maxlags=8)

print("\nLag Order Selection")
print(lag_results.summary())

# Select lag using AIC
selected_lag = lag_results.aic

print("\nSelected Lag :", selected_lag)

# =========================================================
# FIT VAR MODEL
# =========================================================

var_model = model.fit(selected_lag)

print("\nVAR MODEL SUMMARY")
print(var_model.summary())

# =========================================================
# FORECASTING
# =========================================================

forecast_steps = len(test)

forecast_values = var_model.forecast(
    y=train.values[-selected_lag:],
    steps=forecast_steps
)

# Convert forecast to dataframe
forecast_df = pd.DataFrame(
    forecast_values,
    index=test.index,
    columns=test.columns
)

print("\nForecasted Differenced Values")
print(forecast_df.head())

# =========================================================
# CONVERT BACK TO ORIGINAL SCALE
# =========================================================

forecast_original = forecast_df.copy()

# Last original observation before test set
last_original = df.iloc[train_size]

for column in forecast_original.columns:

    forecast_original[column] = (
        forecast_original[column].cumsum()
        + last_original[column]
    )

# =========================================================
# HANDLE NEGATIVE VALUES
# =========================================================

# Keep values between 0 and 100
forecast_original = forecast_original.clip(
    lower=0,
    upper=100
)

# Normalize rows so total = 100
forecast_original = forecast_original.div(
    forecast_original.sum(axis=1),
    axis=0
) * 100

print("\nForecasts After Normalization")
print(forecast_original.head())

# =========================================================
# ACTUAL VALUES
# =========================================================

actual_original = df.iloc[train_size + max_diff:]

actual_original = actual_original.loc[
    forecast_original.index
]

# =========================================================
# EVALUATION METRICS
# =========================================================

print("\nMODEL EVALUATION")

evaluation_results = []

for column in actual_original.columns:

    actual = actual_original[column]
    predicted = forecast_original[column]

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    evaluation_results.append([
        column,
        mae,
        rmse,
        mape
    ])

    print(f"\n{column}")

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")

# =========================================================
# EVALUATION TABLE
# =========================================================

evaluation_df = pd.DataFrame(
    evaluation_results,
    columns=['Variable', 'MAE', 'RMSE', 'MAPE']
)

print("\nEvaluation Summary")
print(evaluation_df)

# =========================================================
# ACTUAL VS FORECAST PLOTS
# =========================================================

for column in actual_original.columns:

    plt.figure(figsize=(12,6))

    plt.plot(
        actual_original.index,
        actual_original[column],
        marker='o',
        label='Actual'
    )

    plt.plot(
        forecast_original.index,
        forecast_original[column],
        marker='o',
        label='Forecast'
    )

    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage (%)", fontsize=12)

    plt.title(
        f"Actual vs Forecast - {column}",
        fontsize=14
    )

    plt.legend()

    plt.grid(axis='y')

    plt.show()

# =========================================================
# COMBINED PLOT
# =========================================================

fig, axes = plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(14,10)
)

axes = axes.flatten()

for i, column in enumerate(actual_original.columns):

    axes[i].plot(
        actual_original.index,
        actual_original[column],
        marker='o',
        label='Actual'
    )

    axes[i].plot(
        forecast_original.index,
        forecast_original[column],
        marker='o',
        label='Forecast'
    )

    axes[i].set_title(column)

    axes[i].xaxis.set_major_locator(
        mdates.YearLocator(2)
    )

    axes[i].xaxis.set_major_formatter(
        mdates.DateFormatter('%Y')
    )

    axes[i].tick_params(
        axis='x',
        rotation=45
    )

    axes[i].set_ylabel("Percentage (%)")

    axes[i].legend()

    axes[i].grid(axis='y')

plt.tight_layout()

plt.show()

# =========================================================
# SAVE FORECASTS
# =========================================================

forecast_original.to_excel(
    "VAR_Forecasts.xlsx"
)

print("\nForecasts saved successfully!")