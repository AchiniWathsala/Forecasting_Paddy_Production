# =========================================================
# IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_excel("df.xlsx")

df = data.iloc[27:, :]

df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values('Date')

df = df[['Date',
         'Chemical_only',
         'Organic_only',
         'Both',
         'None']]

df.set_index('Date', inplace=True)

# =========================================================
# CONVERT TO PROPORTIONS
# =========================================================

df_prop = df / 100

# =========================================================
# ADD SMALL VALUE TO AVOID log(0)
# =========================================================

epsilon = 1e-6

df_prop = df_prop + epsilon

# =========================================================
# LOG-RATIO TRANSFORMATION
# Reference category = None
# =========================================================

df_logratio = pd.DataFrame(index=df.index)

df_logratio['Chemical_only'] = np.log(
    df_prop['Chemical_only'] / df_prop['None']
)

df_logratio['Organic_only'] = np.log(
    df_prop['Organic_only'] / df_prop['None']
)

df_logratio['Both'] = np.log(
    df_prop['Both'] / df_prop['None']
)

print(df_logratio.head())

# =========================================================
# ADF TEST FUNCTION
# =========================================================

def adf_check(series):

    result = adfuller(series.dropna())

    return result[1]

# =========================================================
# DIFFERENCING
# =========================================================

diff_count = {}

for col in df_logratio.columns:

    d = 0

    temp = df_logratio[col]

    while adf_check(temp) > 0.05:

        temp = temp.diff().dropna()

        d += 1

        if d == 3:
            break

    diff_count[col] = d

max_diff = max(diff_count.values())

# =========================================================
# DIFFERENCE DATA
# =========================================================

df_diff = df_logratio.copy()

for i in range(max_diff):

    df_diff = df_diff.diff()

df_diff = df_diff.dropna()

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

train_size = int(len(df_diff) * 0.8)

train = df_diff.iloc[:train_size]

test = df_diff.iloc[train_size:]

# =========================================================
# LAG SELECTION
# =========================================================

model = VAR(train)

lag_results = model.select_order(maxlags=5)

print(lag_results.summary())

selected_lag = lag_results.aic

print("\nSelected Lag:", selected_lag)

# =========================================================
# FIT VAR MODEL
# =========================================================

var_model = model.fit(selected_lag)

print(var_model.summary())

# =========================================================
# FORECAST
# =========================================================

forecast_steps = len(test)

forecast_values = var_model.forecast(
    y=train.values[-selected_lag:],
    steps=forecast_steps
)

forecast_df = pd.DataFrame(
    forecast_values,
    index=test.index,
    columns=test.columns
)

# =========================================================
# INVERSE DIFFERENCING
# =========================================================

forecast_logratio = forecast_df.copy()

last_values = df_logratio.iloc[train_size]

for col in forecast_logratio.columns:

    forecast_logratio[col] = (
        forecast_logratio[col].cumsum()
        + last_values[col]
    )

# =========================================================
# CONVERT LOG-RATIO BACK TO PERCENTAGES
# =========================================================

exp_chem = np.exp(forecast_logratio['Chemical_only'])

exp_org = np.exp(forecast_logratio['Organic_only'])

exp_both = np.exp(forecast_logratio['Both'])

denominator = (
    1
    + exp_chem
    + exp_org
    + exp_both
)

forecast_percent = pd.DataFrame(index=forecast_logratio.index)

forecast_percent['None'] = 1 / denominator

forecast_percent['Chemical_only'] = (
    exp_chem / denominator
)

forecast_percent['Organic_only'] = (
    exp_org / denominator
)

forecast_percent['Both'] = (
    exp_both / denominator
)

# Convert to percentages
forecast_percent = forecast_percent * 100

# =========================================================
# REORDER COLUMNS
# =========================================================

forecast_percent = forecast_percent[
    ['Chemical_only',
     'Organic_only',
     'Both',
     'None']
]

print("\nForecast Percentages")
print(forecast_percent.head())

# =========================================================
# ACTUAL VALUES
# =========================================================

actual = df.iloc[train_size + max_diff:]

actual = actual.loc[forecast_percent.index]

# =========================================================
# EVALUATION
# =========================================================

print("\nMODEL EVALUATION")

for col in actual.columns:

    mae = mean_absolute_error(
        actual[col],
        forecast_percent[col]
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual[col],
            forecast_percent[col]
        )
    )

    mape = np.mean(
        np.abs(
            (actual[col] - forecast_percent[col])
            / actual[col]
        )
    ) * 100

    print(f"\n{col}")

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")

# =========================================================
# PLOTS
# =========================================================

for col in actual.columns:

    plt.figure(figsize=(12,6))

    plt.plot(
        actual.index,
        actual[col],
        marker='o',
        label='Actual'
    )

    plt.plot(
        forecast_percent.index,
        forecast_percent[col],
        marker='o',
        label='Forecast'
    )

    plt.title(f"Actual vs Forecast - {col}")

    plt.xlabel("Year")

    plt.ylabel("Percentage (%)")

    plt.legend()

    plt.grid(axis='y')

    plt.show()