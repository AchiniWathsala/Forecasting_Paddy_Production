# =========================================================
# RANDOM FOREST FORECASTING FOR PADDY PRODUCTION
# WITH COMPLETE MODEL EVALUATION
# =========================================================

# ======================
# 1. IMPORT LIBRARIES
# ======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ======================
# 2. LOAD DATA
# ======================

data = pd.read_excel("df.xlsx")

# Remove unwanted rows
df = data.iloc[27:, :]

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Sort values by date
df = df.sort_values('Date')

# Set Date as index
df.set_index('Date', inplace=True)

# Target variable
ts = df['Production']


# ======================
# 3. CREATE LAG FEATURES
# ======================

df_rf = pd.DataFrame(ts)

# Create lag variables
df_rf['lag1'] = df_rf['Production'].shift(1)
df_rf['lag2'] = df_rf['Production'].shift(2)
df_rf['lag3'] = df_rf['Production'].shift(3)

# Remove missing values
df_rf.dropna(inplace=True)


# ======================
# 4. DEFINE X AND y
# ======================

X = df_rf[['lag1', 'lag2', 'lag3']]

y = df_rf['Production']


# ======================
# 5. TRAIN / TEST SPLIT
# ======================

train_size = int(len(df_rf) * 0.8)

X_train = X.iloc[:train_size]
X_test  = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test  = y.iloc[train_size:]


# ======================
# 6. BUILD RANDOM FOREST MODEL
# ======================

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)


# ======================
# 7. PREDICTIONS
# ======================

y_pred = rf_model.predict(X_test)


# ======================
# 8. MODEL EVALUATION
# ======================

# RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# MAE
mae = mean_absolute_error(y_test, y_pred)

# MAPE
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

# R² Score
r2 = r2_score(y_test, y_pred)

# Correlation
correlation = np.corrcoef(y_test, y_pred)[0,1]

# Print evaluation metrics
print("===================================")
print(" RANDOM FOREST MODEL EVALUATION")
print("===================================")

print(f"RMSE        : {rmse:.2f}")
print(f"MAE         : {mae:.2f}")
print(f"MAPE        : {mape:.2f}%")
print(f"R² Score    : {r2:.4f}")
print(f"Correlation : {correlation:.4f}")


# ======================
# 9. ACTUAL VS PREDICTED PLOT
# ======================

plt.figure(figsize=(12,6))

plt.plot(
    y_train.index,
    y_train,
    label='Train Data',
    marker='o',
    linewidth=2
)

plt.plot(
    y_test.index,
    y_test,
    label='Test Data',
    marker='o',
    linewidth=2
)

plt.plot(
    y_test.index,
    y_pred,
    label='Predicted Data',
    marker='o',
    linestyle='--',
    linewidth=2
)

plt.xlabel("Year")
plt.ylabel("Production")

plt.title("Actual vs Predicted Production")

plt.legend()

plt.grid(True)

plt.show()


# ======================
# 10. RESIDUAL ANALYSIS
# ======================

residuals = y_test - y_pred

plt.figure(figsize=(10,5))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, color='red', linestyle='--')

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.title("Residual Plot")

plt.grid(True)

plt.show()


# ======================
# 11. FEATURE IMPORTANCE
# ======================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

print("\nFeature Importance")
print(importance)


# ======================
# 12. FEATURE IMPORTANCE PLOT
# ======================

importance.sort_values('Importance').plot(
    x='Feature',
    y='Importance',
    kind='barh',
    figsize=(8,5)
)

plt.xlabel("Importance")

plt.title("Feature Importance")

plt.grid(True)

plt.show()


# ======================
# 13. FUTURE FORECASTING
# ======================

future_forecast = []

# Get last 3 observations
last_values = list(ts.tail(3))

# Forecast next 5 years
for i in range(5):

    # Prepare input
    X_future = np.array(last_values[-3:]).reshape(1, -1)

    # Predict next value
    next_pred = rf_model.predict(X_future)[0]

    # Store prediction
    future_forecast.append(next_pred)

    # Update lag values
    last_values.append(next_pred)


# ======================
# 14. CREATE FUTURE DATES
# ======================

future_dates = pd.date_range(
    start=ts.index[-1] + pd.DateOffset(years=1),
    periods=5,
    freq='Y'
)


# ======================
# 15. FORECAST DATAFRAME
# ======================

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast': future_forecast
})

print("\nFuture Forecast")
print(forecast_df)


# ======================
# 16. PLOT FORECAST
# ======================

plt.figure(figsize=(12,6))

# Historical data
plt.plot(
    ts.index,
    ts,
    label='Historical Data',
    marker='o',
    linewidth=2
)

# Forecast values
plt.plot(
    forecast_df['Date'],
    forecast_df['Forecast'],
    label='Forecast',
    marker='o',
    linestyle='--',
    linewidth=2
)

plt.xlabel("Year")
plt.ylabel("Production")

plt.title("Random Forest Future Forecast")

plt.legend()

plt.grid(True)

plt.show()