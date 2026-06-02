# =========================================================
# RANDOM FOREST FORECASTING WITH HYPERPARAMETER TUNING
# =========================================================

# ======================
# 1. IMPORT LIBRARIES
# ======================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ======================
# 2. LOAD DATA
# ======================

data = pd.read_excel("df.xlsx")

df = data.iloc[27:, :]

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

ts = df['Production']

# ======================
# 3. CREATE LAG FEATURES
# ======================

df_rf = pd.DataFrame(ts)

df_rf['lag1'] = df_rf['Production'].shift(1)
df_rf['lag2'] = df_rf['Production'].shift(2)
df_rf['lag3'] = df_rf['Production'].shift(3)

df_rf.dropna(inplace=True)

# ======================
# 4. DEFINE X AND y
# ======================

X = df_rf[['lag1', 'lag2', 'lag3']]
y = df_rf['Production']

# ======================
# 5. TRAIN / TEST SPLIT (TIME ORDER)
# ======================

train_size = int(len(df_rf) * 0.8)

X_train = X.iloc[:train_size]
X_test  = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test  = y.iloc[train_size:]

# ======================
# 6. TIME SERIES CROSS VALIDATION
# ======================

tscv = TimeSeriesSplit(n_splits=5)

# ======================
# 7. MODEL + PARAM GRID
# ======================

rf = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 0.5, None]
}

# ======================
# 8. GRID SEARCH
# ======================

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# ======================
# 9. BEST MODEL
# ======================

best_model = grid_search.best_estimator_

print("\nBest Parameters:")
print(grid_search.best_params_)

# ======================
# 10. PREDICTIONS
# ======================

y_pred = best_model.predict(X_test)

# ======================
# 11. EVALUATION
# ======================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test == 0, 1e-9, y_test))) * 100
r2 = r2_score(y_test, y_pred)
correlation = np.corrcoef(y_test, y_pred)[0,1]

print("\n==============================")
print(" MODEL EVALUATION (TUNED RF)")
print("==============================")

print(f"RMSE        : {rmse:.2f}")
print(f"MAE         : {mae:.2f}")
print(f"MAPE        : {mape:.2f}%")
print(f"R2 Score    : {r2:.4f}")
print(f"Correlation : {correlation:.4f}")

# ======================
# 12. ACTUAL VS PREDICTED
# ======================

plt.figure(figsize=(12,6))

plt.plot(y_test.index, y_test, label='Actual', marker='o')
plt.plot(y_test.index, y_pred, label='Predicted', marker='o', linestyle='--')

plt.title("Actual vs Predicted Production (Tuned RF)")
plt.xlabel("Year")
plt.ylabel("Production")
plt.legend()
plt.grid(True)
plt.show()

# ======================
# 13. FEATURE IMPORTANCE
# ======================

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
})

print("\nFeature Importance:")
print(importance)

importance.sort_values('Importance').plot(
    x='Feature',
    y='Importance',
    kind='barh',
    figsize=(8,5)
)

plt.title("Feature Importance")
plt.grid(True)
plt.show()

# ======================
# 14. FUTURE FORECASTING
# ======================

future_forecast = []

last_values = list(ts.tail(3))

for i in range(5):

    X_future = np.array(last_values[-3:]).reshape(1, -1)

    next_pred = best_model.predict(X_future)[0]

    future_forecast.append(next_pred)

    last_values.append(next_pred)

# ======================
# 15. FUTURE DATES
# ======================

future_dates = pd.date_range(
    start=ts.index[-1] + pd.DateOffset(years=1),
    periods=5,
    freq='YE'
)

forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecast': future_forecast
})

print("\nFuture Forecast:")
print(forecast_df)

# ======================
# 16. FORECAST PLOT
# ======================

plt.figure(figsize=(12,6))

plt.plot(ts.index, ts, label='Historical Data', marker='o')
plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', marker='o', linestyle='--')

plt.title("Random Forest Forecast (Tuned)")
plt.xlabel("Year")
plt.ylabel("Production")
plt.legend()
plt.grid(True)
plt.show()