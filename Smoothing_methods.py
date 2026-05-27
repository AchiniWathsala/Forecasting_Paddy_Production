# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

#load dataset
data = pd.read_excel("df.xlsx")
df = data.iloc[27:,:] # from 1992-2025


df.set_index('Date', inplace=True)
df = df.sort_values('Date')

# Optional but recommended
#df = df.asfreq('6MS')

ts = df['Production']

# train / test split
train_size = int(len(ts) * 0.8)

train = ts.iloc[:train_size]
test = ts.iloc[train_size:]

# Additive
model_add = ExponentialSmoothing(
    train,
    trend='add',
    seasonal='add',
    seasonal_periods=2
).fit()

forecast_add = model_add.forecast(len(test))

rmse_add = np.sqrt(mean_squared_error(test, forecast_add))
mape_add = mean_absolute_percentage_error(test, forecast_add) * 100

# Multiplicative
model_mul = ExponentialSmoothing(
    train,
    trend='mul',
    seasonal='mul',
    seasonal_periods=2
).fit()

forecast_mul = model_mul.forecast(len(test))

rmse_mul = np.sqrt(mean_squared_error(test, forecast_mul))
mape_mul = mean_absolute_percentage_error(test, forecast_mul) * 100

print("Additive Model")
print("RMSE:", rmse_add)
print("MAPE:", mape_add)

print("\nMultiplicative Model")
print("RMSE:", rmse_mul)
print("MAPE:", mape_mul)


# ---------------------------------------------------
# Select Best Model
# ---------------------------------------------------

if rmse_add < rmse_mul:
    print("\nAdditive model performs better.")
else:
    print("\nMultiplicative model performs better.")

# ---------------------------------------------------
# Plot Forecasts
# ---------------------------------------------------

plt.figure(figsize=(12,6))

plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Actual')

# plt.plot(
#     test.index,
#     forecast_add,
#     label='Additive Forecast'
# )

# plt.plot(
#     test.index,
#     forecast_mul,
#     label='Multiplicative Forecast'
# )

# plt.title("Paddy Production Forecast")
# plt.xlabel("Date")
# plt.ylabel("Production")

# plt.legend()
# plt.grid(True)

# plt.show()


##########################################################

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

# ---------------------------------------------------
# Residuals
# ---------------------------------------------------

residuals = model_mul.resid

# ---------------------------------------------------
# Plot Residuals
# ---------------------------------------------------

plt.figure(figsize=(12,5))
plt.plot(residuals)
plt.title("Residuals")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()

# ---------------------------------------------------
# Histogram of Residuals
# ---------------------------------------------------

plt.figure(figsize=(8,5))
plt.hist(residuals, bins=10)
plt.title("Histogram of Residuals")
plt.show()

# ---------------------------------------------------
# ACF Plot
# ---------------------------------------------------

plot_acf(residuals, lags=10)
plt.show()

# ---------------------------------------------------
# Q-Q Plot
# ---------------------------------------------------

stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()

# ---------------------------------------------------
# Ljung-Box Test
# ---------------------------------------------------

lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)

print(lb_test)

print(model_mul.summary())
