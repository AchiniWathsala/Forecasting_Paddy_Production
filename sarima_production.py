# pip install pmdarima

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

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


# Automatically Find Best SARIMA Model
auto_model = auto_arima(
    train,

    seasonal=True,
    m=2,                 # Yala + Maha cycle

    start_p=0,
    start_q=0,
    max_p=3,
    max_q=3,
    

    start_P=0,
    start_Q=0,
    max_P=2,
    max_Q=2,

    d=None,
    D=None,

    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(auto_model.summary())

#Extract Best Parameters
print("Best Model Order:", auto_model.order)

print("Best Seasonal Order:", auto_model.seasonal_order)

#Fit Final SARIMA Model
model = SARIMAX(
    train,
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order
)

model_fit = model.fit()

print(model_fit.summary())

#Forecast
forecast = model_fit.forecast(steps=len(test))

# forecast = pd.Series(
#     model_fit.forecast(steps=len(test)),
#     index=test.index
# )

#Evaluate Model
rmse = np.sqrt(mean_squared_error(test, forecast))

print("RMSE:", rmse)

#Plot Forecast
plt.figure(figsize=(12,5))

plt.plot(train.index, train, label='Train')

plt.plot(test.index, test, label='Actual')

plt.plot(test.index, forecast, label='Forecast')

plt.legend()

plt.title("Best SARIMA Forecast")

plt.show()