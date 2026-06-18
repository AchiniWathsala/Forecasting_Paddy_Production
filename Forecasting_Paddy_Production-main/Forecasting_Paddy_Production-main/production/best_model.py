# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error,mean_absolute_error, r2_score

#load dataset
data = pd.read_excel(r"../df.xlsx")
df = data.iloc[27:,:] # from 1992-2025
df.head()

print(df.info())

print(df.shape)

df = df.sort_values('Date')

df.set_index('Date', inplace=True)
df.head(3)

ts = df["Production"]
ts.tail()

plt.figure(figsize=(12,5))
plt.plot(ts,marker="o",linewidth=1.5)

# plt.title("Paddy Production Time Series")
plt.xlabel("Year",fontsize=14,fontweight='bold',fontname='Times New Roman')
plt.ylabel("Production ('000 MT)",fontsize=14,fontweight='bold',fontname='Times New Roman')
plt.grid(axis='y')
plt.xticks(rotation=45)
plt.ylim(500,)
plt.xticks(fontsize=13, fontname='Times New Roman')
plt.yticks(fontsize=13, fontname='Times New Roman')

plt.show()

# train / test split
train_size = int(len(ts) * 0.8)

train = ts.iloc[:train_size]
test = ts.iloc[train_size:]

print("Train size:", len(train))
print("Test size:", len(test))

from statsmodels.tsa.stattools import adfuller, kpss
def adf_test(series, label="Series"):
    result = adfuller(series, autolag="AIC")
    p = result[1]
    print(f"\n  ADF Test [{label}]")
    print(f"    Statistic : {result[0]:.4f}")
    print(f"    p-value   : {p:.4f}")
    # print(f"    Critical 5%: {result[4]['5%']:.4f}")
    print(f"    → {'✅ STATIONARY' if p < 0.05 else '❌ NON-STATIONARY — differencing needed'}")
    return p

adf_test(ts, "Original")

# Automatically Find Best SARIMA Model
auto_model = auto_arima(train, 
                   seasonal=True,
                   test='adf', 
                   m=2, 
                   stepwise=True, 
                   suppress_warnings=True,
                   trace=True)

print(auto_model)


# Generate a summary of the best model found
print(auto_model.summary())

print(auto_model.fit(train))

forecast = auto_model.predict(n_periods=len(test))
forecast

dates = test.index
forecast_df = pd.DataFrame(index=dates)

forecast_df['Prediction'] = forecast.values.round(2)

# Season
forecast_df['Season'] = np.where(
    dates.month == 4,
    'Yala',
    'Maha'
)

# Agricultural year
forecast_df['Year'] = np.where(
    dates.month == 4,          # Yala starts in April
    dates.year,
    dates.year + 1            # Maha starts in September, assigned to next year
)

forecast_df = forecast_df[['Prediction', 'Year', 'Season']]

forecast_df

#Plot Forecast
plt.figure(figsize=(12,6))


plt.plot(train.index, train, label="Train Data",marker="o",linewidth=1.5)
plt.plot(test.index, test, label="Test Data", color="green",marker="o",linewidth=1.5)
plt.plot(test.index, forecast, label="Predicted Data", color="red",marker="o",linewidth=1.5, linestyle='--')

plt.legend(prop={'size': 12, 'family':'Times New Roman'}  )

plt.title("Best SARIMA Forecast",pad=20)

plt.ylim(500,)


plt.xlabel("Year",fontsize=14,fontweight='bold',fontname='Times New Roman')
plt.ylabel("Production ('000 MT)",fontsize=14,fontweight='bold',fontname='Times New Roman')

plt.grid(axis='y')
plt.xticks(rotation=45)

plt.xticks(fontsize=13, fontname='Times New Roman')
plt.yticks(fontsize=13, fontname='Times New Roman')

plt.show()

# Diagnostic plots
auto_model.plot_diagnostics(figsize=(12,8))
plt.show()

#Evaluate Model
rmse = np.sqrt(mean_squared_error(test, forecast))
mape = mean_absolute_percentage_error(test, forecast) * 100
mae = mean_absolute_error(test, forecast)
r2 = r2_score(test, forecast)

print(f"RMSE : {rmse:.2f}")
print(f"MAPE : {mape:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R²   : {r2:.4f}")


# ── Refit on FULL Data ─────────────────────────────────────────────────────────
auto_model.fit(ts)

# ── Build Future Dates (Apr ↔ Sep alternating) ────────────────────────────────
n_future  = 6        # 2 seasons × 3 years
last_date = ts.index[-1]

future_dates = []
current = last_date

for _ in range(n_future):
    if current.month == 4:
        next_date = current.replace(month=9)
    else:
        next_date = current.replace(month=4, year=current.year + 1)
    future_dates.append(next_date)
    current = next_date

future_dates = pd.DatetimeIndex(future_dates)

# ── Forecast + Confidence Intervals ───────────────────────────────────────────
future_forecast, conf_int = auto_model.predict(
    n_periods=n_future,
    return_conf_int=True
)

pred_vals = np.array(future_forecast).round(2)
lower     = conf_int[:, 0].round(2)
upper     = conf_int[:, 1].round(2)

# ── Build Forecast DataFrame ──────────────────────────────────────────────────
future_df = pd.DataFrame({
    'Prediction' : pred_vals,
    'Lower_95'   : lower,
    'Upper_95'   : upper
}, index=future_dates)

future_df['Season'] = np.where(future_dates.month == 4, 'Yala', 'Maha')

future_df['Year'] = np.where(
    future_dates.month == 4,
    future_dates.year,
    future_dates.year + 1
)

future_df = future_df[['Year', 'Season', 'Prediction', 'Lower_95', 'Upper_95']]

print("\n3-Year Future Forecast:")
print(future_df.to_string())

# ── Plot ───────────────────────────────────────────────────────────────────────
plt.figure(figsize=(14, 6))

plt.plot(ts.index, ts,
         label="Historical Data",
         marker="o", linewidth=1.5, color='steelblue')

plt.plot(future_dates, pred_vals,
         label="3-Year Forecast",
         marker="o", linewidth=1.5, color="red", linestyle='--')

plt.fill_between(future_dates, lower, upper,
                 color="red", alpha=0.15,
                 label="95% Confidence Interval")

plt.axvline(x=last_date, color='gray',
            linestyle=':', linewidth=1.5,
            label='Forecast Start')

plt.xlabel("Year",                 fontsize=14, fontweight='bold', fontname='Times New Roman')
plt.ylabel("Production ('000 MT)", fontsize=14, fontweight='bold', fontname='Times New Roman')
plt.legend(prop={'size': 12, 'family': 'Times New Roman'})
plt.grid(axis='y')
plt.xticks(rotation=45, fontsize=13, fontname='Times New Roman')
plt.yticks(fontsize=13, fontname='Times New Roman')
plt.ylim(500,)
plt.tight_layout()
plt.show()
