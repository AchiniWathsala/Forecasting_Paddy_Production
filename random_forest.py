import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ======================
# 1. LOAD DATA
# ======================
data = pd.read_excel("df.xlsx")

df = data.iloc[27:, :]  # remove unwanted rows
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

ts = df["Production"]


# ======================
# 2. TRAIN / TEST SPLIT
# ======================
train_size = int(len(ts) * 0.8)

train = ts.iloc[:train_size]
test  = ts.iloc[train_size:]


# ======================
# 3. NORMALIZATION
# ======================
scaler = MinMaxScaler(feature_range=(-1, 1))

train_data = train.values.reshape(-1, 1)
train_scaled = scaler.fit_transform(train_data).flatten()


# ======================
# 4. CREATE SEQUENCES
# ======================
def create_inout_sequences(input_data, seq_len):
    X, y = [], []
    for i in range(len(input_data) - seq_len):
        X.append(input_data[i:i+seq_len])   # features (past values)
        y.append(input_data[i+seq_len])      # label (next value)
    return np.array(X), np.array(y)

seq_len = 5
X_train, y_train = create_inout_sequences(train_scaled, seq_len)


# ======================
# 5. RANDOM FOREST MODEL
# ======================
model = RandomForestRegressor(
    n_estimators=100,    # number of trees
    max_depth=10,        # depth of each tree
    random_state=42
)


# ======================
# 6. TRAINING
# ======================
model.fit(X_train, y_train)
print("✅ Random Forest training complete!")


# ======================
# 7. PREDICTION
# ======================
predictions = []

# Start with last seq_len values from training data
test_input = list(train_scaled[-seq_len:])

for i in range(len(test)):
    x_input = np.array(test_input[-seq_len:]).reshape(1, -1)
    y_pred = model.predict(x_input)[0]
    predictions.append(y_pred)
    test_input.append(y_pred)   # append prediction for next step


# ======================
# 8. INVERSE TRANSFORM
# ======================
predictions = np.array(predictions).reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)


# ======================
# 9. PLOT RESULTS
# ======================
train_plot = ts[:train_size]
test_plot  = ts[train_size:]

plt.figure(figsize=(12, 6))
plt.plot(train_plot.index, train_plot.values, label="Train Data",        color="blue")
plt.plot(test_plot.index,  test_plot.values,  label="Actual Test Data",  color="green")
plt.plot(test_plot.index,  predictions,       label="Predicted Data",    color="red")

plt.title("Random Forest Time Series Forecasting")
plt.xlabel("Date")
plt.ylabel("Production")
plt.legend()
plt.show()


# ======================
# 10. EVALUATION METRICS
# ======================
y_true = test_plot.values
y_pred = predictions.reshape(-1)

mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

print("\n📊 Model Evaluation Metrics")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"MAPE : {mape:.2f}%")


# ======================
# 11. RESIDUAL ERROR PLOT
# ======================
residuals = y_true - y_pred

plt.figure(figsize=(10, 4))
plt.plot(residuals, color="purple")
plt.axhline(0, linestyle="--", color="black")
plt.title("Residual Errors (Actual - Predicted)")
plt.xlabel("Time Step")
plt.ylabel("Error")
plt.show()


# ======================
# 12. ERROR DISTRIBUTION
# ======================
plt.figure(figsize=(8, 4))
plt.hist(residuals, bins=20, color="gray", edgecolor="black")
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()