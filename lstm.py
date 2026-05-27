import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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
test = ts.iloc[train_size:]


# ======================
# 3. NORMALIZATION
# ======================
scaler = MinMaxScaler(feature_range=(-1, 1))

train_data = train.values.reshape(-1, 1)
train_scaled = scaler.fit_transform(train_data)

train_scaled = torch.FloatTensor(train_scaled).view(-1)


# ======================
# 4. CREATE SEQUENCES
# ======================
def create_inout_sequences(input_data, seq_len):
    inout_seq = []
    L = len(input_data)

    for i in range(L - seq_len):
        train_seq = input_data[i:i+seq_len]
        train_label = input_data[i+seq_len:i+seq_len+1]
        inout_seq.append((train_seq, train_label))

    return inout_seq


seq_len = 5
train_sequences = create_inout_sequences(train_scaled, seq_len)


# ======================
# 5. LSTM MODEL
# ======================
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, output_size=1):
        super().__init__()

        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ======================
# 6. TRAINING
# ======================
epochs = 50
losses = []

for epoch in range(epochs):
    epoch_loss = 0

    for seq, label in train_sequences:
        optimizer.zero_grad()

        y_pred = model(seq)
        loss = loss_function(y_pred, label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    losses.append(epoch_loss / len(train_sequences))

    print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.6f}")


# ======================
# 7. LOSS PLOT
# ======================
plt.figure(figsize=(10,4))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# ======================
# 8. PREDICTION
# ======================
model.eval()

predictions = []

test_input = train_scaled[-seq_len:].clone()

with torch.no_grad():
    for i in range(len(test)):
        y_pred = model(test_input)
        predictions.append(y_pred.item())

        test_input = torch.cat((test_input[1:], y_pred.view(1)))


# ======================
# 9. INVERSE TRANSFORM
# ======================
predictions = np.array(predictions).reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)


# ======================
# 10. PLOT RESULTS
# ======================
train_plot = ts[:train_size]
test_plot = ts[train_size:]

plt.figure(figsize=(12,6))

plt.plot(train_plot.index, train_plot.values, label="Train Data", color="blue")
plt.plot(test_plot.index, test_plot.values, label="Actual Test Data", color="green")
plt.plot(test_plot.index, predictions, label="Predicted Data", color="red")

plt.title("LSTM Time Series Forecasting")
plt.xlabel("Date")
plt.ylabel("Production")
plt.legend()
plt.show()

# evaluation matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ======================
# ALIGN ARRAYS
# ======================
y_true = test_plot.values
y_pred = predictions.reshape(-1)

# ======================
# METRICS
# ======================
mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mse = mean_squared_error(y_true, y_pred)

# MAPE (avoid divide-by-zero)
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

print("\n📊 Model Evaluation Metrics")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"MAPE : {mape:.2f}%")

# residual error plot
residuals = y_true - y_pred

plt.figure(figsize=(10,4))
plt.plot(residuals, color="purple")
plt.axhline(0, linestyle="--", color="black")
plt.title("Residual Errors (Actual - Predicted)")
plt.xlabel("Time Step")
plt.ylabel("Error")
plt.show()



#Error Distribution (Check model quality)
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=20, color="gray", edgecolor="black")
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

