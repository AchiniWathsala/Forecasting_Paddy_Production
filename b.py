# =========================================
# 1. IMPORT LIBRARIES
# =========================================

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from itertools import product

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================
# 2. LOAD DATA
# =========================================

data = pd.read_excel("df.xlsx")

df = data.iloc[27:, :]
df = df.sort_values("Date")
df.set_index("Date", inplace=True)

ts = df["Avg_Yield_Total"]

# =========================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# =========================================

train_size = int(len(ts) * 0.7)
val_size   = int(len(ts) * 0.1)   # increased from 0.10 to 0.15
test_size  = int(len(ts) * 0.1)   # adjusted accordingly

train      = ts.iloc[:train_size]
validation = ts.iloc[train_size : train_size + val_size]
test       = ts.iloc[train_size + val_size :]

print(f"Dataset size  : {len(ts)}")
print(f"Train size    : {len(train)}")
print(f"Validation sz : {len(validation)}")
print(f"Test size     : {len(test)}")

# =========================================
# 4. NORMALIZATION
# =========================================

scaler = MinMaxScaler(feature_range=(-1, 1))

train_scaled      = scaler.fit_transform(train.values.reshape(-1, 1))
validation_scaled = scaler.transform(validation.values.reshape(-1, 1))
test_scaled       = scaler.transform(test.values.reshape(-1, 1))

train_scaled      = torch.FloatTensor(train_scaled).view(-1)
validation_scaled = torch.FloatTensor(validation_scaled).view(-1)
test_scaled       = torch.FloatTensor(test_scaled).view(-1)

# =========================================
# 5. CREATE SEQUENCES (SLIDING WINDOW)
# =========================================

def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        x = data[i : i + seq_len]
        y = data[i + seq_len : i + seq_len + 1]
        sequences.append((x, y))
    return sequences

# =========================================
# 6. LSTM MODEL
# =========================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(len(x), 1, -1)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[-1])
        return out

# =========================================
# 7. PARAMETER COMBINATIONS TO TRY
# =========================================

# Cap seq_len values to be safe relative to the smallest split
min_split_len = min(len(train_scaled), len(validation_scaled), len(test_scaled))
max_seq_len   = min_split_len - 1   # need at least 1 sample after the window

raw_seq_lens = [3, 5, 10]
valid_seq_lens = [s for s in raw_seq_lens if s < max_seq_len]

if not valid_seq_lens:
    # Fallback: use half the smallest split
    valid_seq_lens = [max(1, max_seq_len // 2)]
    print(f"Warning: all seq_len values exceeded data size. "
          f"Using seq_len={valid_seq_lens} as fallback.")

param_grid = {
    "seq_len"     : valid_seq_lens,
    "hidden_size" : [32, 50, 100],
    "lr"          : [0.001, 0.0005],
    "epochs"      : [80],
}

# Build all combinations
keys   = list(param_grid.keys())
combos = list(product(*param_grid.values()))

print(f"\nTotal combinations to try: {len(combos)}\n")
print(f"{'#':<4} {'seq_len':<10} {'hidden':<10} {'lr':<10} {'epochs':<8} {'Val Loss'}")
print("-" * 58)

# =========================================
# 8. GRID SEARCH LOOP
# =========================================

results = []

for idx, combo in enumerate(combos):
    params = dict(zip(keys, combo))

    seq_len     = params["seq_len"]
    hidden_size = params["hidden_size"]
    lr          = params["lr"]
    epochs      = params["epochs"]

    # Build sequences for this seq_len
    train_seq = create_sequences(train_scaled, seq_len)
    val_seq   = create_sequences(validation_scaled, seq_len)

    # ── GUARD: skip if not enough data to form at least one sequence ──
    if len(train_seq) == 0 or len(val_seq) == 0:
        print(f"{idx+1:<4} {seq_len:<10} {hidden_size:<10} {lr:<10} {epochs:<8} "
              f"SKIPPED (train_seq={len(train_seq)}, val_seq={len(val_seq)})")
        continue

    # Build model
    model     = LSTMModel(hidden_size=hidden_size)
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training
    for epoch in range(epochs):
        model.train()
        for seq, label in train_seq:
            optimizer.zero_grad()
            y_pred = model(seq)
            loss   = loss_fn(y_pred, label)
            loss.backward()
            optimizer.step()

    # Validation loss after training
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seq, label in val_seq:
            y_pred    = model(seq)
            val_loss += loss_fn(y_pred, label).item()
    val_loss /= len(val_seq)    # safe — guarded above

    results.append({
        "combo_id"    : idx + 1,
        "seq_len"     : seq_len,
        "hidden_size" : hidden_size,
        "lr"          : lr,
        "epochs"      : epochs,
        "val_loss"    : val_loss,
        "model"       : model,
    })

    print(f"{idx+1:<4} {seq_len:<10} {hidden_size:<10} {lr:<10} {epochs:<8} {val_loss:.6f}")

# ── Abort gracefully if no valid results ──
if not results:
    raise RuntimeError(
        "No valid parameter combinations found. "
        "Your dataset may be too small for the chosen seq_len values. "
        f"Dataset size: {len(ts)}, smallest split: {min_split_len}."
    )

# =========================================
# 9. PICK BEST COMBINATION
# =========================================

best = min(results, key=lambda x: x["val_loss"])

print("\n" + "=" * 50)
print("  BEST COMBINATION")
print("=" * 50)
print(f"  seq_len     : {best['seq_len']}")
print(f"  hidden_size : {best['hidden_size']}")
print(f"  lr          : {best['lr']}")
print(f"  epochs      : {best['epochs']}")
print(f"  val_loss    : {best['val_loss']:.6f}")

# =========================================
# 10. RETRAIN BEST MODEL (fresh weights)
#     with training + validation losses
# =========================================

seq_len     = best["seq_len"]
hidden_size = best["hidden_size"]
lr          = best["lr"]
epochs      = best["epochs"]

train_seq = create_sequences(train_scaled, seq_len)
val_seq   = create_sequences(validation_scaled, seq_len)

model     = LSTMModel(hidden_size=hidden_size)
loss_fn   = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
val_losses   = []

for epoch in range(epochs):

    # Training
    model.train()
    train_loss = 0.0
    for seq, label in train_seq:
        optimizer.zero_grad()
        y_pred      = model(seq)
        loss        = loss_fn(y_pred, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_seq))

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for seq, label in val_seq:
            y_pred    = model(seq)
            val_loss += loss_fn(y_pred, label).item()
    val_losses.append(val_loss / len(val_seq))

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_losses[-1]:.6f} | "
          f"Val Loss: {val_losses[-1]:.6f}")

# =========================================
# 11. LOSS CURVES
# =========================================

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses,   label="Validation Loss")
plt.title(f"LSTM Loss Curves — Best Params "
          f"(seq={seq_len}, hidden={hidden_size}, lr={lr})")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================
# 12. COMPARE ALL COMBINATIONS (bar chart)
# =========================================

combo_labels = [
    f"seq={r['seq_len']}\nhid={r['hidden_size']}\nlr={r['lr']}"
    for r in results
]
val_losses_all = [r["val_loss"] for r in results]

colors = ["green" if r["combo_id"] == best["combo_id"] else "steelblue"
          for r in results]

plt.figure(figsize=(max(10, len(results) * 1.2), 5))
bars = plt.bar(range(len(results)), val_losses_all, color=colors)
plt.xticks(range(len(results)), combo_labels, fontsize=7)
plt.title("Validation Loss per Parameter Combination\n(green = best)")
plt.ylabel("Val Loss (MSE)")
plt.tight_layout()
plt.show()

# =========================================
# 13. FORECASTING (TEST DATA)
# =========================================

model.eval()
predictions = []
input_seq   = train_scaled[-seq_len:].clone()

with torch.no_grad():
    for i in range(len(test)):
        y_pred = model(input_seq)
        predictions.append(y_pred.item())
        input_seq = torch.cat((input_seq[1:], y_pred.view(1)))

# =========================================
# 14. INVERSE TRANSFORM
# =========================================

predictions = np.array(predictions).reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)

# =========================================
# 15. PLOT RESULTS
# =========================================

plt.figure(figsize=(12, 6))
plt.plot(train.index, train.values, label="Train")
plt.plot(test.index,  test.values,  label="Actual Test")
plt.plot(test.index,  predictions,  label="Predicted", linestyle="--")
plt.title(f"LSTM Forecasting — Best Params "
          f"(seq={seq_len}, hidden={hidden_size}, lr={lr})")
plt.xlabel("Date")
plt.ylabel("Production")
plt.legend()
plt.tight_layout()
plt.show()

# =========================================
# 16. MODEL EVALUATION
# =========================================

y_true = test.values
y_pred = predictions.reshape(-1)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
r2   = r2_score(y_true, y_pred)

print("\n===== MODEL PERFORMANCE (Best Combination) =====")
print(f"  seq_len     : {seq_len}")
print(f"  hidden_size : {hidden_size}")
print(f"  lr          : {lr}")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

# =========================================
# 17. SUMMARY TABLE OF ALL COMBINATIONS
# =========================================

summary = pd.DataFrame([{
    "seq_len"     : r["seq_len"],
    "hidden_size" : r["hidden_size"],
    "lr"          : r["lr"],
    "epochs"      : r["epochs"],
    "val_loss"    : round(r["val_loss"], 6),
} for r in results]).sort_values("val_loss")

print("\n===== ALL COMBINATIONS (sorted by val_loss) =====")
print(summary.to_string(index=False))