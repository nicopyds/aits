import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/nicolaepopescul/code/streams/aits/ts_passangers.csv")
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m")
df["y_log"] = np.log(df["y"])

n = len(df)
train_size = int(n * 0.9)

train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

train_log = train["y_log"].reset_index(drop=True)
test_log = test["y_log"].reset_index(drop=True)

# Best SARIMA: (0,1,1)(1,0,1,12)
model = SARIMAX(train_log, order=(0, 1, 1), seasonal_order=(1, 0, 1, 12))
fitted = model.fit(disp=False)

forecast_log = fitted.forecast(steps=len(test))
# extrae los "pred" del train
train_pred_log = fitted.fittedvalues.iloc[1:]

# RMSE on log scale
train_rmse_log = np.sqrt(mean_squared_error(train_log.iloc[1:], train_pred_log))
test_rmse_log = np.sqrt(mean_squared_error(test_log, forecast_log))

print(f"Train RMSE best ARIMA model (log): {train_rmse_log:.4f}")
print(f"Test RMSE best ARIMA model (log): {test_rmse_log:.4f}")

# Convert to original scale
test_pred_orig = np.exp(forecast_log)
train_pred_orig = np.exp(train_pred_log)

train_rmse = np.sqrt(mean_squared_error(train["y"].iloc[1:], train_pred_orig))
test_rmse = np.sqrt(mean_squared_error(test["y"], test_pred_orig))

print(f"Train RMSE (original): {train_rmse:.4f}")
print(f"Test RMSE (original): {test_rmse:.4f}")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

train_dates = train["ds"].iloc[1:].reset_index(drop=True)
test_dates = test["ds"].reset_index(drop=True)

ax1.plot(train_dates, train_log.iloc[1:], label="Train", color="blue")
ax1.plot(test_dates, test_log, label="Test", color="green")
ax1.plot(test_dates, forecast_log, label="Prediction", color="red", linestyle="--")
ax1.set_xlabel("Date")
ax1.set_ylabel("y_log")
ax1.set_title("SARIMA (0,1,1)(1,0,1,12): Log Scale")
ax1.legend()
ax1.grid(True)

ax2.plot(train_dates, train["y"].iloc[1:], label="Train", color="blue")
ax2.plot(test_dates, test["y"], label="Test", color="green")
ax2.plot(test_dates, test_pred_orig, label="Prediction", color="red", linestyle="--")
ax2.set_xlabel("Date")
ax2.set_ylabel("Passengers")
ax2.set_title("SARIMA (0,1,1)(1,0,1,12): Original Scale")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("/Users/nicolaepopescul/code/streams/aits/sarima_pred2.png", dpi=100)
print("Plot saved to /Users/nicolaepopescul/code/streams/aits/sarima_pred.png")
