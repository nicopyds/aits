import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

df = pd.read_csv("/Users/nicolaepopescul/code/streams/aits/ts_passangers.csv")
df.columns = ["ds", "y"]
df["ds"] = pd.to_datetime(df["ds"], format="%Y-%m")

df["y_log"] = np.log(df["y"])
df["y_log_diff"] = df["y_log"].diff()

n = len(df)
train_size = int(n * 0.9)

train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

train_y = train["y_log_diff"].dropna()
test_y = test["y_log_diff"].dropna()

train_prophet = train[train["y_log_diff"].notna()].copy()
test_prophet = test[test["y_log_diff"].notna()].copy()

train_prophet["y"] = train_prophet["y_log_diff"]
test_prophet["y"] = test_prophet["y_log_diff"]

model = Prophet(
    yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False
)
model.fit(train_prophet[["ds", "y"]])

train_pred = model.predict(train_prophet[["ds"]])
test_pred = model.predict(test_prophet[["ds"]])

train_rmse = np.sqrt(mean_squared_error(train_prophet["y"], train_pred["yhat"]))
test_rmse = np.sqrt(mean_squared_error(test_prophet["y"], test_pred["yhat"]))

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(train_prophet["ds"], train_prophet["y"], label="Train", color="blue")
plt.plot(test_prophet["ds"], test_prophet["y"], label="Test", color="green")
plt.plot(
    test_prophet["ds"],
    test_pred["yhat"],
    label="Prediction",
    color="red",
    linestyle="--",
)
plt.xlabel("Date")
plt.ylabel("y_log_diff")
plt.title("Prophet Model on Stationary TS: Train, Test and Predictions")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/Users/nicolaepopescul/code/streams/aits/prophet_pred1.png", dpi=100)
print("Plot saved to /Users/nicolaepopescul/code/streams/aits/prophet_pred1.png")
