import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading dataset...")

# Load dataset
df = pd.read_csv("sales.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Feature Engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Features and Target
X = df[["Year", "Month", "Day"]]
y = df["Sales"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("-" * 30)
print(f"MAE      : {mae:.2f}")
print(f"R2 Score : {r2:.4f}")

# Actual vs Predicted Graph
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Sales", marker="o")
plt.plot(predictions, label="Predicted Sales", marker="x")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Records")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()

# ---------------------------------------------------
# Future Sales Forecast (Next 7 Days)
# ---------------------------------------------------

last_date = df["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7
)

future_df = pd.DataFrame({
    "Year": future_dates.year,
    "Month": future_dates.month,
    "Day": future_dates.day
})

future_predictions = model.predict(future_df)

print("\nNext 7 Days Sales Forecast")
print("-" * 30)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_predictions
})

print(forecast_df)

# Forecast Visualization
plt.figure(figsize=(10, 5))
plt.bar(
    forecast_df["Date"].dt.strftime("%d-%m"),
    forecast_df["Predicted Sales"]
)

plt.title("Future Sales Forecast (Next 7 Days)")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.grid(True)
plt.show()

# Business Summary
avg_sales = forecast_df["Predicted Sales"].mean()

print("\nBusiness Insight")
print("-" * 30)
print(f"Average predicted sales for next 7 days: {avg_sales:.2f}")

if avg_sales > df["Sales"].mean():
    print("Forecast indicates sales are expected to increase.")
else:
    print("Forecast indicates sales are expected to remain stable or decrease.")