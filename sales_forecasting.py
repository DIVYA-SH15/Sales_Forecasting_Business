import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

print("Loading dataset...")

# ----------------------------
# Load Dataset
# ----------------------------
df = pd.read_csv("sales.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("Columns found:", df.columns.tolist())

# Check required columns
if "Date" not in df.columns:
    raise Exception(f"'Date' column not found. Columns available: {df.columns.tolist()}")

if "Sales" not in df.columns:
    raise Exception(f"'Sales' column not found. Columns available: {df.columns.tolist()}")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Feature Engineering
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Features and Target
X = df[["Year", "Month", "Day"]]
y = df["Sales"]

# Split Data
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

# Predict
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"Mean Absolute Error : {mae:.2f}")
print(f"R2 Score            : {r2:.4f}")

# ----------------------------
# Actual vs Predicted
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(y_test.values, marker="o", label="Actual")
plt.plot(predictions, marker="x", label="Predicted")
plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Records")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.savefig("actual_vs_predicted.png")
plt.close()

print("Saved graph: actual_vs_predicted.png")

# ----------------------------
# Future Forecast
# ----------------------------
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

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted Sales": future_predictions
})

print("\n==============================")
print("NEXT 7 DAYS FORECAST")
print("==============================")
print(forecast_df)

# Forecast Graph
plt.figure(figsize=(10,5))
plt.bar(
    forecast_df["Date"].dt.strftime("%d-%m"),
    forecast_df["Predicted Sales"]
)

plt.title("Next 7 Days Sales Forecast")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")
plt.grid(True)

plt.savefig("forecast.png")
plt.close()

print("Saved graph: forecast.png")

# ----------------------------
# Business Insight
# ----------------------------
average_prediction = forecast_df["Predicted Sales"].mean()
average_sales = df["Sales"].mean()

print("\n==============================")
print("BUSINESS INSIGHT")
print("==============================")
print(f"Average Historical Sales : {average_sales:.2f}")
print(f"Average Forecast Sales   : {average_prediction:.2f}")

if average_prediction > average_sales:
    print("Sales are expected to increase.")
else:
    print("Sales are expected to remain stable or decrease.")