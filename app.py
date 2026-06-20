import io
import pandas as pd
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(
    page_title="Sales Forecasting",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv_from_upload(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        raise ValueError("No file provided")
    raw = uploaded_file.read()
    df = pd.read_csv(io.BytesIO(raw))
    return df


def validate_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    if "Date" not in df.columns:
        raise KeyError(f"'Date' column not found. Columns available: {df.columns.tolist()}")
    if "Sales" not in df.columns:
        raise KeyError(f"'Sales' column not found. Columns available: {df.columns.tolist()}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Some values in 'Date' could not be parsed. Check date format.")

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    return df


def train_and_forecast(df: pd.DataFrame, n_days: int, n_estimators: int, random_state: int):
    features = df[["Year", "Month", "Day"]]
    target = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=random_state,
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    last_date = df["Date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
    future_df = pd.DataFrame({
        "Year": future_dates.year,
        "Month": future_dates.month,
        "Day": future_dates.day,
    })

    future_predictions = model.predict(future_df)

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Sales": future_predictions,
    })

    business_avg_hist = float(df["Sales"].mean())
    business_avg_forecast = float(forecast_df["Predicted Sales"].mean())

    insight = (
        "Sales are expected to increase."
        if business_avg_forecast > business_avg_hist
        else "Sales are expected to remain stable or decrease."
    )

    return {
        "model": model,
        "mae": float(mae),
        "r2": float(r2),
        "y_test": y_test.reset_index(drop=True),
        "predictions": pd.Series(predictions),
        "forecast_df": forecast_df,
        "business_avg_hist": business_avg_hist,
        "business_avg_forecast": business_avg_forecast,
        "insight": insight,
    }


def plot_actual_vs_predicted(y_test: pd.Series, predictions: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test.values, marker="o", linestyle="-", label="Actual")
    ax.plot(predictions, marker="x", linestyle="--", label="Predicted")
    ax.set_title("Actual vs Predicted Sales")
    ax.set_xlabel("Test Records")
    ax.set_ylabel("Sales")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_forecast(forecast_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    x_labels = forecast_df["Date"].dt.strftime("%d-%m")
    ax.bar(x_labels, forecast_df["Predicted Sales"], color="#4f46e5")
    ax.set_title(f"Next {len(forecast_df)} Days Sales Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Sales")
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def main():
    st.title("📈 Sales Forecasting UI")
    st.caption("Train a Random Forest model on `sales.csv` (or your uploaded CSV), then forecast the next days.")

    with st.sidebar:
        st.header("Configuration")

        default_days = 7
        n_days = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=default_days)

        n_estimators = st.slider("Random Forest trees", min_value=50, max_value=500, value=100, step=10)
        random_state = st.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)

        st.divider()
        st.subheader("Dataset")
        st.write("Upload a CSV with columns: **Date**, **Sales**.")

        uploaded_file = st.file_uploader("Choose CSV", type=["csv"], accept_multiple_files=False)

        use_default = st.toggle("Use repo `sales.csv`", value=(uploaded_file is None))

    try:
        if use_default and uploaded_file is None:
            df = pd.read_csv("sales.csv")
        else:
            df = load_csv_from_upload(uploaded_file)

        df = validate_and_engineer_features(df)
    except Exception as e:
        st.error(f"Dataset error: {e}")
        return

    if st.checkbox("Show dataset preview", value=False):
        st.dataframe(df.head(50), use_container_width=True)

    st.divider()

    if st.button("✅ Train & Forecast", type="primary"):
        with st.spinner("Training model and generating forecast..."):
            result = train_and_forecast(
                df=df,
                n_days=int(n_days),
                n_estimators=int(n_estimators),
                random_state=int(random_state),
            )

        mae = result["mae"]
        r2 = result["r2"]

        c1, c2, c3 = st.columns([1, 1, 2])
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("R²", f"{r2:.4f}")
        c3.info(
            f"**Historical avg:** {result['business_avg_hist']:.2f} | "
            f"**Forecast avg:** {result['business_avg_forecast']:.2f}\n\n{result['insight']}"
        )

        st.subheader("Model performance")
        st.pyplot(plot_actual_vs_predicted(result["y_test"], result["predictions"]), use_container_width=True)

        st.subheader("Forecast table")
        forecast_df = result["forecast_df"].copy()
        forecast_df["Predicted Sales"] = forecast_df["Predicted Sales"].round(2)
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        st.subheader("Forecast chart")
        st.pyplot(plot_forecast(forecast_df), use_container_width=True)


if __name__ == "__main__":
    main()

