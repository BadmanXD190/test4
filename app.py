import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Thailand Inflation Dashboard", layout="wide")

# ---------- Helpers ----------
def load_csv_safely(path_or_buffer):
    try:
        df = pd.read_csv(path_or_buffer)
    except UnicodeDecodeError:
        df = pd.read_csv(path_or_buffer, encoding="latin-1")
    return df

def detect_cols(df):
    cols = {c.lower().strip(): c for c in df.columns}
    # find year-like column
    year_col = None
    for k, v in cols.items():
        if k in ["year", "tahun"]:
            year_col = v
            break
    if year_col is None:
        # fallback to the first integer column that looks like a year
        for c in df.columns:
            if pd.api.types.is_integer_dtype(df[c]) and df[c].between(1900, 2100).all():
                year_col = c
                break
    # value column = first numeric column that is not year
    value_col = None
    for c in df.columns:
        if c != year_col and pd.api.types.is_numeric_dtype(df[c]):
            value_col = c
            break
    return year_col, value_col

def plot_history_forecast(hist_df, fc_df, hist_year, hist_val, fc_year, fc_val, train_year_end=2017, test_year_end=2023):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(hist_df[hist_year], hist_df[hist_val], label="Actual (History)", linewidth=2)

    # Forecast line and markers
    ax.plot(fc_df[fc_year], fc_df[fc_val], label="Forecast", linestyle="--", marker="o")

    # Shaded regions for train and test if years overlap
    try:
        ymin, ymax = ax.get_ylim()
        if hist_df[hist_year].min() <= train_year_end:
            ax.axvspan(hist_df[hist_year].min(), train_year_end, color="tab:green", alpha=0.08, label="Train period")
        if train_year_end < test_year_end <= hist_df[hist_year].max():
            ax.axvspan(train_year_end, test_year_end, color="tab:orange", alpha=0.08, label="Test period")
    except Exception:
        pass

    ax.set_title("Thailand Headline Inflation YoY — History and Forecast")
    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation YoY (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    st.pyplot(fig)

def df_download_button(df, filename, label):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ---------- Sidebar inputs ----------
st.sidebar.header("Data files")
default_hist = "thai_headline_inflation_yoy_annual.csv"
default_fc = "annual_lstm_forecast.csv"

hist_path = st.sidebar.text_input("History CSV path", value=default_hist)
fc_path = st.sidebar.text_input("Forecast CSV path", value=default_fc)

st.sidebar.caption("Keep the default names if you commit the CSVs in the same folder as this app.")

show_images = st.sidebar.checkbox("Show saved model images if available", value=True)
st.sidebar.caption("Files searched: annual_lstm_test_plot.png, annual_lstm_residuals.png, annual_lstm_learning_curves.png, annual_lstm_forecast_full.png")

# ---------- Load data ----------
hist_df = None
fc_df = None
hist_error = None
fc_error = None

if os.path.exists(hist_path):
    hist_df = load_csv_safely(hist_path)
else:
    hist_error = f"Could not find {hist_path}. Please upload or correct the path."

if os.path.exists(fc_path):
    fc_df = load_csv_safely(fc_path)
else:
    fc_error = f"Could not find {fc_path}. Please upload or correct the path."

# ---------- Main layout ----------
st.title("Thailand Headline Inflation YoY")

if hist_error or fc_error:
    st.error("File loading issue")
    if hist_error:
        st.write(hist_error)
    if fc_error:
        st.write(fc_error)
    st.stop()

# Detect columns
hist_year_col, hist_val_col = detect_cols(hist_df)
fc_year_col, fc_val_col = detect_cols(fc_df)

if not hist_year_col or not hist_val_col:
    st.error("Could not detect the year and value columns in the history CSV. Make sure it has columns like Year and Inflation.")
    st.write(hist_df.head())
    st.stop()

if not fc_year_col or not fc_val_col:
    st.error("Could not detect the year and value columns in the forecast CSV. Make sure it has columns like Year and Forecast.")
    st.write(fc_df.head())
    st.stop()

# Sort by year, ensure numeric
hist_df = hist_df.copy().sort_values(hist_year_col)
fc_df = fc_df.copy().sort_values(fc_year_col)
hist_df[hist_year_col] = pd.to_numeric(hist_df[hist_year_col], errors="coerce")
fc_df[fc_year_col] = pd.to_numeric(fc_df[fc_year_col], errors="coerce")

# Combined table
combined = pd.concat(
    [
        hist_df[[hist_year_col, hist_val_col]].rename(columns={hist_year_col: "Year", hist_val_col: "Inflation_YoY"}),
        fc_df[[fc_year_col, fc_val_col]].rename(columns={fc_year_col: "Year", fc_val_col: "Inflation_YoY"})
    ],
    axis=0,
    ignore_index=True
)
combined["Type"] = np.where(combined["Year"].isin(hist_df[hist_year_col]), "Actual", "Forecast")
combined = combined.sort_values("Year")

# ---------- KPI row ----------
col1, col2, col3 = st.columns(3)
last_actual_row = hist_df.dropna(subset=[hist_year_col, hist_val_col]).tail(1)
last_actual_year = int(last_actual_row[hist_year_col].values[0])
last_actual_value = float(last_actual_row[hist_val_col].values[0])
col1.metric("Last actual year", f"{last_actual_year}")
col2.metric("Last actual inflation", f"{last_actual_value:.2f}%")

# average forecast
avg_fc = float(fc_df[fc_val_col].mean())
col3.metric("Average forecast", f"{avg_fc:.2f}%")

# ---------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["Charts", "Data tables", "Saved images"])

with tab1:
    st.subheader("History and forecast")
    train_end = st.number_input("Train period ends at year", value=2017, step=1)
    test_end = st.number_input("Test period ends at year", value=2023, step=1)

    plot_history_forecast(
        hist_df, fc_df,
        hist_year_col, hist_val_col,
        fc_year_col, fc_val_col,
        train_year_end=int(train_end),
        test_year_end=int(test_end)
    )

with tab2:
    st.subheader("Data tables")
    st.markdown("**History (1979–2023)**")
    st.dataframe(hist_df.rename(columns={hist_year_col: "Year", hist_val_col: "Inflation_YoY"}), use_container_width=True)
    df_download_button(hist_df.rename(columns={hist_year_col: "Year", hist_val_col: "Inflation_YoY"}), "thai_headline_inflation_yoy_annual_clean.csv", "Download history CSV")

    st.markdown("**Forecast (2024–2026)**")
    st.dataframe(fc_df.rename(columns={fc_year_col: "Year", fc_val_col: "Inflation_YoY"}), use_container_width=True)
    df_download_button(fc_df.rename(columns={fc_year_col: "Year", fc_val_col: "Inflation_YoY"}), "annual_lstm_forecast_clean.csv", "Download forecast CSV")

    st.markdown("**Combined**")
    st.dataframe(combined, use_container_width=True)
    df_download_button(combined, "combined_inflation.csv", "Download combined CSV")

with tab3:
    st.subheader("Saved images")
    if show_images:
        img_files = [
            ("annual_lstm_test_plot.png", "Test-window plot"),
            ("annual_lstm_residuals.png", "Residuals histogram"),
            ("annual_lstm_learning_curves.png", "Learning curves"),
            ("annual_lstm_forecast_full.png", "Full history and forecast")
        ]
        any_found = False
        for fname, caption in img_files:
            if os.path.exists(fname):
                st.image(fname, caption=caption, use_column_width=True)
                any_found = True
        if not any_found:
            st.info("No image files found in the app folder. Add PNGs to display them here.")
    else:
        st.info("Toggle on the sidebar to display images if they exist.")
