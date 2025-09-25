import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
import base64

# --- Page config ---
st.set_page_config(layout="wide", page_title="Latin Countries Regression Explorer")

# --- App Title and Author ---
st.title("Regression & Function Analysis — Argentina, Chile, Mexico (70-year World Bank data)")
st.markdown("**Created by Amarachi Onwo**")

# --- App Description ---
st.markdown("""
This app fetches historical data from the World Bank for Argentina (ARG), Chile (CHL), and Mexico (MEX),
fits a polynomial regression (degree >= 3), and performs function analysis. This includes:
- Finding local maxima and minima,
- Determining when the function is increasing or decreasing,
- Identifying when the rate of change is fastest,
- Allowing interpolation and extrapolation for future years.

Use the controls in the left sidebar to pick:
- The category of data (Population, Unemployment rate, Education, Life expectancy, etc.),
- The countries to include in your analysis,
- Polynomial degree for the regression,
- How far back in years to analyze.

The raw data is shown in an editable table. The regression model is plotted as a scatter plot with the fitted curve.
Extrapolated future data can be shown with a dashed line. You can also generate printer-friendly reports.
""")

# --- Countries and Indicators ---
COUNTRIES = {"Argentina": "ARG", "Chile": "CHL", "Mexico": "MEX"}
INDICATORS = {
    "Population": {"code": "SP.POP.TOTL", "unit": "people"},
    "Unemployment rate": {"code": "SL.UEM.TOTL.ZS", "unit": "%"},
    "Life expectancy": {"code": "SP.DYN.LE00.IN", "unit": "years"},
    "Birth rate": {"code": "SP.DYN.CBRT.IN", "unit": "births per 1,000 people"},
    "Average income (GNI per capita, Atlas method)": {"code": "NY.GNP.PCAP.CD", "unit": "USD"},
    "Average wealth (proxy: GDP per capita)": {"code": "NY.GDP.PCAP.CD", "unit": "USD"},
    "Immigration (net migration per 1000)": {"code": "SM.POP.NETM", "unit": "net migrants per 1,000"},
    "Education (mean years of schooling, proxy)": {"code": "SE.SCH.LIFE", "unit": "0-25 scale"},
    "Murder Rate": {"code": "VC.IHR.PSRC.P5", "unit": "per 100,000"}
}

# --- Sidebar Controls ---
st.sidebar.header("Controls")
selected_category = st.sidebar.selectbox("Select data category", list(INDICATORS.keys()))
years_back = st.sidebar.slider("Amount of past years to display (max 70)", min_value=10, max_value=70, value=70, step=1)
countries_to_plot = st.sidebar.multiselect("Select countries to include", options=list(COUNTRIES.keys()), default=list(COUNTRIES.keys()))
poly_degree = st.sidebar.number_input("Polynomial degree (>=3)", min_value=3, max_value=8, value=3, step=1)
year_increment = st.sidebar.slider("Regression graph tick increment (years)", min_value=1, max_value=10, value=1)
show_extrapolation = st.sidebar.checkbox("Allow extrapolation (show future years)", value=True)
extrapolate_years = st.sidebar.number_input("Years to extrapolate (if enabled)", min_value=0, max_value=100, value=10, step=1)
printer_friendly = st.sidebar.checkbox("Show printer-friendly results area", value=False)

# --- Helper Functions ---

def fetch_wb(country_code, indicator_code, years_back=70):
    current_year = datetime.now().year
    start_year = current_year - years_back + 1
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={start_year}:{current_year}&format=json&per_page=2000"
    try:
        res = requests.get(url, timeout=15)
        data = res.json()
        if not data or len(data) < 2:
            return pd.DataFrame(columns=["year", "value"])
        rows = data[1]
        records = []
        for row in rows:
            year = int(row.get("date"))
            val = row.get("value")
            if val is None:
                continue
            records.append({"year": year, "value": float(val)})
        df = pd.DataFrame(records)
        df = df.sort_values("year").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Failed to fetch World Bank data: {e}")
        return pd.DataFrame(columns=["year", "value"])

def INDICATOR_UNIT(name):
    return INDICATORS.get(name, {}).get("unit", "")
    
def get_table_download_link(df, filename="data.csv"):
    if df is None or not isinstance(df, pd.DataFrame):
        st.error("No valid data to download!")
        return None
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# Example usage
if 'edited_df' in st.session_state and isinstance(st.session_state.edited_df, pd.DataFrame):
    st.markdown(get_table_download_link(st.session_state.edited_df, "edited_data.csv"), unsafe_allow_html=True)
else:
    st.info("No data available to download.")

# --- Fetch Data ---
all_data = {}
for country in countries_to_plot:
    code = COUNTRIES[country]
    indicator_code = INDICATORS[selected_category]["code"]
    df = fetch_wb(code, indicator_code, years_back)
    df["value_for_model"] = df["value"]
    all_data[country] = df

# --- Combine Data for Editable Table ---
years_union = set()
for df in all_data.values():
    years_union.update(df["year"].tolist())

if len(years_union) == 0:
    st.warning("No data available for the selected category / countries.")
    st.stop()

years_sorted = sorted(list(years_union))
base_table = pd.DataFrame({"year": years_sorted})
for country, df in all_data.items():
    merged = base_table.merge(df[["year", "value_for_model"]], on="year", how="left")
    base_table[country] = merged["value_for_model"]

# --- Editable table using latest Streamlit API ---
try:
    edited = st.experimental_data_editor(base_table, num_rows="dynamic")
except AttributeError:
    st.warning("Your Streamlit version is too old for `experimental_data_editor`. Displaying read-only table instead.")
    edited = base_table

# --- Ensure edited is a DataFrame for plotting and download ---
if not isinstance(edited, pd.DataFrame):
    try:
        edited_df = edited.data  # in some Streamlit versions
    except AttributeError:
        edited_df = base_table
else:
    edited_df = edited

# --- Prepare data for model ---
model_data = {}
for country in countries_to_plot:
    if country in edited_df.columns:
        country_df = edited_df[["year", country]].dropna().rename(columns={country: "value"})
        model_data[country] = country_df

# --- Plotting ---
st.subheader("Scatter plot + Polynomial fit")
fig, ax = plt.subplots(figsize=(10,6))
colors = {"Argentina":"tab:blue","Chile":"tab:green","Mexico":"tab:orange"}
analysis_text = []

for country in countries_to_plot:
    df = model_data.get(country, pd.DataFrame())
    if df.empty:
        continue
    x = df["year"].values
    y = df["value"].values
    ax.scatter(x, y, label=f"{country} raw", alpha=0.6)
    x_rel = x - x.min()
    coeffs = np.polyfit(x_rel, y, deg=poly_degree)
    poly = np.poly1d(coeffs)
    x_plot = np.arange(x.min(), x.max()+1, year_increment)
    x_plot_rel = x_plot - x.min()
    y_plot = poly(x_plot_rel)
    if show_extrapolation and extrapolate_years>0:
        x_future = np.arange(x.max()+1, x.max()+extrapolate_years+1, year_increment)
        y_future = poly(x_future - x.min())
        ax.plot(x_plot, y_plot, linestyle='-', color=colors.get(country))
        ax.plot(x_future, y_future, linestyle='--', color=colors.get(country))
    else:
        ax.plot(x_plot, y_plot, linestyle='-', color=colors.get(country))
  eq_terms = [f'({c:.4e})*t^{len(coeffs)-1-i}' for i, c in enumerate(coeffs)]
equation = ' + '.join(eq_terms)
analysis_text.append(f"Equation for {country} (t = years since {int(x.min())}): {equation}")
