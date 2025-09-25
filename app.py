import streamlit as st

st.set_page_config(layout="wide", page_title="Latin Countries Regression Explorer")

# App Title
st.title("Regression & Function Analysis — Argentina, Chile, Mexico (70-year World Bank data)")

# Your name at the top
st.markdown("**Created by Amarachi Onwo**")

# App description — safe ASCII, properly terminated triple quotes
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

**Notes:** Data is loaded from the World Bank API. Some indicators have gaps for early years; the app uses available annual observations across the last 70 years.


# --- Countries and World Bank codes
COUNTRIES = {"Argentina":"ARG","Chile":"CHL","Mexico":"MEX"}

INDICATORS = {
    "Population": {"code":"SP.POP.TOTL","unit":"people"},
    "Unemployment rate": {"code":"SL.UEM.TOTL.ZS","unit":"percent"},
    "Life expectancy": {"code":"SP.DYN.LE00.IN","unit":"years"},
    "Birth rate": {"code":"SP.DYN.CBRT.IN","unit":"births per 1,000 people"},
    "Murder Rate": {"code":"VC.IHR.PSRC.P5","unit":"per 100,000"}, # placeholder - World Bank doesn't have universal murder; will try a better one below
    "Average income (GNI per capita, Atlas method)": {"code":"NY.GNP.PCAP.CD","unit":"USD"},
    "Average wealth (proxy: GDP per capita)": {"code":"NY.GDP.PCAP.CD","unit":"USD"},
    "Immigration (net migration per 1000)": {"code":"SM.POP.NETM","unit":"net migrants per 1,000"},
    "Education (mean years of schooling, proxy)":{"code":"SE.SCH.LIFE","unit":"education index (0-25, scaled later)"}
}

# Adjust mapping for indicators that World Bank doesn't directly provide with the requested names:
CUSTOM_INDICATOR_FIXES = {
    "Murder Rate": {"wb_code":"VC.IHR.PSRC.P5","note":"World Bank homicide data is uneven; app will attempt to use 'VC.IHR.PSRC.P5' if available, otherwise gaps may appear."},
    "Education (mean years of schooling, proxy)": {"wb_code":"SE.SCH.LIFE","note":"This is a proxy placeholder; OWID/UN data may be required for precise mean years; we will scale to 0-25 for visualization."}
}

# Sidebar controls
st.sidebar.header("Controls")
selected_category = st.sidebar.selectbox("Select data category", list(INDICATORS.keys()))
years_back = st.sidebar.slider("Amount of past years to display (max 70)", min_value=10, max_value=70, value=70, step=1)
countries_to_plot = st.sidebar.multiselect("Select countries to include", options=list(COUNTRIES.keys()), default=list(COUNTRIES.keys()))
poly_degree = st.sidebar.number_input("Polynomial degree (>=3)", min_value=3, max_value=8, value=3, step=1)
year_increment = st.sidebar.slider("Regression graph tick increment (years)", min_value=1, max_value=10, value=1)
show_extrapolation = st.sidebar.checkbox("Allow extrapolation (show future years)", value=True)
extrapolate_years = st.sidebar.number_input("Years to extrapolate (if enabled)", min_value=0, max_value=100, value=10, step=1)
compare_latins_in_us = st.sidebar.checkbox("Compare Latin groups in the U.S. (placeholder)", value=False)
printer_friendly = st.sidebar.checkbox("Show printer-friendly results area", value=False)

# Helper: fetch World Bank data for a country + indicator for the last N years
def fetch_wb(country_code, indicator_code, years_back=70):
    current_year = datetime.now().year
    start_year = current_year - years_back + 1
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={start_year}:{current_year}&format=json&per_page=2000"
    try:
        res = requests.get(url, timeout=15)
        data = res.json()
        if not data or len(data) < 2:
            return pd.DataFrame(columns=["year","value"])
        rows = data[1]
        records = []
        for row in rows:
            year = int(row.get("date"))
            val = row.get("value")
            if val is None:
                continue
            records.append({"year":year,"value":float(val)})
        df = pd.DataFrame(records)
        df = df.sort_values("year").reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Failed to fetch World Bank data: {e}")
        return pd.DataFrame(columns=["year","value"])

# Prepare and combine data for selected countries
all_data = {}
for country in countries_to_plot:
    code = COUNTRIES[country]
    # pick code; allow fixes
    if selected_category in CUSTOM_INDICATOR_FIXES:
        wb_code = CUSTOM_INDICATOR_FIXES[selected_category].get("wb_code", INDICATORS[selected_category].get("code"))
    else:
        wb_code = INDICATORS[selected_category]["code"]
    df = fetch_wb(code, wb_code, years_back=years_back)
    # Handle scaling for education proxy: scale to 0-25 based on known min/max in dataset
    if selected_category == "Education (mean years of schooling, proxy)" and not df.empty:
        # scale to 0-25 using min/max from available values
        vmin = df["value"].min()
        vmax = df["value"].max()
        if vmax>vmin:
            df["value_scaled"] = 25 * (df["value"] - vmin) / (vmax - vmin)
            df["value_for_model"] = df["value_scaled"]
        else:
            df["value_for_model"] = df["value"]
    else:
        df["value_for_model"] = df["value"] if "value" in df.columns else np.nan
    
    all_data[country] = df

# Show raw combined editable table (merge by year, countries as columns)
st.subheader("Raw Data (editable)")
# Create a years index based on union of years
years_union = set()
for df in all_data.values():
    years_union.update(df["year"].tolist())
if len(years_union)==0:
    st.warning("No data available for the selected category / countries. Try a different category or fewer years.")
    st.stop()

years_sorted = sorted(list(years_union))
base_table = pd.DataFrame({"year": years_sorted})
for country, df in all_data.items():
    merged = base_table.merge(df[["year","value_for_model"]], on="year", how="left")
    base_table[country] = merged["value_for_model"]

# Make the table editable by the user
edited = st.experimental_data_editor(base_table, num_rows="dynamic")

# Save edited table back into country-specific dataframes for modeling
model_data = {}
for country in countries_to_plot:
    country_df = edited[["year", country]].dropna().rename(columns={country:"value"})
    country_df = country_df.sort_values("year").reset_index(drop=True)
    model_data[country] = country_df

# Plotting area
st.subheader("Scatter plot + Polynomial fit")
fig, ax = plt.subplots(figsize=(10,6))

colors = {"Argentina":"tab:blue","Chile":"tab:green","Mexico":"tab:orange"}
legend_items = []

# For analysis text
analysis_text = []

for country in countries_to_plot:
    df = model_data[country]
    if df.empty:
        continue
    x = df["year"].values
    y = df["value"].values
    ax.scatter(x, y, label=f"{country} raw", alpha=0.6)
    # Fit polynomial (years -> convert to numeric relative to first year to reduce conditioning issues)
    x_rel = x - x.min()
    coeffs = np.polyfit(x_rel, y, deg=poly_degree)
    poly = np.poly1d(coeffs)
    # Prepare domain for plotting including extrapolation
    x_plot = np.arange(x.min(), x.max()+1, year_increment)
    x_plot_rel = x_plot - x.min()
    y_plot = poly(x_plot_rel)
    # Extrapolation part
    if show_extrapolation and extrapolate_years>0:
        x_future = np.arange(x.max()+1, x.max()+extrapolate_years+1, year_increment)
        x_future_rel = x_future - x.min()
        y_future = poly(x_future_rel)
        # plot known-range curve
        ax.plot(x_plot, y_plot, linestyle='-', label=f"{country} fit", color=colors.get(country))
        ax.plot(x_future, y_future, linestyle='--', label=f"{country} extrapolation", color=colors.get(country))
    else:
        ax.plot(x_plot, y_plot, linestyle='-', label=f"{country} fit", color=colors.get(country))
    # Equation text (show coefficients)
    eq_terms = []
    deg = len(coeffs)-1
    for i,c in enumerate(coeffs):
        p = deg - i
        eq_terms.append(f"({c:.4e})*t^{p}")
    equation = " + ".join(eq_terms)
    analysis_text.append(f"Equation for {country} (t = years since {int(x.min())}):\n{equation}")
    # Function analysis via derivative
    # Convert poly to np.poly1d then derivative
    dpoly = np.polyder(poly)
    ddpoly = np.polyder(poly, 2)
    # Find critical points in the observed domain (relative), solve derivative roots
    crit_roots = np.roots(dpoly.coeffs)
    real_crit = [r for r in crit_roots if np.isreal(r)]
    real_crit = np.real(real_crit)
    # Filter critical points to those within observed relative domain
    rel_domain_min = 0
    rel_domain_max = x.max()-x.min()
    crit_in_domain = [r for r in real_crit if r>=rel_domain_min and r<=rel_domain_max]
    # Evaluate second derivative to classify
    crit_points = []
    for r in crit_in_domain:
        second = ddpoly(r)
        typ = "minimum" if second>0 else ("maximum" if second<0 else "inflection/flat")
        year_abs = int(round(r + x.min()))
        crit_points.append((year_abs, typ))
    # Fastest increase/decrease approximated by finding max/min of derivative on domain
    # Evaluate derivative over fine grid
    fine_rel = np.linspace(rel_domain_min, rel_domain_max, 500)
    deriv_vals = dpoly(fine_rel)
    max_idx = np.argmax(deriv_vals)
    min_idx = np.argmin(deriv_vals)
    year_fastest_increase = int(round(fine_rel[max_idx] + x.min()))
    year_fastest_decrease = int(round(fine_rel[min_idx] + x.min()))
    rate_fastest_increase = deriv_vals[max_idx]
    rate_fastest_decrease = deriv_vals[min_idx]
    # Domain and range (observed)
    dom_text = f"Observed domain: {int(x.min())} to {int(x.max())} (years)"
    range_text = f"Observed range (approx): {y.min():.3g} to {y.max():.3g} ({INDICATOR_UNIT(selected_category)})" if 'INDICATOR_UNIT' in globals() else f"Observed range (approx): {y.min():.3g} to {y.max():.3g}"
    # Compose analysis sentences
    sentences = []
    for yp in crit_points:
        sentences.append(f"The {selected_category.lower()} of {country} reached a local {yp[1]} on {yp[0]}.")
    sentences.append(f"The {country} was growing at its fastest rate on {year_fastest_increase} at about {rate_fastest_increase:.3g} {INDICATOR_UNIT(selected_category) if 'INDICATOR_UNIT' in globals() else ''} per year.")
    sentences.append(f"The {country} was decreasing at its fastest rate on {year_fastest_decrease} at about {rate_fastest_decrease:.3g} {INDICATOR_UNIT(selected_category) if 'INDICATOR_UNIT' in globals() else ''} per year.")
    sentences.append(dom_text)
    sentences.append(range_text)
    analysis_text.append("\\n".join(sentences))

ax.set_xlabel("Year")
ax.set_ylabel(selected_category + (f" ({INDICATORS[selected_category]['unit']})" if selected_category in INDICATORS else ""))
ax.legend()
st.pyplot(fig)

# Display analysis text
st.subheader("Function analysis (automatic)")
for t in analysis_text:
    st.markdown(t)

# Interpolation / extrapolation tool
st.subheader("Interpolate / Extrapolate a value using the model")
country_choice = st.selectbox("Pick a country for interpolation/extrapolation", options=countries_to_plot)
year_input = st.number_input("Input year (can be outside dataset)", value=int(datetime.now().year), step=1)
use_country_df = model_data[country_choice]
if not use_country_df.empty:
    x = use_country_df["year"].values
    y = use_country_df["value"].values
    x_rel = x - x.min()
    coeffs = np.polyfit(x_rel, y, deg=poly_degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(year_input - x.min())
    st.write(f"Predicted {selected_category} for {country_choice} in {year_input}: {y_pred:.3g} {INDICATORS.get(selected_category,{}).get('unit','')} (model extrapolation/interpolation)")
else:
    st.write("No data for selected country to predict.")

# Average rate of change tool
st.subheader("Average rate of change over an interval (model-based)")
country_arc = st.selectbox("Choose country", options=countries_to_plot, key="arc_country")
year_a = st.number_input("Start year (for rate)", value=int(x.min()) if 'x' in globals() and len(x)>0 else int(datetime.now().year)-5)
year_b = st.number_input("End year (for rate)", value=int(x.max()) if 'x' in globals() and len(x)>0 else int(datetime.now().year))
if country_arc in model_data and not model_data[country_arc].empty:
    df = model_data[country_arc]
    x = df["year"].values
    y = df["value"].values
    coeffs = np.polyfit(x - x.min(), y, deg=poly_degree)
    poly = np.poly1d(coeffs)
    val_a = poly(year_a - x.min())
    val_b = poly(year_b - x.min())
    avg_rate = (val_b - val_a) / (year_b - year_a) if year_b!=year_a else np.nan
    st.write(f"Average rate of change of {selected_category} in {country_arc} from {year_a} to {year_b}: {avg_rate:.5g} {INDICATORS.get(selected_category,{}).get('unit','')} per year")

# Printer friendly area: produce simple HTML for printing
if printer_friendly:
    st.subheader("Printer-friendly report")
    report_html = "<html><body>"
    report_html += f"<h2>Report: {selected_category} — {' ,'.join(countries_to_plot)}</h2>"
    report_html += "<h3>Raw data</h3>"
    report_html += edited.to_html(index=False)
    report_html += "<h3>Analysis</h3>"
    for t in analysis_text:
        report_html += f"<p>{t}</p>"
    report_html += "</body></html>"
    st.markdown(report_html, unsafe_allow_html=True)
    st.markdown("Use your browser print function to print this report (or save as PDF).")

# Utility function to create download links for CSV of edited table
def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False).encode('utf-8')
    b64 = base64.b64encode(csv).decode()
    return f"data:file/csv;base64,{b64}"

st.subheader("Download data")
csv_link = get_table_download_link(edited, "edited_data.csv")
st.markdown(f"[Download edited data as CSV]({csv_link})")

# Helper to display unit - define here to avoid errors above
def INDICATOR_UNIT(name):
    return INDICATORS.get(name,{}).get("unit","")

# End of app
