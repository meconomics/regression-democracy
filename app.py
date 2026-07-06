"""
Democracy, Development & Growth — OLS Regression Dashboard
============================================================
An interactive Streamlit tool for exploring the relationship between
democratic institutions, economic development, and average YoY GDP growth
(or any other numeric dataset you upload).

Run with:
    streamlit run regression_dashboard.py
"""

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import streamlit as st

sns.set_style("whitegrid")

st.set_page_config(
    page_title="Democracy, Development & Growth — OLS Dashboard",
    layout="wide",
)

# ----------------------------------------------------------------------
# Sample data generator
# ----------------------------------------------------------------------
def generate_sample_data(n=160, seed=42):
    """
    Synthetic country-year style panel resembling the kind of variables used
    in democracy/growth research (Polity5, V-Dem, World Bank WDI style):
      - democracy_index      : 0 (closed autocracy) - 10 (full democracy)
      - rule_of_law           : World Bank WGI-style index, -2.5 to 2.5
      - control_corruption    : WGI-style index, -2.5 to 2.5
      - gdp_per_capita_start  : GDP per capita (USD) at start of period
      - investment_gdp_share  : gross fixed capital formation, % of GDP
      - avg_yoy_growth        : average annual real GDP growth, % (target)

    This is illustrative data for demoing the tool, NOT real-world figures.
    Replace it with actual V-Dem / Polity5 / World Bank WDI data for
    genuine analysis.
    """
    rng = np.random.default_rng(seed)

    democracy_index = np.clip(rng.normal(5.5, 2.8, n), 0, 10)
    rule_of_law = np.clip(0.35 * (democracy_index - 5) + rng.normal(0, 0.7, n), -2.5, 2.5)
    control_corruption = np.clip(0.3 * (democracy_index - 5) + rng.normal(0, 0.8, n), -2.5, 2.5)
    gdp_per_capita_start = np.clip(
        np.exp(rng.normal(8.5, 1.1, n)) + 300 * democracy_index, 300, None
    )
    investment_gdp_share = np.clip(22 + 0.4 * rule_of_law + rng.normal(0, 5, n), 10, 45)

    # Growth story: some catch-up/convergence (poorer grows faster),
    # a modest positive institutional effect, positive investment effect,
    # plus noise. Coefficients chosen for a plausible, debatable pattern —
    # not a claim about the real world.
    log_gdp = np.log(gdp_per_capita_start)
    avg_yoy_growth = (
        6.5
        - 0.55 * (log_gdp - log_gdp.mean())
        + 0.18 * democracy_index
        + 0.35 * rule_of_law
        + 0.05 * investment_gdp_share
        + rng.normal(0, 1.4, n)
    )

    df = pd.DataFrame(
        {
            "country_id": [f"C{i:03d}" for i in range(1, n + 1)],
            "democracy_index": democracy_index.round(2),
            "rule_of_law": rule_of_law.round(2),
            "control_corruption": control_corruption.round(2),
            "gdp_per_capita_start": gdp_per_capita_start.round(0),
            "investment_gdp_share": investment_gdp_share.round(2),
            "avg_yoy_growth": avg_yoy_growth.round(2),
        }
    )
    return df


# ----------------------------------------------------------------------
# Main — header
# ----------------------------------------------------------------------
st.title("📊 Democracy, Development & Growth — OLS Dashboard")
st.caption(
    "Upload a dataset, type your own data in by hand, or use the built-in sample "
    "(e.g. democracy indices, GDP per capita, growth rates) and run an OLS "
    "regression with full diagnostics."
)

# ----------------------------------------------------------------------
# Sidebar — data input
# ----------------------------------------------------------------------
st.sidebar.header("1. Data")

data_mode = st.sidebar.radio(
    "Data source",
    ["Upload CSV", "Enter data manually", "Built-in sample dataset"],
)

st.sidebar.markdown(
    "**Looking for real institutional data?** Try "
    "[V-Dem](https://www.v-dem.net/), "
    "[Polity5](https://www.systemicpeace.org/polityproject.html), "
    "the [World Bank WGI](https://www.worldbank.org/en/publication/worldwide-governance-indicators), "
    "or [World Bank WDI](https://databank.worldbank.org/source/world-development-indicators)."
)

df = None
data_source = None

if data_mode == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        data_source = uploaded_file.name

elif data_mode == "Built-in sample dataset":
    df = generate_sample_data()
    data_source = "synthetic sample data"

elif data_mode == "Enter data manually":
    DEFAULT_MANUAL_COLS = [
        "country", "democracy_index", "gdp_per_capita_start", "avg_yoy_growth"
    ]

    if "manual_columns" not in st.session_state:
        st.session_state.manual_columns = DEFAULT_MANUAL_COLS.copy()
    if "manual_df" not in st.session_state:
        st.session_state.manual_df = pd.DataFrame(
            [{c: None for c in st.session_state.manual_columns} for _ in range(5)]
        )

    st.subheader("✍️ Enter your data")
    st.caption(
        "Edit the column names below (comma-separated), then fill in rows in the "
        "table. Click the '+' row at the bottom to add more observations, or the "
        "trash icon on a row to delete it."
    )

    cols_text = st.text_input(
        "Column names (comma-separated)",
        value=", ".join(st.session_state.manual_columns),
    )
    new_cols = [c.strip() for c in cols_text.split(",") if c.strip()]

    if new_cols and new_cols != st.session_state.manual_columns:
        old_df = st.session_state.manual_df
        rebuilt = pd.DataFrame(index=old_df.index)
        for c in new_cols:
            rebuilt[c] = old_df[c] if c in old_df.columns else None
        st.session_state.manual_columns = new_cols
        st.session_state.manual_df = rebuilt

    edited_df = st.data_editor(
        st.session_state.manual_df,
        num_rows="dynamic",
        use_container_width=True,
        key="manual_editor",
    )
    st.session_state.manual_df = edited_df

    # Convert columns to numeric where possible, leave as text otherwise
    manual_df = edited_df.copy()
    for c in manual_df.columns:
        converted = pd.to_numeric(manual_df[c], errors="coerce")
        # Keep numeric conversion only if at least one value actually parsed
        if converted.notna().any():
            manual_df[c] = converted

    manual_df = manual_df.dropna(how="all")

    if manual_df.empty:
        st.info("Add at least one row of data above to continue.")
        st.stop()

    df = manual_df
    data_source = "manually entered data"

if df is None:
    st.info(
        "Upload a CSV, switch to 'Enter data manually', or select the built-in "
        "sample dataset in the sidebar to get started."
    )
    st.stop()

st.success(f"Loaded data from: **{data_source}**  ·  {df.shape[0]} rows × {df.shape[1]} columns")

with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.write("**Summary statistics**")
    st.dataframe(df.describe().T, use_container_width=True)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) < 2:
    st.error("Need at least two numeric columns to run a regression.")
    st.stop()

# ----------------------------------------------------------------------
# Sidebar — variable selection
# ----------------------------------------------------------------------
st.sidebar.header("2. Variables")

default_y = "avg_yoy_growth" if "avg_yoy_growth" in numeric_cols else numeric_cols[-1]
y_var = st.sidebar.selectbox(
    "Dependent variable (Y)", numeric_cols, index=numeric_cols.index(default_y)
)

x_candidates = [c for c in numeric_cols if c != y_var]
default_x = [c for c in x_candidates if c in ("democracy_index", "gdp_per_capita_start")] or x_candidates[:1]
x_vars = st.sidebar.multiselect(
    "Independent variable(s) (X)", x_candidates, default=default_x
)

log_transform_y = st.sidebar.checkbox("Use log(Y)", value=False)
log_transform_x = st.sidebar.multiselect(
    "Use log(X) for:", x_vars, default=[]
)
add_constant = st.sidebar.checkbox("Include intercept", value=True)
robust_se = st.sidebar.checkbox("Use heteroskedasticity-robust SEs (HC3)", value=False)

if not x_vars:
    st.warning("Select at least one independent variable in the sidebar.")
    st.stop()

# ----------------------------------------------------------------------
# Build regression dataframe
# ----------------------------------------------------------------------
work_cols = list(set([y_var] + x_vars))
reg_df = df[work_cols].apply(pd.to_numeric, errors="coerce").dropna()

y_label = y_var
X_labels = list(x_vars)

y = reg_df[y_var].copy()
if log_transform_y:
    if (y <= 0).any():
        st.error(f"Cannot take log of '{y_var}': contains non-positive values.")
        st.stop()
    y = np.log(y)
    y_label = f"log({y_var})"

X = reg_df[x_vars].copy()
for col in log_transform_x:
    if (X[col] <= 0).any():
        st.error(f"Cannot take log of '{col}': contains non-positive values.")
        st.stop()
    X[col] = np.log(X[col])
    X_labels[x_vars.index(col)] = f"log({col})"
X.columns = X_labels

n_obs = len(reg_df)
n_dropped = len(df) - n_obs
if n_dropped > 0:
    st.warning(f"Dropped {n_dropped} rows with missing/non-numeric values in selected columns.")

if n_obs <= len(X_labels) + 1:
    st.error("Not enough observations to fit this model. Select fewer variables or upload more data.")
    st.stop()

if add_constant:
    X_model = sm.add_constant(X)
else:
    X_model = X

# ----------------------------------------------------------------------
# Fit OLS
# ----------------------------------------------------------------------
cov_type = "HC3" if robust_se else "nonrobust"
model = sm.OLS(y, X_model).fit(cov_type=cov_type)

st.header("Regression Results")
st.markdown(
    f"**Model:** `{y_label} ~ {' + '.join(X_labels)}{' + const' if add_constant else ''}`  "
    f"&nbsp;·&nbsp; N = {n_obs} &nbsp;·&nbsp; SE type: `{cov_type}`"
)

col_summary, col_metrics = st.columns([2, 1])

with col_summary:
    st.subheader("Coefficient table")
    coef_table = pd.DataFrame(
        {
            "coef": model.params,
            "std err": model.bse,
            "t": model.tvalues,
            "p-value": model.pvalues,
            "[0.025": model.conf_int()[0],
            "0.975]": model.conf_int()[1],
        }
    )
    stars = coef_table["p-value"].apply(
        lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
    )
    coef_table["sig"] = stars
    st.dataframe(coef_table.style.format(precision=4), use_container_width=True)
    st.caption("Significance: *** p<0.01, ** p<0.05, * p<0.10")

with col_metrics:
    st.subheader("Model fit")
    st.metric("R²", f"{model.rsquared:.4f}")
    st.metric("Adj. R²", f"{model.rsquared_adj:.4f}")
    st.metric("F-statistic", f"{model.fvalue:.3f}")
    st.metric("Prob (F-stat)", f"{model.f_pvalue:.4g}")
    st.metric("N observations", f"{int(model.nobs)}")
    st.metric("AIC / BIC", f"{model.aic:.1f} / {model.bic:.1f}")

with st.expander("Full statsmodels summary (text)"):
    st.code(str(model.summary()), language="text")

# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
st.header("Diagnostic Plots & Tests")

fitted = model.fittedvalues
resid = model.resid

d1, d2 = st.columns(2)

with d1:
    st.subheader("Fitted vs. Actual")
    if len(X_labels) == 1:
        # Simple bivariate case: scatter + fitted line against the single X
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        xcol = X_labels[0]
        sns.scatterplot(x=X[xcol], y=y, ax=ax, alpha=0.7, edgecolor="white")
        order = np.argsort(X[xcol].values)
        ax.plot(X[xcol].values[order], fitted.values[order], color="crimson", linewidth=2)
        ax.set_xlabel(xcol)
        ax.set_ylabel(y_label)
        ax.set_title(f"{y_label} vs. {xcol} with OLS fit")
        st.pyplot(fig)
    else:
        # Multivariate case: predicted vs actual
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        sns.scatterplot(x=fitted, y=y, ax=ax, alpha=0.7, edgecolor="white")
        lims = [min(fitted.min(), y.min()), max(fitted.max(), y.max())]
        ax.plot(lims, lims, color="crimson", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Fitted values")
        ax.set_ylabel(f"Actual {y_label}")
        ax.set_title("Predicted vs. Actual")
        st.pyplot(fig)

with d2:
    st.subheader("Residuals vs. Fitted")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.scatterplot(x=fitted, y=resid, ax=ax, alpha=0.7, edgecolor="white")
    ax.axhline(0, color="crimson", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual plot (check for heteroskedasticity)")
    st.pyplot(fig)

d3, d4 = st.columns(2)

with d3:
    st.subheader("Residual distribution")
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    sns.histplot(resid, kde=True, ax=ax, color="steelblue")
    ax.set_xlabel("Residual")
    ax.set_title("Histogram of residuals")
    st.pyplot(fig)

with d4:
    st.subheader("Q-Q plot")
    fig = sm.qqplot(resid, line="s")
    fig.set_size_inches(5.5, 4.2)
    st.pyplot(fig)

st.subheader("Formal diagnostic tests")
test_cols = st.columns(3)

# Breusch-Pagan test for heteroskedasticity
try:
    bp_stat, bp_p, bp_f, bp_fp = het_breuschpagan(resid, X_model)
    with test_cols[0]:
        st.metric("Breusch-Pagan p-value", f"{bp_p:.4g}")
        st.caption(
            "H0: homoskedasticity. "
            + ("⚠️ Evidence of heteroskedasticity (p<0.05)." if bp_p < 0.05 else "No strong evidence of heteroskedasticity.")
        )
except Exception:
    with test_cols[0]:
        st.write("Breusch-Pagan test unavailable.")

# Durbin-Watson for autocorrelation (relevant for time series / panel data)
dw = durbin_watson(resid)
with test_cols[1]:
    st.metric("Durbin-Watson", f"{dw:.3f}")
    st.caption("~2 = no autocorrelation; <1.5 or >2.5 may indicate autocorrelation.")

# VIF for multicollinearity (only meaningful with 2+ X variables)
with test_cols[2]:
    if len(X_labels) > 1:
        vif_data = pd.DataFrame()
        vif_df = X_model.drop(columns="const") if add_constant else X_model
        vif_data["variable"] = vif_df.columns
        vif_data["VIF"] = [
            variance_inflation_factor(vif_df.values, i) for i in range(vif_df.shape[1])
        ]
        st.write("**Variance Inflation Factors**")
        st.dataframe(vif_data.style.format({"VIF": "{:.2f}"}), use_container_width=True)
        st.caption("VIF > 5–10 suggests problematic multicollinearity.")
    else:
        st.write("VIF requires 2+ independent variables.")

# ----------------------------------------------------------------------
# Correlation matrix (context for variable selection)
# ----------------------------------------------------------------------
st.header("Correlation Matrix")
corr_cols = list(set([y_var] + x_vars))
fig, ax = plt.subplots(figsize=(0.9 * len(corr_cols) + 2, 0.7 * len(corr_cols) + 2))
sns.heatmap(
    df[corr_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, vmin=-1, vmax=1
)
st.pyplot(fig)

st.markdown("---")
st.caption(
    "Built for exploring institutional-quality / development / growth relationships. "
    "Correlation and regression results here are descriptive, not causal — "
    "consider omitted-variable bias, reverse causality (growth may also strengthen "
    "institutions), and measurement error in cross-country indices before drawing "
    "policy conclusions."
)
