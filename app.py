# app.py
from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_prep import load_data
from src.metrics import (
    build_company_summary,
    build_portfolio_summary,
    value_creation_decomposition,
    build_risk_flags,
    build_key_insights,
)

st.set_page_config(page_title="PE Portfolio Performance Dashboard", layout="wide")


# -----------------------------
# Utility helpers
# -----------------------------
def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _latest_available_reporting_date(financials: pd.DataFrame, valuation: pd.DataFrame) -> pd.Timestamp:
    fin_max = pd.to_datetime(financials["period_end"], errors="coerce").max() if "period_end" in financials.columns else pd.NaT
    val_max = pd.to_datetime(valuation["as_of_date"], errors="coerce").max() if "as_of_date" in valuation.columns else pd.NaT
    candidates = [d for d in [fin_max, val_max] if pd.notna(d)]
    latest = max(candidates) if candidates else pd.Timestamp.today()
    return pd.Timestamp(latest).normalize()


def _to_quarter_end(dt: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(dt).to_period("Q").end_time.normalize()


def _df_cols(df: pd.DataFrame, desired: list[str]) -> list[str]:
    return [c for c in desired if c in df.columns]


# -----------------------------
# Formatting helpers
# -----------------------------
def _money_mm(x: float) -> float:
    """Convert to USD mm (heuristic: if abs >= 1e5 assume dollars)."""
    if x is None or pd.isna(x):
        return float("nan")
    x = float(x)
    return x / 1e6 if abs(x) >= 1e5 else x


def _fmt_mm(x: float) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{_money_mm(x):,.1f}"


def _fmt_x(x: float) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):.2f}x"


def _fmt_pct(x: float) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x) * 100:.1f}%"


# -----------------------------
# Plotly "PE clean" helpers
# -----------------------------
def _q_end(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, errors="coerce").dt.to_period("Q").dt.end_time.dt.normalize()


def _q_label(ts: pd.Series) -> pd.Series:
    p = pd.to_datetime(ts, errors="coerce").dt.to_period("Q")
    return p.astype(str).str.replace("Q", " Q", regex=False)  # 2024Q3 -> 2024 Q3


def _to_mm(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").map(_money_mm)


def _pe_layout(fig: go.Figure, *, height: int = 380, showlegend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    fig.update_xaxes(title_text="", tickangle=0)
    fig.update_yaxes(title_text="")
    return fig


def _hover_mm(name: str) -> str:
    return f"{name}<br>%{{x}}<br>%{{y:,.1f}} USD mm<extra></extra>"


def _hover_x(name: str) -> str:
    return f"{name}<br>%{{x}}<br>%{{y:.2f}}x<extra></extra>"


def _hover_pct(name: str) -> str:
    return f"{name}<br>%{{x}}<br>%{{y:.1%}}<extra></extra>"


def _shorten(s: str, n: int = 22) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _bar_h_x(
    df: pd.DataFrame,
    *,
    label_col: str,
    value_col: str,
    value_name: str,
    key: str,
    height: int | None = None,
):
    """Horizontal bar chart for 'x' metrics like MOIC/Leverage. Short labels + full hover."""
    if df.empty:
        st.caption("No data available.")
        return

    d = df.copy()
    d[label_col] = d[label_col].astype(str)
    d["label_short"] = d[label_col].map(_shorten)

    if height is None:
        height = min(900, max(380, 28 * len(d) + 140))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d[value_col],
            y=d["label_short"],
            orientation="h",
            name=value_name,
            customdata=d[[label_col]].values,
            hovertemplate=f"{value_name}<br>%{{customdata[0]}}<br>%{{x:.2f}}x<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="", automargin=True)
    fig = _pe_layout(fig, height=height, showlegend=False)
    fig.update_layout(margin=dict(l=170, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)


def _bar_h_pct(
    df: pd.DataFrame,
    *,
    label_col: str,
    value_col: str,
    value_name: str,
    key: str,
    height: int | None = None,
):
    """Horizontal bar chart for % metrics like IRR (stored as fraction)."""
    if df.empty:
        st.caption("No data available.")
        return

    d = df.copy()
    d[label_col] = d[label_col].astype(str)
    d["label_short"] = d[label_col].map(_shorten)

    if height is None:
        height = min(900, max(380, 28 * len(d) + 140))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=d[value_col],
            y=d["label_short"],
            orientation="h",
            name=value_name,
            customdata=d[[label_col]].values,
            hovertemplate=f"{value_name}<br>%{{customdata[0]}}<br>%{{x:.1%}}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="")
    fig.update_yaxes(title_text="", automargin=True)
    fig = _pe_layout(fig, height=height, showlegend=False)
    fig.update_layout(margin=dict(l=170, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)


def _pie_usdmm(df: pd.DataFrame, names_col: str, values_col: str, *, key: str, height: int = 380):
    if df.empty:
        st.caption("No data available.")
        return
    fig = go.Figure(
        data=[
            go.Pie(
                labels=df[names_col],
                values=df[values_col],
                textinfo="label+percent",
                hovertemplate=f"%{{label}}<br>%{{value:,.1f}} USD mm<br>%{{percent}}<extra></extra>",
            )
        ]
    )
    fig = _pe_layout(fig, height=height, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=key)


def _bar_h_usdmm(df: pd.DataFrame, y_col: str, x_col: str, *, key: str, height: int = 420, name: str = "Value"):
    if df.empty:
        st.caption("No data available.")
        return
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df[x_col],
            y=df[y_col],
            orientation="h",
            name=name,
            hovertemplate=_hover_mm(name),
        )
    )
    fig.update_xaxes(title_text="USD mm")
    fig.update_yaxes(title_text="", automargin=True)
    fig = _pe_layout(fig, height=height, showlegend=False)
    fig.update_layout(margin=dict(l=170, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True, key=key)


# -----------------------------
# Load
# -----------------------------
@st.cache_data
def load_all():
    return load_data("data")


companies, financials, valuation, capital_flows = load_all()

# -----------------------------
# Sidebar — Company selector (persistent)
# -----------------------------
def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

company_id_col = _first_existing_col(companies, ["company_id", "asset_id", "id", "portfolio_company_id"])
company_name_col = _first_existing_col(companies, ["company_name", "name", "company", "asset_name"])

if company_id_col is None or company_name_col is None:
    st.sidebar.error("Companies table must include an id + name column.")
    st.stop()

companies_list = (
    companies[[company_id_col, company_name_col]]
    .dropna()
    .drop_duplicates()
    .sort_values(company_name_col)
)

ALL_LABEL = "All companies"

options = [ALL_LABEL] + companies_list[company_name_col].tolist()

# Persist selection across reruns
selected_name = st.sidebar.selectbox(
    "Company",
    options=options,
    index=0,
    key="selected_company_name",
)

selected_company_id = None
if selected_name != ALL_LABEL:
    selected_company_id = companies_list.loc[
        companies_list[company_name_col] == selected_name, company_id_col
    ].iloc[0]

# -----------------------------
# Drilldown — Company detail expander (only when one company selected)
# -----------------------------
def _coerce_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def _pick_date_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return _first_existing_col(df, candidates)

def _pick_num_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return _first_existing_col(df, candidates)

def _filter_company(df: pd.DataFrame, company_id: object) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cid = _first_existing_col(df, ["company_id", "asset_id", "id", "portfolio_company_id"])
    if cid is None:
        return df.iloc[0:0]
    return df[df[cid] == company_id].copy()

def _latest_row(df: pd.DataFrame, date_col: str) -> pd.Series | None:
    if df is None or df.empty or date_col not in df.columns:
        return None
    tmp = df.copy()
    tmp[date_col] = _coerce_dt(tmp[date_col])
    tmp = tmp.dropna(subset=[date_col]).sort_values(date_col)
    if tmp.empty:
        return None
    return tmp.iloc[-1]

def _safe_div(a, b):
    try:
        if a is None or b is None:
            return None
        if pd.isna(a) or pd.isna(b) or float(b) == 0.0:
            return None
        return float(a) / float(b)
    except Exception:
        return None

def _fmt_money(x):
    if x is None or pd.isna(x):
        return "n/a"
    x = float(x)
    # heuristic: show mm if large
    return f"${x/1e6:,.1f}m" if abs(x) >= 1e6 else f"${x:,.0f}"

def _fmt_x(x, suffix="x"):
    if x is None or pd.isna(x):
        return "n/a"
    return f"{float(x):,.2f}{suffix}"

def _fmt_pct(x):
    if x is None or pd.isna(x):
        return "n/a"
    return f"{100*float(x):,.1f}%"

if selected_company_id is not None:
    with st.expander(f"Company detail — {selected_name}", expanded=False):
        fin_c = _filter_company(financials, selected_company_id)
        val_c = _filter_company(valuation, selected_company_id)
        cf_c  = _filter_company(capital_flows, selected_company_id)

        # --- Financials: trends + latest KPI row
        fin_date = _pick_date_col(fin_c, ["period_end", "date", "as_of_date", "quarter_end", "month_end"])
        rev_col  = _pick_num_col(fin_c, ["revenue", "sales", "total_revenue"])
        ebitda_col = _pick_num_col(fin_c, ["ebitda", "adj_ebitda", "adjusted_ebitda"])

        debt_col = _pick_num_col(fin_c, ["total_debt", "debt", "net_debt"])  # net_debt may already exist
        cash_col = _pick_num_col(fin_c, ["cash", "cash_and_equivalents"])

        # If net_debt exists, prefer it; else derive net debt if possible
        net_debt_col = _pick_num_col(fin_c, ["net_debt"])
        if net_debt_col is None and debt_col is not None and cash_col is not None:
            fin_c["net_debt_derived"] = fin_c[debt_col] - fin_c[cash_col]
            net_debt_col = "net_debt_derived"

        if fin_date is not None:
            fin_c[fin_date] = _coerce_dt(fin_c[fin_date])
            fin_c = fin_c.dropna(subset=[fin_date]).sort_values(fin_date)

        latest_fin = _latest_row(fin_c, fin_date) if fin_date else None

        # --- KPI row (company)
        st.subheader("KPIs (latest reporting period)")

        kpi_cols = st.columns(6)

        latest_rev = latest_fin[rev_col] if (latest_fin is not None and rev_col) else None
        latest_ebitda = latest_fin[ebitda_col] if (latest_fin is not None and ebitda_col) else None
        latest_margin = _safe_div(latest_ebitda, latest_rev)

        latest_net_debt = latest_fin[net_debt_col] if (latest_fin is not None and net_debt_col) else None
        latest_leverage = _safe_div(latest_net_debt, latest_ebitda)

        report_label = str(latest_fin[fin_date].date()) if (latest_fin is not None and fin_date) else "n/a"
        kpi_cols[0].metric("Report date", report_label)
        kpi_cols[1].metric("Revenue", _fmt_money(latest_rev))
        kpi_cols[2].metric("EBITDA", _fmt_money(latest_ebitda))
        kpi_cols[3].metric("EBITDA margin", _fmt_pct(latest_margin))
        kpi_cols[4].metric("Net debt", _fmt_money(latest_net_debt))
        kpi_cols[5].metric("Leverage (ND/EBITDA)", _fmt_x(latest_leverage))

        # --- Revenue/EBITDA trend
        st.subheader("Revenue & EBITDA trend")

        if fin_date and rev_col and ebitda_col and not fin_c.empty:
            import plotly.express as px
            trend_df = fin_c[[fin_date, rev_col, ebitda_col]].copy()
            trend_df = trend_df.rename(columns={fin_date: "date", rev_col: "Revenue", ebitda_col: "EBITDA"})
            fig = px.line(trend_df, x="date", y=["Revenue", "EBITDA"], markers=True)
            fig.update_layout(legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Revenue/EBITDA trend requires financials columns: date + revenue + ebitda.")

        # --- Margin + leverage trend
        st.subheader("Margin & leverage trend")

        if fin_date and rev_col and ebitda_col and net_debt_col and not fin_c.empty:
            t = fin_c[[fin_date, rev_col, ebitda_col, net_debt_col]].copy()
            t["margin"] = t[ebitda_col] / t[rev_col]
            t["leverage"] = t[net_debt_col] / t[ebitda_col]
            t = t.rename(columns={fin_date: "date"})

            import plotly.graph_objects as go
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=t["date"], y=t["margin"], name="EBITDA margin", mode="lines+markers", yaxis="y1"))
            fig2.add_trace(go.Scatter(x=t["date"], y=t["leverage"], name="ND/EBITDA", mode="lines+markers", yaxis="y2"))
            fig2.update_layout(
                yaxis=dict(title="Margin", tickformat=".0%"),
                yaxis2=dict(title="Leverage (x)", overlaying="y", side="right"),
                legend_title_text="",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Margin/leverage trend requires: date, revenue, ebitda, and net debt (or debt + cash).")

        # --- Cashflows timeline
        st.subheader("Cashflows timeline")

        cf_date = _pick_date_col(cf_c, ["date", "flow_date", "txn_date"])
        cf_amt  = _pick_num_col(cf_c, ["amount", "cashflow", "flow_amount"])
        cf_type = _first_existing_col(cf_c, ["type", "flow_type", "txn_type"])  # optional

        if cf_date and cf_amt and not cf_c.empty:
            cf_c[cf_date] = _coerce_dt(cf_c[cf_date])
            cf_c = cf_c.dropna(subset=[cf_date]).sort_values(cf_date)

            import plotly.express as px
            plot_df = cf_c[[cf_date, cf_amt] + ([cf_type] if cf_type else [])].copy()
            plot_df = plot_df.rename(columns={cf_date: "date", cf_amt: "amount"})
            fig3 = px.bar(plot_df, x="date", y="amount", color=cf_type if cf_type else None)
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Cashflows timeline requires capital_flows columns: date + amount.")

        # --- Last valuation mark
        st.subheader("Last valuation mark")

        val_date = _pick_date_col(val_c, ["as_of_date", "date", "period_end", "mark_date"])
        ev_col   = _pick_num_col(val_c, ["enterprise_value", "ev"])
        eq_col   = _pick_num_col(val_c, ["equity_value", "equity"])
        mult_col = _pick_num_col(val_c, ["ev_ebitda_multiple", "multiple", "ev_to_ebitda"])

        latest_val = _latest_row(val_c, val_date) if val_date else None
        if latest_val is not None:
            c2 = st.columns(4)
            c2[0].metric("Mark date", str(pd.to_datetime(latest_val[val_date]).date()))
            c2[1].metric("Enterprise value (EV)", _fmt_money(latest_val[ev_col]) if ev_col else "n/a")
            c2[2].metric("Equity value", _fmt_money(latest_val[eq_col]) if eq_col else "n/a")
            c2[3].metric("EV/EBITDA", _fmt_x(latest_val[mult_col]) if mult_col else "n/a")

            # optional: show the raw latest mark row
            with st.expander("Show valuation mark row", expanded=False):
                st.dataframe(pd.DataFrame([latest_val]).T, use_container_width=True)
        else:
            st.info("Valuation mark requires valuation columns: date + EV/equity/multiple (at least date + one value).")


# -----------------------------
# Standardize date columns + keys
# -----------------------------
fin_date_col = _first_existing_col(financials, ["period_end", "date", "quarter", "as_of", "report_date"])
if not fin_date_col:
    st.error("Financials is missing a date column (expected period_end/date/quarter/as_of/report_date).")
    st.stop()
financials = financials.copy()
financials[fin_date_col] = _coerce_dt(financials[fin_date_col])
financials = financials.dropna(subset=[fin_date_col])
if fin_date_col != "period_end":
    financials = financials.rename(columns={fin_date_col: "period_end"})

val_date_col = _first_existing_col(valuation, ["as_of_date", "as_of", "valuation_date", "date", "period_end"])
if not val_date_col:
    st.error("Valuation is missing a date column (expected as_of_date/as_of/valuation_date/date/period_end).")
    st.stop()
valuation = valuation.copy()
valuation[val_date_col] = _coerce_dt(valuation[val_date_col])
valuation = valuation.dropna(subset=[val_date_col])
if val_date_col != "as_of_date":
    valuation = valuation.rename(columns={val_date_col: "as_of_date"})

cf_date_col = _first_existing_col(capital_flows, ["date", "flow_date", "transaction_date"])
if not cf_date_col:
    st.error("Capital flows is missing a date column (expected date/flow_date/transaction_date).")
    st.stop()
capital_flows = capital_flows.copy()
capital_flows[cf_date_col] = _coerce_dt(capital_flows[cf_date_col])
capital_flows = capital_flows.dropna(subset=[cf_date_col])
if cf_date_col != "date":
    capital_flows = capital_flows.rename(columns={cf_date_col: "date"})

for df_name, df in [("companies", companies), ("financials", financials), ("valuation", valuation), ("capital_flows", capital_flows)]:
    if "company_id" not in df.columns:
        st.error(f"{df_name} is missing company_id.")
        st.stop()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

all_sectors = sorted(companies["sector"].dropna().unique().tolist()) if "sector" in companies.columns else []
selected_sectors = st.sidebar.multiselect("Sector", options=all_sectors, default=all_sectors, key="filter_sectors")

companies_s = companies[companies["sector"].isin(selected_sectors)].copy() if "sector" in companies.columns else companies.copy()
company_options = (
    companies_s[["company_id", "company_name"]].drop_duplicates().sort_values("company_name")
    if "company_name" in companies_s.columns
    else companies_s[["company_id"]].drop_duplicates()
)

selected_company_ids = st.sidebar.multiselect(
    "Company",
    options=company_options["company_id"].tolist(),
    default=company_options["company_id"].tolist(),
    format_func=(lambda cid: company_options.loc[company_options["company_id"] == cid, "company_name"].iloc[0])
    if "company_name" in company_options.columns
    else None,
    key="filter_companies",
)

# -----------------------------
# Sidebar: As-of + Trend window
# -----------------------------
latest_rep = _latest_available_reporting_date(financials, valuation)
default_asof = _to_quarter_end(latest_rep)

min_rep = min(financials["period_end"].min(), valuation["as_of_date"].min())
min_rep = pd.Timestamp(min_rep).normalize() if pd.notna(min_rep) else (default_asof - pd.DateOffset(years=5))

as_of_dt = st.sidebar.date_input(
    "As-of (snapshot) date",
    value=default_asof.date(),
    min_value=min_rep.date(),
    max_value=default_asof.date(),
)
as_of_ts = pd.Timestamp(as_of_dt).normalize()

trend_quarters = st.sidebar.slider("Trend window (quarters)", min_value=4, max_value=20, value=8, step=1)
trend_start_ts = (as_of_ts.to_period("Q") - trend_quarters + 1).start_time.normalize()

st.sidebar.caption(f"As-of: {as_of_ts.date():%Y-%m-%d}")
st.sidebar.caption(f"Trend: {trend_start_ts.date():%Y-%m-%d} → {as_of_ts.date():%Y-%m-%d}")

# -----------------------------
# Filtered frames (snapshot vs trend)
# -----------------------------
companies_f = companies[
    (companies["sector"].isin(selected_sectors) if "sector" in companies.columns else True)
    & companies["company_id"].isin(selected_company_ids)
].copy()

ids = set(companies_f["company_id"].unique())

financials_asof = financials[(financials["company_id"].isin(ids)) & (financials["period_end"] <= as_of_ts)].copy()
valuation_asof = valuation[(valuation["company_id"].isin(ids)) & (valuation["as_of_date"] <= as_of_ts)].copy()
capital_flows_asof = capital_flows[(capital_flows["company_id"].isin(ids)) & (capital_flows["date"] <= as_of_ts)].copy()

valuation_latest = (
    valuation_asof.sort_values(["company_id", "as_of_date"])
    .groupby("company_id", as_index=False)
    .tail(1)
    .copy()
)

financials_trend = financials_asof[financials_asof["period_end"] >= trend_start_ts].copy()
valuation_trend = valuation_asof[valuation_asof["as_of_date"] >= trend_start_ts].copy()
capital_flows_trend = capital_flows_asof[capital_flows_asof["date"] >= trend_start_ts].copy()

with st.expander("Debug: row counts (snapshot vs trend)"):
    st.write(
        {
            "companies_f": len(companies_f),
            "financials_asof": len(financials_asof),
            "financials_trend": len(financials_trend),
            "valuation_asof": len(valuation_asof),
            "valuation_latest": len(valuation_latest),
            "valuation_trend": len(valuation_trend),
            "capital_flows_asof": len(capital_flows_asof),
            "capital_flows_trend": len(capital_flows_trend),
            "as_of": str(as_of_ts.date()),
            "trend_start": str(trend_start_ts.date()),
        }
    )

# -----------------------------
# Header
# -----------------------------
st.title("PE Portfolio Performance Dashboard")
st.caption(f"As of {as_of_ts.date():%Y-%m-%d} | Trend window: {trend_start_ts.date():%Y-%m-%d} → {as_of_ts.date():%Y-%m-%d}")

# -----------------------------
# Fund KPIs (portfolio summary)
# -----------------------------
portfolio = build_portfolio_summary(capital_flows_asof, valuation_latest, as_of_ts)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_div(a, b):
    a = _safe_float(a)
    b = _safe_float(b)
    if pd.isna(a) or pd.isna(b) or b == 0:
        return float("nan")
    return a / b


paid_in = _safe_float(portfolio.get("paid_in"))
dists = _safe_float(portfolio.get("distributions"))
nav = _safe_float(portfolio.get("nav"))
tvpi = _safe_float(portfolio.get("tvpi"))
dpi = _safe_float(portfolio.get("dpi"))
irr = _safe_float(portfolio.get("irr"))

rvpi = _safe_div(nav, paid_in)
num_companies = len(ids)
pct_realized = _safe_div(dpi, tvpi)

top3_conc = float("nan")
if not valuation_latest.empty and "equity_value" in valuation_latest.columns:
    v = valuation_latest.copy()
    v["equity_value"] = pd.to_numeric(v["equity_value"], errors="coerce")
    v = v.dropna(subset=["equity_value"])
    total_nav = v["equity_value"].sum()
    if total_nav and total_nav > 0:
        top3 = v.sort_values("equity_value", ascending=False).head(3)["equity_value"].sum()
        top3_conc = top3 / total_nav

# Health metrics
if not financials_asof.empty:
    fin_latest = (
        financials_asof.sort_values("period_end")
        .groupby("company_id", as_index=False)
        .tail(1)
        .copy()
    )
    missing_latest_fin = int((fin_latest["period_end"].dt.normalize() != as_of_ts.normalize()).sum())
else:
    fin_latest = pd.DataFrame(columns=["company_id", "period_end"])
    missing_latest_fin = num_companies

if not valuation_latest.empty:
    missing_latest_val = int((valuation_latest["as_of_date"].dt.normalize() != as_of_ts.normalize()).sum())
else:
    missing_latest_val = num_companies

missing_any_fin = int(len(ids - set(fin_latest["company_id"].unique()))) if "company_id" in fin_latest.columns else num_companies
missing_any_val = int(len(ids - set(valuation_latest["company_id"].unique()))) if "company_id" in valuation_latest.columns else num_companies

avg_leverage = float("nan")
if not financials_asof.empty and ("net_debt" in financials_asof.columns) and ("ebitda" in financials_asof.columns):
    fl = fin_latest.copy()
    fl["net_debt"] = pd.to_numeric(fl["net_debt"], errors="coerce")
    fl["ebitda"] = pd.to_numeric(fl["ebitda"], errors="coerce")
    fl["leverage_x"] = fl["net_debt"] / fl["ebitda"]
    lev = fl["leverage_x"].replace([pd.NA, pd.NaT, float("inf"), -float("inf")], pd.NA).dropna()
    if not lev.empty:
        avg_leverage = float(lev.mean())

# KPI Row 1
r1 = st.columns(7)
with r1[0]:
    st.metric("Paid-in (USD mm)", _fmt_mm(paid_in))
    st.caption("Total contributed capital (as-of).")
with r1[1]:
    st.metric("Distributions (USD mm)", _fmt_mm(dists))
    st.caption("Capital returned to LPs (as-of).")
with r1[2]:
    st.metric("NAV (USD mm)", _fmt_mm(nav))
    st.caption("Unrealized value at latest mark ≤ as-of.")
with r1[3]:
    st.metric("TVPI", _fmt_x(tvpi))
    st.caption("(Distributions + NAV) / Paid-in.")
with r1[4]:
    st.metric("DPI", _fmt_x(dpi))
    st.caption("Distributions / Paid-in.")
with r1[5]:
    st.metric("RVPI", _fmt_x(rvpi))
    st.caption("NAV / Paid-in.")
with r1[6]:
    st.metric("IRR", _fmt_pct(irr))
    st.caption("XIRR using cashflows to as-of (and terminal NAV if implemented).")

st.write("")

# KPI Row 1b
r1b = st.columns(3)
with r1b[0]:
    st.metric("# Companies", f"{num_companies:,}")
    st.caption("Count in current filters.")
with r1b[1]:
    st.metric("% Realized", _fmt_pct(pct_realized))
    st.caption("Proxy: DPI / TVPI.")
with r1b[2]:
    st.metric("Top 3 Concentration", _fmt_pct(top3_conc))
    st.caption("Share of NAV in top 3 positions.")

st.divider()

# KPI Row 2
st.subheader("Data Health & Portfolio Health")
h = st.columns(5)
with h[0]:
    st.metric("# Active Companies", f"{num_companies:,}")
    st.caption("Companies in scope (filters applied).")
with h[1]:
    st.metric("# Missing any financials", f"{missing_any_fin:,}")
    st.caption("No financial rows ≤ as-of.")
with h[2]:
    st.metric("# Missing any valuation", f"{missing_any_val:,}")
    st.caption("No valuation marks ≤ as-of.")
with h[3]:
    st.metric("# Missing latest financials", f"{int(missing_latest_fin):,}")
    st.caption("Latest financial period != as-of quarter-end.")
with h[4]:
    st.metric("Avg Leverage (x)", "n/a" if pd.isna(avg_leverage) else f"{avg_leverage:.2f}x")
    st.caption("Simple avg Net Debt / EBITDA (latest period).")

st.divider()


# -----------------------------
# Derived metrics for IC memo
# -----------------------------
def compute_margin_change(financials_asof: pd.DataFrame) -> pd.DataFrame:
    if financials_asof.empty:
        return pd.DataFrame(columns=["company_id", "margin_change"])

    needed = {"company_id", "period_end", "revenue", "ebitda"}
    if not needed.issubset(set(financials_asof.columns)):
        return pd.DataFrame(columns=["company_id", "margin_change"])

    f = financials_asof.copy()
    f["revenue"] = pd.to_numeric(f["revenue"], errors="coerce")
    f["ebitda"] = pd.to_numeric(f["ebitda"], errors="coerce")
    f["margin_calc"] = f["ebitda"] / f["revenue"].replace({0: pd.NA})

    f = f.sort_values(["company_id", "period_end"])
    last2 = f.groupby("company_id", as_index=False).tail(2)

    def _chg(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("period_end")
        if len(g) < 2:
            return pd.Series({"margin_change": pd.NA})
        return pd.Series({"margin_change": g["margin_calc"].iloc[-1] - g["margin_calc"].iloc[-2]})

    out = last2.groupby("company_id").apply(_chg).reset_index()
    # groupby+apply can create an extra column depending on pandas version
    if "level_1" in out.columns:
        out = out.drop(columns=["level_1"])
    return out


# -----------------------------
# Company summary
# -----------------------------
cs = build_company_summary(
    companies_f=companies_f,
    financials_f=financials_asof,
    valuation_f=valuation_asof,
    capital_flows_f=capital_flows_asof,
    start_dt=trend_start_ts,
    end_dt=as_of_ts,
)

mchg = compute_margin_change(financials_asof)
if not mchg.empty:
    cs = cs.merge(mchg, on="company_id", how="left")
else:
    cs["margin_change"] = pd.NA

if not valuation_latest.empty and "equity_value" in valuation_latest.columns:
    nav_map = valuation_latest[["company_id", "equity_value"]].copy()
    nav_map["equity_value"] = pd.to_numeric(nav_map["equity_value"], errors="coerce")
    nav_map["nav_mm"] = nav_map["equity_value"].map(_money_mm)
    cs = cs.merge(nav_map[["company_id", "nav_mm"]], on="company_id", how="left")
else:
    cs["nav_mm"] = pd.NA


# =============================
# Page Navigation
# =============================
PAGES = [
    "Overview",
    "Performance (Fund + Company)",
    "Operating",
    "Valuation",
    "Liquidity",
    "Risk",
    "Data Health",
]
page = st.sidebar.radio("Page", PAGES, index=0, key="page_nav")


# =============================
# Page renderers
# =============================
def render_risk(embedded: bool = False):
    if not embedded:
        st.subheader("Risk View")

    if cs.empty:
        st.caption("No risk view available (no companies in current filters).")
        return

    risk = build_risk_flags(cs, leverage_hi=6.0, margin_lo=0.15)
    rr1, rr2 = st.columns([2, 3])

    with rr1:
        st.caption("Leverage (Net Debt / EBITDA) — latest period")

        if "leverage" not in cs.columns:
            st.caption("Leverage column not available.")
        else:
            df = cs.copy()
            df["leverage"] = pd.to_numeric(df["leverage"], errors="coerce")
            df = df.dropna(subset=["leverage"]).sort_values("leverage", ascending=False)

            if df.empty:
                st.caption("No leverage data available.")
            else:
                label_col = "company_name" if "company_name" in df.columns else "company_id"
                TOP_N = 15
                top = df.head(TOP_N).copy()

                chart_key = "pe_overview_risk_leverage_h" if embedded else "pe_risk_leverage_h"

                _bar_h_x(
                    top,
                    label_col=label_col,
                    value_col="leverage",
                    value_name="Leverage",
                    key=chart_key,
                )

                if len(df) > TOP_N:
                    with st.expander(f"Show remaining companies ({len(df) - TOP_N})"):
                        st.dataframe(
                            df.iloc[TOP_N:][_df_cols(df, [label_col, "sector", "leverage", "margin_latest", "moic"])],
                            use_container_width=True,
                            hide_index=True,
                        )

    with rr2:
        st.caption("Risk Flags")
        if risk is None or risk.empty:
            st.caption("No risk flags available.")
        else:
            st.dataframe(
                risk[_df_cols(risk, ["company_name", "sector", "leverage", "margin_latest", "high_leverage_flag", "low_margin_flag"])],
                use_container_width=True,
                hide_index=True,
            )


def render_overview():
    # A) narrative
    try:
        summary_box = st.container(border=True)
    except TypeError:
        summary_box = st.container()

    with summary_box:
        st.subheader("This period summary")
        lines = build_key_insights(cs) if not cs.empty else []
        if not lines:
            st.caption("No insights available for the current filters.")
        else:
            for line in lines[:4]:
                st.write(f"• {line}")

    st.divider()

    # B) exposure
    st.subheader("Concentration & Exposure")

    if valuation_latest.empty or ("equity_value" not in valuation_latest.columns):
        st.caption("No valuation marks available as of the selected date.")
    else:
        v_last = valuation_latest.copy()
        v_last["equity_value"] = pd.to_numeric(v_last["equity_value"], errors="coerce")
        v_last = v_last.dropna(subset=["equity_value"])

        merge_cols = _df_cols(companies_f, ["company_id", "company_name", "sector"])
        v_last = v_last.merge(companies_f[merge_cols], on="company_id", how="left")
        v_last["nav_mm"] = v_last["equity_value"].map(_money_mm)

        c1, c2 = st.columns(2)

        with c1:
            st.caption("NAV by Sector (USD mm)")
            if "sector" not in v_last.columns:
                st.caption("No sector column available.")
            else:
                nav_sector = v_last.groupby("sector", as_index=False)["nav_mm"].sum().sort_values("nav_mm", ascending=False)
                _pie_usdmm(nav_sector, "sector", "nav_mm", key="pe_overview_nav_sector")

        with c2:
            region_col = _first_existing_col(companies_f, ["region", "geography", "geo", "country", "hq_region"])
            st.caption("NAV by Region (USD mm)")
            if not region_col:
                st.caption("No region column found in companies.")
            else:
                v_reg = v_last.merge(companies_f[["company_id", region_col]], on="company_id", how="left")
                nav_region = v_reg.groupby(region_col, as_index=False)["nav_mm"].sum().sort_values("nav_mm", ascending=False)
                _pie_usdmm(nav_region, region_col, "nav_mm", key="pe_overview_nav_region")

        st.caption("Top 10 NAV Concentration (USD mm)")
        top10 = v_last.sort_values("nav_mm", ascending=False).head(10).copy()
        if top10.empty:
            st.caption("No NAV values to display.")
        else:
            label_col = "company_name" if "company_name" in top10.columns else "company_id"
            top10 = top10.sort_values("nav_mm", ascending=True)
            _bar_h_usdmm(top10, y_col=label_col, x_col="nav_mm", key="pe_overview_top10_nav", name="NAV")

    st.divider()

    # Snapshot performance view (FIX: horizontal bars to avoid label collisions)
    st.subheader("Performance Snapshot")
    c1, c2 = st.columns(2)

    with c1:
        st.caption("MOIC by Company")
        if cs.empty or ("moic" not in cs.columns) or cs["moic"].dropna().empty:
            st.caption("No MOIC data for the current filters.")
        else:
            df = cs.copy()
            df["moic"] = pd.to_numeric(df["moic"], errors="coerce")
            df = df.dropna(subset=["moic"]).sort_values("moic", ascending=False)

            label_col = "company_name" if "company_name" in df.columns else "company_id"
            TOP_N = 15
            top = df.head(TOP_N)

            _bar_h_x(
                top,
                label_col=label_col,
                value_col="moic",
                value_name="MOIC",
                key="pe_overview_moic_h",
            )

            if len(df) > TOP_N:
                with st.expander(f"Show remaining companies ({len(df) - TOP_N})"):
                    st.dataframe(
                        df.iloc[TOP_N:][_df_cols(df, [label_col, "sector", "moic", "irr", "nav_mm"])],
                        use_container_width=True,
                        hide_index=True,
                    )

    with c2:
        st.caption("Company Comparison Table")
        if cs.empty:
            st.caption("No company data available for the current filters.")
        else:
            show = cs.copy()
            if "irr" in show.columns:
                show["irr"] = pd.to_numeric(show["irr"], errors="coerce")
            desired = [
                "company_name",
                "sector",
                "latest_period",
                "revenue_latest",
                "ebitda_latest",
                "margin_latest",
                "margin_change",
                "leverage",
                "nav_mm",
                "moic",
                "irr",
            ]
            st.dataframe(show[_df_cols(show, desired)], use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Winners & Laggards")
    if cs.empty:
        st.caption("No company data available for the current filters.")
    else:
        tmp = cs.copy()
        for col in ["moic", "irr", "leverage", "margin_change", "margin_latest", "nav_mm"]:
            if col in tmp.columns:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

        w1, w2 = st.columns(2)

        with w1:
            st.caption("Top 5 Winners — MOIC")
            winners_moic = tmp.dropna(subset=["moic"]).sort_values("moic", ascending=False).head(5)
            st.dataframe(winners_moic[_df_cols(winners_moic, ["company_name", "sector", "nav_mm", "moic", "irr"])], use_container_width=True, hide_index=True)

            st.caption("Top 5 Winners — IRR")
            winners_irr = tmp.dropna(subset=["irr"]).sort_values("irr", ascending=False).head(5)
            st.dataframe(winners_irr[_df_cols(winners_irr, ["company_name", "sector", "nav_mm", "irr", "moic"])], use_container_width=True, hide_index=True)

        with w2:
            st.caption("Bottom 5 Laggards — EBITDA Margin Δ")
            lag_margin = tmp.dropna(subset=["margin_change"]).sort_values("margin_change", ascending=True).head(5)
            st.dataframe(lag_margin[_df_cols(lag_margin, ["company_name", "sector", "margin_latest", "margin_change", "leverage"])], use_container_width=True, hide_index=True)

            st.caption("Bottom 5 Laggards — Highest Leverage")
            lag_lev = tmp.dropna(subset=["leverage"]).sort_values("leverage", ascending=False).head(5)
            st.dataframe(lag_lev[_df_cols(lag_lev, ["company_name", "sector", "leverage", "margin_latest", "moic"])], use_container_width=True, hide_index=True)

    st.divider()

    st.subheader("Risk View")
    render_risk(embedded=True)


def render_performance():
    st.subheader("Company Comparison")
    if cs.empty:
        st.caption("No company data available for the current filters.")
        return

    cols = _df_cols(
        cs,
        ["company_name", "sector", "latest_period", "revenue_latest", "ebitda_latest", "margin_latest", "margin_change", "leverage", "nav_mm", "moic", "irr"],
    )
    st.dataframe(cs[cols], use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("MOIC by Company")
        df = cs.copy()
        if "moic" not in df.columns:
            st.caption("MOIC column not available.")
        else:
            df["moic"] = pd.to_numeric(df["moic"], errors="coerce")
            df = df.dropna(subset=["moic"]).sort_values("moic", ascending=False)
            label_col = "company_name" if "company_name" in df.columns else "company_id"
            _bar_h_x(df.head(20), label_col=label_col, value_col="moic", value_name="MOIC", key="pe_perf_moic_h")

    with c2:
        st.subheader("IRR by Company")
        df = cs.copy()
        if "irr" not in df.columns:
            st.caption("IRR column not available.")
        else:
            df["irr"] = pd.to_numeric(df["irr"], errors="coerce")
            df = df.dropna(subset=["irr"]).sort_values("irr", ascending=False)
            label_col = "company_name" if "company_name" in df.columns else "company_id"
            _bar_h_pct(df.head(20), label_col=label_col, value_col="irr", value_name="IRR", key="pe_perf_irr_h")


def render_operating():
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Portfolio Revenue Trend (USD mm)")
        if financials_trend.empty or "revenue" not in financials_trend.columns:
            st.caption("No revenue in the trend window.")
        else:
            df = financials_trend.copy()
            df["q_end"] = _q_end(df["period_end"])
            df["revenue_mm"] = _to_mm(df["revenue"])

            q = df.groupby("q_end", as_index=False)["revenue_mm"].sum().sort_values("q_end")
            q["q_label"] = _q_label(q["q_end"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=q["q_label"], y=q["revenue_mm"], mode="lines+markers", hovertemplate=_hover_mm("Revenue"), name="Revenue"))
            fig.update_yaxes(title_text="USD mm")
            fig = _pe_layout(fig, height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="pe_ops_revenue_q")

    with c2:
        st.subheader("Portfolio EBITDA Trend (USD mm)")
        if financials_trend.empty or "ebitda" not in financials_trend.columns:
            st.caption("No EBITDA in the trend window.")
        else:
            df = financials_trend.copy()
            df["q_end"] = _q_end(df["period_end"])
            df["ebitda_mm"] = _to_mm(df["ebitda"])

            q = df.groupby("q_end", as_index=False)["ebitda_mm"].sum().sort_values("q_end")
            q["q_label"] = _q_label(q["q_end"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=q["q_label"], y=q["ebitda_mm"], mode="lines+markers", hovertemplate=_hover_mm("EBITDA"), name="EBITDA"))
            fig.update_yaxes(title_text="USD mm")
            fig = _pe_layout(fig, height=420, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, key="pe_ops_ebitda_q")


def render_liquidity():
    st.subheader("Cashflows: Paid-in vs Distributions (USD mm)")

    if capital_flows_trend.empty:
        st.caption("No cashflows in the trend window.")
        return
    if "flow_type" not in capital_flows_trend.columns:
        st.caption("capital_flows is missing flow_type.")
        return

    cf = capital_flows_trend[capital_flows_trend["flow_type"].isin(["Contribution", "Distribution"])].copy()
    if cf.empty:
        st.caption("No Contribution/Distribution cashflows in the trend window.")
        return

    cf["date"] = pd.to_datetime(cf["date"], errors="coerce")
    cf["amount"] = pd.to_numeric(cf["amount"], errors="coerce")
    cf = cf.dropna(subset=["date", "amount"])

    cf["amount_mm"] = cf["amount"].map(_money_mm)
    cf["paid_in_mm"] = 0.0
    cf["dist_mm"] = 0.0
    cf.loc[cf["flow_type"] == "Contribution", "paid_in_mm"] = -cf.loc[cf["flow_type"] == "Contribution", "amount_mm"]
    cf.loc[cf["flow_type"] == "Distribution", "dist_mm"] = cf.loc[cf["flow_type"] == "Distribution", "amount_mm"]

    cf["q_end"] = _q_end(cf["date"])
    q = cf.groupby("q_end", as_index=False)[["paid_in_mm", "dist_mm"]].sum().sort_values("q_end")
    q["net_mm"] = q["dist_mm"] - q["paid_in_mm"]
    q["q_label"] = _q_label(q["q_end"])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=q["q_label"], y=q["paid_in_mm"], name="Paid-in", hovertemplate=_hover_mm("Paid-in")))
    fig.add_trace(go.Bar(x=q["q_label"], y=q["dist_mm"], name="Distributions", hovertemplate=_hover_mm("Distributions")))
    fig.add_trace(go.Scatter(x=q["q_label"], y=q["net_mm"], name="Net", mode="lines+markers", hovertemplate=_hover_mm("Net")))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="USD mm")
    fig = _pe_layout(fig, height=420, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, key="pe_liq_cashflows_bars_net")

    q2 = q.copy()
    q2["cum_paid_in"] = q2["paid_in_mm"].cumsum()
    q2["cum_dists"] = q2["dist_mm"].cumsum()

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=q2["q_label"], y=q2["cum_paid_in"], name="Cumulative Paid-in", mode="lines", hovertemplate=_hover_mm("Cumulative Paid-in")))
    fig2.add_trace(go.Scatter(x=q2["q_label"], y=q2["cum_dists"], name="Cumulative Distributions", mode="lines", hovertemplate=_hover_mm("Cumulative Distributions")))
    fig2.update_yaxes(title_text="USD mm")
    fig2 = _pe_layout(fig2, height=380, showlegend=True)
    st.plotly_chart(fig2, use_container_width=True, key="pe_liq_cashflows_cumulative")


def render_valuation():
    st.subheader("Value Creation Decomposition (Company)")

    if cs.empty:
        st.caption("No companies in current filters.")
        return

    pick = st.selectbox(
        "Select company",
        options=cs["company_id"].tolist(),
        format_func=lambda cid: cs.loc[cs["company_id"] == cid, ("company_name" if "company_name" in cs.columns else "company_id")].iloc[0],
        key="pe_val_company_select",
    )

    bridge = value_creation_decomposition(pick, financials_trend, valuation_trend)
    if bridge is None:
        st.caption("Not enough data (need financials + valuation marks in the trend window).")
        return

    x = ["Entry Equity", "EBITDA Growth", "Multiple Change", "Deleveraging", "Current Equity"]
    y = [
        _money_mm(bridge.get("entry_equity")),
        _money_mm(bridge.get("ebitda_growth")),
        _money_mm(bridge.get("multiple_change")),
        _money_mm(bridge.get("deleveraging")),
        _money_mm(bridge.get("current_equity")),
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, name="USD mm", hovertemplate=_hover_mm("Value")))
    fig.update_yaxes(title_text="USD mm")
    fig = _pe_layout(fig, height=420, showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key="pe_bridge_value_creation")


def render_data_health():
    st.subheader("Data Health")
    st.write(
        {
            "Active companies": num_companies,
            "Missing any financials (<= as-of)": missing_any_fin,
            "Missing any valuation marks (<= as-of)": missing_any_val,
            "Missing latest financials (period != as-of)": int(missing_latest_fin),
            "Missing latest valuations (mark date != as-of)": int(missing_latest_val),
        }
    )

    st.divider()
    st.subheader("Company-level gaps")

    have_fin = set(fin_latest["company_id"].unique()) if ("company_id" in fin_latest.columns) else set()
    have_val = set(valuation_latest["company_id"].unique()) if ("company_id" in valuation_latest.columns) else set()

    miss_fin_ids = sorted(list(ids - have_fin))
    miss_val_ids = sorted(list(ids - have_val))

    c1, c2 = st.columns(2)

    with c1:
        st.caption("Missing any financials (<= as-of)")
        if not miss_fin_ids:
            st.caption("None.")
        else:
            df = companies_f[companies_f["company_id"].isin(miss_fin_ids)]
            st.dataframe(df[_df_cols(df, ["company_id", "company_name", "sector"])], use_container_width=True, hide_index=True)

    with c2:
        st.caption("Missing any valuation marks (<= as-of)")
        if not miss_val_ids:
            st.caption("None.")
        else:
            df = companies_f[companies_f["company_id"].isin(miss_val_ids)]
            st.dataframe(df[_df_cols(df, ["company_id", "company_name", "sector"])], use_container_width=True, hide_index=True)


# =============================
# Render selected page
# =============================
st.header(page)

if page == "Overview":
    render_overview()
elif page == "Performance (Fund + Company)":
    render_performance()
elif page == "Operating":
    render_operating()
elif page == "Valuation":
    render_valuation()
elif page == "Liquidity":
    render_liquidity()
elif page == "Risk":
    render_risk()
elif page == "Data Health":
    render_data_health()
else:
    st.caption("Page not implemented.")
