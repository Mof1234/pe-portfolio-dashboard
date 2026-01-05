# src/metrics.py
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# -----------------------------
# XIRR (dated IRR) utilities
# -----------------------------
def _year_frac(d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    return (d1 - d0).days / 365.0


def xnpv(rate: float, cashflows: pd.DataFrame) -> float:
    d0 = cashflows["date"].iloc[0]
    t = cashflows["date"].map(lambda d: _year_frac(d0, d)).to_numpy()
    a = cashflows["amount"].to_numpy()
    return float(np.sum(a / (1.0 + rate) ** t))


def xirr(cashflows: pd.DataFrame) -> float:
    cf = cashflows.dropna(subset=["date", "amount"]).copy()
    if cf.empty:
        return float("nan")
    if not ((cf["amount"] < 0).any() and (cf["amount"] > 0).any()):
        return float("nan")

    cf = cf.sort_values("date").reset_index(drop=True)

    # Newton
    rate = 0.15
    for _ in range(60):
        f = xnpv(rate, cf)
        eps = 1e-6
        f1 = xnpv(rate + eps, cf)
        df = (f1 - f) / eps
        if abs(df) < 1e-10:
            break
        new_rate = rate - f / df
        if not np.isfinite(new_rate):
            break
        new_rate = max(new_rate, -0.9999)
        if abs(new_rate - rate) < 1e-7:
            return float(new_rate)
        rate = new_rate

    # Bisection fallback
    lo, hi = -0.9, 5.0
    f_lo, f_hi = xnpv(lo, cf), xnpv(hi, cf)
    if np.sign(f_lo) == np.sign(f_hi):
        return float("nan")

    for _ in range(100):
        mid = (lo + hi) / 2.0
        f_mid = xnpv(mid, cf)
        if abs(f_mid) < 1e-6:
            return float(mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid

    return float((lo + hi) / 2.0)


# -----------------------------
# Portfolio summary (FILTERED)
# -----------------------------
def build_portfolio_summary(
    capital_flows_f: pd.DataFrame,
    valuation_f: pd.DataFrame,
    as_of_dt: pd.Timestamp,
) -> Dict[str, float]:
    """
    Fund-level KPIs using:
      - Paid-in = -sum(Contribution cashflows)
      - Distributions = sum(Distribution cashflows)
      - NAV = latest equity_value per company as of as_of_dt (from valuation marks)
      - TVPI = (Distributions + NAV) / Paid-in
      - DPI  = Distributions / Paid-in
      - RVPI = NAV / Paid-in
      - IRR  = XIRR(cashflows + terminal NAV)

    Important:
      We treat only Contribution/Distribution as cashflows.
      Any flow_type like NAV/Unrealized Value should NOT be included in cashflows.
    """
    as_of_dt = pd.to_datetime(as_of_dt)

    # --- 1) Restrict to cashflow rows only ---
    cf_cash = capital_flows_f[
        capital_flows_f["flow_type"].isin(["Contribution", "Distribution"])
    ][["date", "flow_type", "amount"]].copy()

    if not cf_cash.empty:
        cf_cash["date"] = pd.to_datetime(cf_cash["date"], errors="coerce")
        cf_cash["amount"] = pd.to_numeric(cf_cash["amount"], errors="coerce")
        cf_cash = cf_cash.dropna(subset=["date", "amount"])
        cf_cash = cf_cash[cf_cash["date"] <= as_of_dt]
    else:
        cf_cash = pd.DataFrame(columns=["date", "flow_type", "amount"])

    paid_in = -cf_cash.loc[cf_cash["flow_type"] == "Contribution", "amount"].sum()
    dists = cf_cash.loc[cf_cash["flow_type"] == "Distribution", "amount"].sum()

    # --- 2) NAV from valuation marks (latest per company as-of date) ---
    nav = 0.0
    if not valuation_f.empty:
        v = valuation_f.copy()
        v["as_of_date"] = pd.to_datetime(v["as_of_date"], errors="coerce")
        v["equity_value"] = pd.to_numeric(v["equity_value"], errors="coerce")
        v = v.dropna(subset=["company_id", "as_of_date", "equity_value"])
        v = v[v["as_of_date"] <= as_of_dt].sort_values(["company_id", "as_of_date"])
        if not v.empty:
            nav = float(v.groupby("company_id", as_index=False).tail(1)["equity_value"].sum())

    # --- 3) Multiples ---
    dpi = (dists / paid_in) if paid_in > 0 else float("nan")
    rvpi = (nav / paid_in) if paid_in > 0 else float("nan")
    tvpi = ((dists + nav) / paid_in) if paid_in > 0 else float("nan")

    # --- 4) Fund IRR (cashflows + terminal NAV) ---
    irr = float("nan")
    if (paid_in > 0) and (not cf_cash.empty or nav != 0.0):
        cf = cf_cash[["date", "amount"]].copy()
        cf = cf.groupby("date", as_index=False)["amount"].sum()

        terminal = pd.DataFrame({"date": [as_of_dt], "amount": [nav]})
        cf2 = pd.concat([cf, terminal], ignore_index=True).groupby("date", as_index=False)["amount"].sum()
        cf2 = cf2.sort_values("date")

        irr = xirr(cf2)

    return {
        "paid_in": float(paid_in),
        "distributions": float(dists),
        "nav": float(nav),
        "dpi": float(dpi),
        "rvpi": float(rvpi),
        "tvpi": float(tvpi),
        "irr": float(irr),
        "total_value": float(dists + nav),
    }


# -----------------------------
# Company summary table
# -----------------------------
def _cagr(first: float, last: float, yrs: float) -> float:
    if yrs <= 0:
        return float("nan")
    if first is None or last is None:
        return float("nan")
    if pd.isna(first) or pd.isna(last):
        return float("nan")
    if first <= 0 or last <= 0:
        return float("nan")
    return (last / first) ** (1.0 / yrs) - 1.0


def build_company_summary(
    companies_f: pd.DataFrame,
    financials_f: pd.DataFrame,
    valuation_f: pd.DataFrame,
    capital_flows_f: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> pd.DataFrame:
    base = companies_f[["company_id", "company_name", "sector"]].drop_duplicates().copy()

    # Financial first/latest within filtered range
    fin = financials_f.sort_values(["company_id", "period_end"]).copy()
    fin_first = fin.groupby("company_id", as_index=False).head(1).rename(
        columns={"period_end": "entry_period", "revenue": "revenue_first", "ebitda": "ebitda_first"}
    )
    fin_latest = fin.groupby("company_id", as_index=False).tail(1).rename(
        columns={"period_end": "latest_period", "revenue": "revenue_latest", "ebitda": "ebitda_latest", "net_debt": "net_debt_latest"}
    )

    # Valuation first/latest within filtered range
    val = valuation_f.sort_values(["company_id", "as_of_date"]).copy()
    val_first = val.groupby("company_id", as_index=False).head(1).rename(
        columns={"equity_value": "equity_value_entry", "ev_to_ebitda": "multiple_entry"}
    )
    val_latest = val.groupby("company_id", as_index=False).tail(1).rename(
        columns={"equity_value": "equity_value_latest", "ev_to_ebitda": "multiple_latest"}
    )

    # Cashflow sums
    cf = capital_flows_f.copy()
    contrib = (
        cf[cf["flow_type"] == "Contribution"]
        .groupby("company_id", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "contribution_sum"})  # negative
    )
    dist = (
        cf[cf["flow_type"] == "Distribution"]
        .groupby("company_id", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "distribution_sum"})  # positive
    )

    out = (
        base.merge(fin_latest[["company_id", "latest_period", "revenue_latest", "ebitda_latest", "net_debt_latest"]], on="company_id", how="left")
        .merge(fin_first[["company_id", "entry_period", "revenue_first", "ebitda_first"]], on="company_id", how="left")
        .merge(val_latest[["company_id", "equity_value_latest", "multiple_latest"]], on="company_id", how="left")
        .merge(val_first[["company_id", "equity_value_entry", "multiple_entry"]], on="company_id", how="left")
        .merge(contrib, on="company_id", how="left")
        .merge(dist, on="company_id", how="left")
    )

    out["contribution_sum"] = out["contribution_sum"].fillna(0.0)
    out["distribution_sum"] = out["distribution_sum"].fillna(0.0)

    out["margin_latest"] = np.where(out["revenue_latest"] > 0, out["ebitda_latest"] / out["revenue_latest"], np.nan)
    out["leverage"] = np.where(out["ebitda_latest"] != 0, out["net_debt_latest"] / out["ebitda_latest"], np.nan)

    years = max((pd.to_datetime(end_dt) - pd.to_datetime(start_dt)).days / 365.0, 0.0)
    out["revenue_cagr"] = [_cagr(a, b, years) for a, b in zip(out["revenue_first"], out["revenue_latest"])]
    out["ebitda_cagr"] = [_cagr(a, b, years) for a, b in zip(out["ebitda_first"], out["ebitda_latest"])]

    paid_in = -out["contribution_sum"]
    total_value = out["distribution_sum"] + out["equity_value_latest"]
    out["moic"] = np.where(paid_in > 0, total_value / paid_in, np.nan)

    # Company IRR (cashflows + terminal equity at end_dt)
    irr_list = []
    for cid in out["company_id"].astype(str):
        ccf = capital_flows_f[capital_flows_f["company_id"] == cid][["date", "amount"]].copy()
        ccf = ccf.groupby("date", as_index=False)["amount"].sum()

        term = out.loc[out["company_id"] == cid, "equity_value_latest"]
        terminal_val = float(term.iloc[0]) if len(term) and pd.notna(term.iloc[0]) else 0.0

        terminal = pd.DataFrame({"date": [pd.to_datetime(end_dt)], "amount": [terminal_val]})
        ccf2 = pd.concat([ccf, terminal], ignore_index=True).sort_values("date")

        irr_list.append(xirr(ccf2))

    out["irr"] = irr_list
    return out


# -----------------------------
# Value creation bridge
# -----------------------------
def value_creation_decomposition(
    company_id: str,
    financials_f: pd.DataFrame,
    valuation_f: pd.DataFrame,
) -> Optional[Dict[str, float]]:
    fin = financials_f[financials_f["company_id"] == company_id].sort_values("period_end")
    val = valuation_f[valuation_f["company_id"] == company_id].sort_values("as_of_date")

    if fin.empty or val.empty:
        return None

    fin_entry, fin_curr = fin.iloc[0], fin.iloc[-1]
    val_entry, val_curr = val.iloc[0], val.iloc[-1]

    ebitda_entry = float(fin_entry["ebitda"]) if pd.notna(fin_entry["ebitda"]) else 0.0
    ebitda_curr = float(fin_curr["ebitda"]) if pd.notna(fin_curr["ebitda"]) else 0.0

    mult_entry = float(val_entry["ev_to_ebitda"]) if pd.notna(val_entry["ev_to_ebitda"]) else 0.0
    mult_curr = float(val_curr["ev_to_ebitda"]) if pd.notna(val_curr["ev_to_ebitda"]) else 0.0

    nd_entry = float(fin_entry["net_debt"]) if pd.notna(fin_entry["net_debt"]) else 0.0
    nd_curr = float(fin_curr["net_debt"]) if pd.notna(fin_curr["net_debt"]) else 0.0

    ev_entry = ebitda_entry * mult_entry
    ev_curr = ebitda_curr * mult_curr

    eq_entry = ev_entry - nd_entry
    eq_curr = ev_curr - nd_curr

    ebitda_growth = (ebitda_curr - ebitda_entry) * mult_entry
    multiple_change = (mult_curr - mult_entry) * ebitda_curr
    deleveraging = (nd_entry - nd_curr)

    return {
        "entry_equity": eq_entry,
        "ebitda_growth": ebitda_growth,
        "multiple_change": multiple_change,
        "deleveraging": deleveraging,
        "current_equity": eq_curr,
    }
def build_risk_flags(cs: pd.DataFrame, leverage_hi: float = 6.0, margin_lo: float = 0.15) -> pd.DataFrame:
    if cs.empty:
        return cs.copy()

    out = cs[["company_id", "company_name", "sector", "leverage", "margin_latest"]].copy()

    out["high_leverage_flag"] = out["leverage"].apply(
        lambda x: "High" if pd.notna(x) and float(x) >= leverage_hi else "OK"
    )
    out["low_margin_flag"] = out["margin_latest"].apply(
        lambda x: "Low" if pd.notna(x) and float(x) <= margin_lo else "OK"
    )

    # sort: most concerning first
    out = out.sort_values(["high_leverage_flag", "low_margin_flag", "leverage"], ascending=[True, True, False])
    return out


def build_key_insights(cs: pd.DataFrame) -> list[str]:
    if cs.empty:
        return ["No insights available for the current filter range."]

    insights = []

    moic_ok = cs.dropna(subset=["moic"])
    if not moic_ok.empty:
        top = moic_ok.sort_values("moic", ascending=False).iloc[0]
        insights.append(f"Top MOIC: {top['company_name']} ({top['moic']:.2f}x)")

    lev_ok = cs.dropna(subset=["leverage"])
    if not lev_ok.empty:
        top = lev_ok.sort_values("leverage", ascending=False).iloc[0]
        insights.append(f"Highest leverage: {top['company_name']} ({top['leverage']:.1f}x Net Debt/EBITDA)")

    margin_ok = cs.dropna(subset=["margin_latest"])
    if not margin_ok.empty:
        top = margin_ok.sort_values("margin_latest", ascending=False).iloc[0]
        insights.append(f"Margin leader: {top['company_name']} ({top['margin_latest']*100:.1f}%)")

    rev_ok = cs.dropna(subset=["revenue_cagr"])
    if not rev_ok.empty:
        top = rev_ok.sort_values("revenue_cagr", ascending=False).iloc[0]
        insights.append(f"Fastest revenue CAGR: {top['company_name']} ({top['revenue_cagr']*100:.1f}%)")

    return insights[:6]

