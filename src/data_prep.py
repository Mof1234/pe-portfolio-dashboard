# src/data_prep.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.schema import validate_df


# -----------------------------
# Cleaning helpers
# -----------------------------
def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
        .str.replace("/", "_")
        .str.replace("__", "_")
    )
    return df


def _rename_first_match(df: pd.DataFrame, target: str, candidates: list[str]) -> pd.DataFrame:
    if target in df.columns:
        return df
    for c in candidates:
        if c in df.columns:
            return df.rename(columns={c: target})
    return df


def _to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _to_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Robust numeric coercion: "$1,234", "12.4x", "(100.5)", "12.3m", "5.2bn", "—", "n/a".
    """
    if col not in df.columns:
        return df

    s = df[col]
    if pd.api.types.is_numeric_dtype(s):
        return df

    s = s.astype(str).str.strip()
    s = s.replace(
        {
            "": pd.NA,
            "na": pd.NA,
            "n/a": pd.NA,
            "none": pd.NA,
            "null": pd.NA,
            "-": pd.NA,
            "—": pd.NA,
        }
    )

    # (123) => -123
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)

    # strip currency + commas
    s = s.str.replace(r"[\$,]", "", regex=True)

    # strip x (multiples)
    s = s.str.replace(r"(?i)\bx\b", "", regex=True)

    mult = pd.Series(1.0, index=s.index)

    # billions
    bn_mask = s.str.contains(r"(?i)\b(bn|b)\b", regex=True, na=False)
    mult.loc[bn_mask] = 1e9
    s = s.str.replace(r"(?i)\b(bn|b)\b", "", regex=True)

    # millions
    mm_mask = s.str.contains(r"(?i)\b(mm|m)\b", regex=True, na=False)
    mult.loc[mm_mask] = 1e6
    s = s.str.replace(r"(?i)\b(mm|m)\b", "", regex=True)

    # thousands
    k_mask = s.str.contains(r"(?i)\bk\b", regex=True, na=False)
    mult.loc[k_mask] = 1e3
    s = s.str.replace(r"(?i)\bk\b", "", regex=True)

    df[col] = pd.to_numeric(s, errors="coerce") * mult
    return df


def _normalize_flow_type(df: pd.DataFrame) -> pd.DataFrame:
    if "flow_type" not in df.columns:
        return df

    s = df["flow_type"].astype(str).str.strip().str.lower()

    mapping = {
        # Contributions (capital calls / investments)
        "contribution": "Contribution",
        "contrib": "Contribution",
        "capital_call": "Contribution",
        "capitalcall": "Contribution",
        "call": "Contribution",
        "investment": "Contribution",
        "equity_investment": "Contribution",
        "equity investment": "Contribution",
        "follow_on_equity": "Contribution",
        "follow-on equity": "Contribution",
        "followon": "Contribution",
        "purchase": "Contribution",
        "buy": "Contribution",

        # Distributions
        "distribution": "Distribution",
        "dist": "Distribution",
        "proceeds": "Distribution",
        "dividend": "Distribution",
        "return": "Distribution",
        "sell": "Distribution",
        "exit": "Distribution",

        # NAV / marks / unrealized (NOT a cashflow; we already add terminal NAV from valuation marks)
        "unrealized_value": "NAV",
        "unrealized value": "NAV",
        "nav": "NAV",
        "mark": "NAV",
        "valuation": "NAV",
        "fair_value": "NAV",
        "fair value": "NAV",
    }

    df["flow_type"] = s.map(mapping).fillna(df["flow_type"])
    return df



def _enforce_sign_convention(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convention:
      - Contribution amounts negative
      - Distribution amounts positive
    """
    if not {"flow_type", "amount"}.issubset(df.columns):
        return df

    df = _to_numeric(df, "amount")
    contrib = df["flow_type"] == "Contribution"
    dist = df["flow_type"] == "Distribution"
    df.loc[contrib, "amount"] = -df.loc[contrib, "amount"].abs()
    df.loc[dist, "amount"] = df.loc[dist, "amount"].abs()
    return df


# -----------------------------
# Safe as-of merge per company
# -----------------------------
def _asof_merge(fin: pd.DataFrame, val: pd.DataFrame, fin_col: str, out_col: str) -> pd.DataFrame:
    """
    For each company_id:
      attach latest fin[fin_col] where fin.period_end <= val.as_of_date
    """
    v = val.copy()

    if v.empty or fin.empty:
        v[out_col] = pd.NA
        return v

    needed_fin = {"company_id", "period_end", fin_col}
    needed_val = {"company_id", "as_of_date"}
    if not needed_fin.issubset(fin.columns) or not needed_val.issubset(v.columns):
        v[out_col] = pd.NA
        return v

    f = fin[["company_id", "period_end", fin_col]].copy()
    f["company_id"] = f["company_id"].astype(str)
    v["company_id"] = v["company_id"].astype(str)

    f["period_end"] = pd.to_datetime(f["period_end"], errors="coerce")
    v["as_of_date"] = pd.to_datetime(v["as_of_date"], errors="coerce")

    f = f.dropna(subset=["company_id", "period_end"]).sort_values(["company_id", "period_end"])
    v = v.dropna(subset=["company_id", "as_of_date"]).sort_values(["company_id", "as_of_date"])

    out_parts = []
    for cid, v_g in v.groupby("company_id", sort=False):
        f_g = f[f["company_id"] == cid].dropna(subset=[fin_col]).sort_values("period_end")
        if f_g.empty:
            v_g[out_col] = pd.NA
            out_parts.append(v_g)
            continue

        merged = pd.merge_asof(
            v_g.sort_values("as_of_date"),
            f_g[["period_end", fin_col]].sort_values("period_end"),
            left_on="as_of_date",
            right_on="period_end",
            direction="backward",
        )
        merged = merged.rename(columns={fin_col: out_col}).drop(columns=["period_end"], errors="ignore")
        out_parts.append(merged)

    return pd.concat(out_parts, ignore_index=True) if out_parts else v.assign(**{out_col: pd.NA})


# -----------------------------
# Canonical valuation builders
# -----------------------------
def _canonicalize_valuation(financials: pd.DataFrame, valuation: pd.DataFrame) -> pd.DataFrame:
    """
    Output canonical valuation schema:
      company_id, as_of_date, current_ev, current_ebitda, ev_to_ebitda, equity_value
    Works across these valuation variants:
      - has current_multiple or ev_to_ebitda
      - has current_ev/current_ebitda or only entry_* fields
      - may/may not have equity_value
    """
    v = valuation.copy()

    # Ensure required ID/date
    v["company_id"] = v["company_id"].astype(str)
    v = _to_datetime(v, "as_of_date")
    v = v.dropna(subset=["company_id", "as_of_date"])

    # Ensure EV + EBITDA columns exist (use entry as fallback if current missing)
    if "current_ev" not in v.columns:
        v["current_ev"] = v["entry_ev"] if "entry_ev" in v.columns else pd.NA
    if "current_ebitda" not in v.columns:
        v["current_ebitda"] = v["entry_ebitda"] if "entry_ebitda" in v.columns else pd.NA

    for c in ["current_ev", "current_ebitda", "entry_ev", "entry_ebitda", "current_multiple", "entry_multiple", "ev_to_ebitda", "equity_value"]:
        v = _to_numeric(v, c)

    # Build ev_to_ebitda if missing
    if "ev_to_ebitda" not in v.columns:
        v["ev_to_ebitda"] = pd.NA

    # Priority: current_multiple -> ev_to_ebitda
    if v["ev_to_ebitda"].isna().all() and "current_multiple" in v.columns:
        v["ev_to_ebitda"] = v["current_multiple"]

    # Next: if still missing, compute from EV/EBITDA
    missing_mult = v["ev_to_ebitda"].isna()
    can_compute = (v["current_ev"].notna()) & (v["current_ebitda"].notna()) & (v["current_ebitda"] != 0)
    v.loc[missing_mult & can_compute, "ev_to_ebitda"] = v.loc[missing_mult & can_compute, "current_ev"] / v.loc[missing_mult & can_compute, "current_ebitda"]

    # Build equity_value if missing
    if "equity_value" not in v.columns:
        v["equity_value"] = pd.NA

    # equity = EV - net_debt_asof
    f = financials.copy()
    f["company_id"] = f["company_id"].astype(str)
    f = _to_datetime(f, "period_end")
    f = _to_numeric(f, "net_debt")
    f = f.dropna(subset=["company_id", "period_end"])

    v2 = _asof_merge(f, v, "net_debt", "net_debt_asof")
    v2["net_debt_asof"] = pd.to_numeric(v2["net_debt_asof"], errors="coerce")

    missing_eq = v2["equity_value"].isna()
    v2.loc[missing_eq, "equity_value"] = v2.loc[missing_eq, "current_ev"] - v2.loc[missing_eq, "net_debt_asof"].fillna(0)

    # Fallback: if still missing (no net debt), equity ~ EV
    still_missing_eq = v2["equity_value"].isna()
    v2.loc[still_missing_eq, "equity_value"] = v2.loc[still_missing_eq, "current_ev"]

    v2 = v2.drop(columns=["net_debt_asof"], errors="ignore")

    # Final coercion
    for c in ["current_ev", "current_ebitda", "ev_to_ebitda", "equity_value"]:
        v2 = _to_numeric(v2, c)

    return v2[["company_id", "as_of_date", "current_ev", "current_ebitda", "ev_to_ebitda", "equity_value"]]


# -----------------------------
# Public API
# -----------------------------
def load_data(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)

    companies = _clean_cols(pd.read_csv(data_path / "portfolio_companies.csv"))
    financials = _clean_cols(pd.read_csv(data_path / "financials.csv"))
    valuation_raw = _clean_cols(pd.read_csv(data_path / "valuation.csv"))
    capital_flows = _clean_cols(pd.read_csv(data_path / "capital_flows.csv"))

    # Companies
    companies = _rename_first_match(companies, "company_id", ["id", "companyid", "company"])
    companies = _rename_first_match(companies, "company_name", ["name", "company", "companyname", "platform_name"])
    companies = _rename_first_match(companies, "sector", ["industry", "vertical"])
    companies["company_id"] = companies["company_id"].astype(str)

    # Financials
    financials = _rename_first_match(financials, "company_id", ["id", "companyid", "company"])
    financials = _rename_first_match(financials, "period_end", ["period_end_date", "date", "as_of"])
    financials["company_id"] = financials["company_id"].astype(str)
    financials = _to_datetime(financials, "period_end")
    for c in ["revenue", "ebitda", "capex", "net_debt"]:
        financials = _to_numeric(financials, c)
    financials = financials.dropna(subset=["company_id", "period_end"])

    # Valuation raw aliases
    valuation_raw = _rename_first_match(valuation_raw, "company_id", ["id", "companyid", "company"])
    valuation_raw = _rename_first_match(valuation_raw, "as_of_date", ["date", "as_of", "valuation_date", "mark_date"])
    # Normalize potential multiple naming
    valuation_raw = _rename_first_match(valuation_raw, "current_multiple", ["ev_to_ebitda", "multiple", "current_mult"])
    valuation_raw["company_id"] = valuation_raw["company_id"].astype(str)

    # Canonical valuation
    valuation = _canonicalize_valuation(financials, valuation_raw)

    # Capital flows
    capital_flows = _rename_first_match(capital_flows, "company_id", ["id", "companyid", "company"])
    capital_flows = _rename_first_match(capital_flows, "date", ["cashflow_date", "flow_date", "transaction_date"])
    capital_flows = _rename_first_match(capital_flows, "flow_type", ["type", "cashflow_type", "transaction_type"])
    capital_flows = _rename_first_match(capital_flows, "amount", ["value", "amt", "cashflow", "cash_flow", "usd_amount"])
    capital_flows["company_id"] = capital_flows["company_id"].astype(str)
    capital_flows = _to_datetime(capital_flows, "date")
    capital_flows = _to_numeric(capital_flows, "amount")
    capital_flows = _normalize_flow_type(capital_flows)
    capital_flows = _enforce_sign_convention(capital_flows)
    capital_flows = capital_flows.dropna(subset=["company_id", "date"])

    # Validate after derivations
    issues = []
    issues += validate_df(companies, "companies")
    issues += validate_df(financials, "financials")
    issues += validate_df(valuation, "valuation")
    issues += validate_df(capital_flows, "capital_flows")

    if issues:
        msg = "Schema validation issues:\n- " + "\n- ".join(issues)
        msg += (
            "\n\nDEBUG COLUMNS:"
            f"\ncompanies: {list(companies.columns)}"
            f"\nfinancials: {list(financials.columns)}"
            f"\nvaluation: {list(valuation.columns)}"
            f"\ncapital_flows: {list(capital_flows.columns)}"
        )
        raise ValueError(msg)

    return companies, financials, valuation, capital_flows
