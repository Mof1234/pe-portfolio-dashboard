# src/schema.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class Schema:
    required_cols: Tuple[str, ...]
    date_cols: Tuple[str, ...] = ()
    numeric_cols: Tuple[str, ...] = ()


SCHEMAS: Dict[str, Schema] = {
    "companies": Schema(
        required_cols=("company_id", "company_name", "sector"),
    ),
    "financials": Schema(
        required_cols=("company_id", "period_end", "revenue", "ebitda", "net_debt"),
        date_cols=("period_end",),
        numeric_cols=("revenue", "ebitda", "net_debt"),
    ),
    "valuation": Schema(
        # Canonical valuation schema AFTER data_prep derivations
        required_cols=("company_id", "as_of_date", "current_ev", "current_ebitda", "ev_to_ebitda", "equity_value"),
        date_cols=("as_of_date",),
        numeric_cols=("current_ev", "current_ebitda", "ev_to_ebitda", "equity_value"),
    ),
    "capital_flows": Schema(
        required_cols=("company_id", "date", "flow_type", "amount"),
        date_cols=("date",),
        numeric_cols=("amount",),
    ),
}


def validate_df(df: pd.DataFrame, schema_name: str) -> List[str]:
    issues: List[str] = []
    if schema_name not in SCHEMAS:
        return [f"Unknown schema '{schema_name}'"]

    s = SCHEMAS[schema_name]

    missing = [c for c in s.required_cols if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {missing}")

    for c in s.date_cols:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            issues.append(f"Column '{c}' is not datetime dtype")

    for c in s.numeric_cols:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            issues.append(f"Column '{c}' is not numeric dtype")

    return issues
