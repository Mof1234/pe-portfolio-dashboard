from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd


# -----------------------------
# Time helpers
# -----------------------------
def q_ends(start: str, end: str) -> pd.DatetimeIndex:
    """Quarter-end dates inclusive."""
    return pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="QE")


def q_label(dt: pd.Timestamp) -> str:
    return f"Q{dt.quarter} {dt.year}"


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def mm(x: float) -> float:
    """All currency is USD mm."""
    return float(x)


# -----------------------------
# Market regime (multiples)
# -----------------------------
def market_regime_multiplier(dt: pd.Timestamp) -> float:
    """
    Simple macro regime curve:
    - 2018-2019: normal
    - 2020-2021: expansion
    - 2022-2023: compression
    - 2024-2026: normalize
    """
    y = dt.year
    if y <= 2019:
        return 1.00
    if y in (2020, 2021):
        return 1.12
    if y in (2022, 2023):
        return 0.90
    return 1.00


# -----------------------------
# Sector parameterization
# -----------------------------
SECTOR_PARAMS = {
    "Technology": dict(
        rev_growth_q_mean=0.030,
        rev_growth_q_sd=0.018,
        margin_start=0.18,
        margin_end=0.32,
        multiple_base=12.0,
        leverage_start=4.5,
        leverage_end=2.8,
        capex_pct_rev=0.03,
    ),
    "Healthcare": dict(
        rev_growth_q_mean=0.020,
        rev_growth_q_sd=0.012,
        margin_start=0.16,
        margin_end=0.24,
        multiple_base=11.0,
        leverage_start=5.0,
        leverage_end=3.5,
        capex_pct_rev=0.04,
    ),
    "Industrials": dict(
        rev_growth_q_mean=0.015,
        rev_growth_q_sd=0.012,
        margin_start=0.12,
        margin_end=0.18,
        multiple_base=9.0,
        leverage_start=5.5,
        leverage_end=3.8,
        capex_pct_rev=0.05,
    ),
    "Consumer": dict(
        rev_growth_q_mean=0.012,
        rev_growth_q_sd=0.015,
        margin_start=0.09,
        margin_end=0.13,
        multiple_base=9.5,
        leverage_start=5.0,
        leverage_end=4.0,
        capex_pct_rev=0.04,
    ),
    "Infrastructure": dict(
        rev_growth_q_mean=0.018,
        rev_growth_q_sd=0.010,
        margin_start=0.25,
        margin_end=0.38,
        multiple_base=13.0,
        leverage_start=6.0,
        leverage_end=5.0,
        capex_pct_rev=0.08,
    ),
    "Energy": dict(
        rev_growth_q_mean=0.010,
        rev_growth_q_sd=0.030,
        margin_start=0.14,
        margin_end=0.17,
        multiple_base=6.0,
        leverage_start=3.5,
        leverage_end=2.8,
        capex_pct_rev=0.06,
    ),
    "FinTech": dict(
        rev_growth_q_mean=0.028,
        rev_growth_q_sd=0.020,
        margin_start=0.12,
        margin_end=0.25,
        multiple_base=12.0,
        leverage_start=4.0,
        leverage_end=3.0,
        capex_pct_rev=0.03,
    ),
}


@dataclass
class FundConfig:
    fund_start: str = "2018-03-31"
    fund_end: str = "2026-12-31"
    n_companies: int = 14
    seed: int = 11

    # Investment period: first N quarters from fund start
    invest_period_quarters: int = 16  # ~4 years

    # Hold period (in quarters)
    hold_min_q: int = 10   # ~2.5 years
    hold_max_q: int = 28   # ~7 years

    # Fee model
    annual_mgmt_fee_rate: float = 0.020  # 2% of cost basis (approx)
    fee_starts_at_fund_start: bool = True

    # Follow-on behavior
    follow_on_prob: float = 0.55
    follow_on_min_pct: float = 0.10
    follow_on_max_pct: float = 0.45

    # Interim recap behavior
    recap_prob: float = 0.25
    recap_min_pct: float = 0.08
    recap_max_pct: float = 0.25


def generate_fund_style(cfg: FundConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.seed)
    quarters = q_ends(cfg.fund_start, cfg.fund_end)

    invest_quarters = quarters[: cfg.invest_period_quarters]
    sectors = list(SECTOR_PARAMS.keys())

    # Company name templates (purely cosmetic)
    name_a = ["Atlas", "Summit", "Meridian", "Evergrid", "Bluepeak", "Harborline", "Pioneer", "Northbridge", "Riftline", "Koru", "Crescent", "Stonegate", "Redwood", "Clearwater"]
    name_b = ["Software", "Healthcare", "Logistics", "Renewables", "Consumer Brands", "Data Centers", "Payments", "Industrial Components", "Mining Services", "AgriTech", "Networks", "Diagnostics", "Field Services", "Analytics"]

    # -----------------------------
    # Build companies with staggered entry/exit
    # -----------------------------
    companies = []
    deal_meta = {}  # company_id -> dict(entry_dt, exit_dt, sector, ownership, entry_check)

    for i in range(cfg.n_companies):
        cid = f"C{i+1:03d}"
        sector = sectors[i % len(sectors)]
        entry_dt = rng.choice(invest_quarters)

        hold_q = int(rng.integers(cfg.hold_min_q, cfg.hold_max_q + 1))
        # exit could exceed fund_end -> still Active
        exit_dt = entry_dt + pd.offsets.QuarterEnd(hold_q)

        ownership = float(rng.uniform(0.55, 0.95))

        # starting revenue scale per quarter (USD mm)
        rev0 = mm(rng.uniform(15, 85))

        companies.append(
            dict(
                company_id=cid,
                company_name=f"{name_a[i % len(name_a)]} {name_b[i % len(name_b)]}",
                platform_name=f"Platform {1 + (i // 4)}",
                sector=sector,
                subsector="N/A",
                region=rng.choice(["North America", "Europe", "Africa"], p=[0.75, 0.15, 0.10]),
                entry_date=pd.to_datetime(entry_dt),
                ownership_pct=ownership,
                status="Active",  # will update later if realized
                investment_thesis="Staggered-entry buyout with value creation through growth + margins + de-levering.",
            )
        )

        deal_meta[cid] = dict(
            sector=sector,
            entry_dt=pd.to_datetime(entry_dt),
            exit_dt=pd.to_datetime(exit_dt),
            ownership=ownership,
            rev0=rev0,
        )

    companies = pd.DataFrame(companies)

    # -----------------------------
    # Generate financials + valuation marks only while "owned"
    # -----------------------------
    fin_rows = []
    val_rows = []

    # Store entry checks to allocate fees realistically
    entry_checks = {}

    for cid, meta in deal_meta.items():
        p = SECTOR_PARAMS[meta["sector"]]
        entry_dt = meta["entry_dt"]
        exit_dt = meta["exit_dt"]

        # owned_quarters: from entry to min(exit, fund_end)
        owned_q = quarters[(quarters >= entry_dt) & (quarters <= min(exit_dt, pd.to_datetime(cfg.fund_end)))].to_list()
        if not owned_q:
            continue

        n_q = len(owned_q)

        # Smooth improvement paths + noise
        margins = np.linspace(p["margin_start"], p["margin_end"], n_q)
        leverages = np.linspace(p["leverage_start"], p["leverage_end"], n_q)

        # Revenue path with noise
        rev = [meta["rev0"]]
        for t in range(1, n_q):
            g = rng.normal(p["rev_growth_q_mean"], p["rev_growth_q_sd"])
            g = clamp(g, -0.10, 0.12)
            rev.append(mm(rev[-1] * (1.0 + g)))

        # Multiple path: base * macro regime + idiosyncratic noise
        for t, dt in enumerate(owned_q):
            r = mm(rev[t])
            m = clamp(float(margins[t] + rng.normal(0, 0.012)), 0.03, 0.55)
            e = mm(r * m)

            capex = mm(r * p["capex_pct_rev"] * (1.0 + rng.normal(0, 0.18)))
            capex = max(0.0, capex)

            lev = clamp(float(leverages[t] + rng.normal(0, 0.18)), 0.0, 12.5)
            net_debt = mm(e * lev)

            fin_rows.append(
                dict(
                    company_id=cid,
                    period_end=dt,
                    fiscal_quarter=q_label(dt),
                    revenue=r,
                    ebitda=e,
                    capex=capex,
                    net_debt=net_debt,
                )
            )

            macro = market_regime_multiplier(dt)
            mult = (p["multiple_base"] * macro) + rng.normal(0, 0.50)
            mult = clamp(float(mult), 3.5, 25.0)

            ev = mm(e * mult)
            equity = mm(ev - net_debt)

            # allow occasional "stress" but keep >= small positive for dashboard stability
            equity = max(2.0, equity)

            val_rows.append(
                dict(
                    company_id=cid,
                    as_of_date=dt,
                    current_ev=ev,
                    current_ebitda=e,
                    ev_to_ebitda=(ev / e) if e else np.nan,
                    equity_value=equity,
                )
            )

        # Entry check linked to entry equity mark and ownership (rough)
        v_entry = [v for v in val_rows if v["company_id"] == cid and v["as_of_date"] == owned_q[0]][0]
        entry_equity = float(v_entry["equity_value"])
        entry_check = mm(entry_equity * meta["ownership"] * rng.uniform(0.16, 0.32))
        entry_checks[cid] = entry_check

    financials = pd.DataFrame(fin_rows)
    valuation = pd.DataFrame(val_rows)

    # -----------------------------
    # Capital flows: entry, follow-ons, recaps, exits + fee drag
    # -----------------------------
    cf_rows = []

    for cid, meta in deal_meta.items():
        entry_dt = meta["entry_dt"]
        exit_dt = meta["exit_dt"]
        ownership = meta["ownership"]

        # Only if this deal actually has marks (owned within fund window)
        if cid not in entry_checks:
            continue

        # Entry
        cf_rows.append(
            dict(
                company_id=cid,
                date=entry_dt,
                flow_type="Contribution",
                amount=-mm(entry_checks[cid]),
                security="Equity",
                notes="Entry investment",
            )
        )

        # Follow-on (during first year after entry)
        if rng.random() < cfg.follow_on_prob:
            follow_dt = entry_dt + pd.offsets.QuarterEnd(int(rng.integers(1, 5)))
            follow_dt = min(follow_dt, pd.to_datetime(cfg.fund_end))
            follow_amt = mm(entry_checks[cid] * rng.uniform(cfg.follow_on_min_pct, cfg.follow_on_max_pct))
            cf_rows.append(
                dict(
                    company_id=cid,
                    date=follow_dt,
                    flow_type="Contribution",
                    amount=-follow_amt,
                    security="Equity",
                    notes="Follow-on / add-on",
                )
            )

        # Interim recap (mid-hold, if exit happens within fund_end)
        if rng.random() < cfg.recap_prob:
            recap_dt = entry_dt + pd.offsets.QuarterEnd(int(rng.integers(6, 14)))
            recap_dt = min(recap_dt, pd.to_datetime(cfg.fund_end))
            recap_amt = mm(entry_checks[cid] * rng.uniform(cfg.recap_min_pct, cfg.recap_max_pct))
            cf_rows.append(
                dict(
                    company_id=cid,
                    date=recap_dt,
                    flow_type="Distribution",
                    amount=recap_amt,
                    security="Dividend Recap",
                    notes="Partial recap distribution",
                )
            )

        # Exit distribution only if exit within fund window
        if exit_dt <= pd.to_datetime(cfg.fund_end):
            # Use last available equity mark at/just before exit
            v_last = (
                valuation[(valuation["company_id"] == cid) & (valuation["as_of_date"] <= exit_dt)]
                .sort_values("as_of_date")
                .tail(1)
            )
            if not v_last.empty:
                exit_equity = float(v_last["equity_value"].iloc[0])
                exit_proceeds = mm(exit_equity * ownership * rng.uniform(0.78, 1.08))

                cf_rows.append(
                    dict(
                        company_id=cid,
                        date=exit_dt,
                        flow_type="Distribution",
                        amount=exit_proceeds,
                        security="Sale",
                        notes="Exit proceeds",
                    )
                )

                # Update company status
                companies.loc[companies["company_id"] == cid, "status"] = "Realized"

    capital_flows = pd.DataFrame(cf_rows)

    # -----------------------------
    # Management fees (fund-style)
    # Implemented as quarterly Contributions allocated pro-rata to cost basis,
    # so they flow through your existing paid-in logic without changing metrics.py.
    # -----------------------------
    if cfg.annual_mgmt_fee_rate > 0:
        # Approx quarterly fee rate
        fee_q = cfg.annual_mgmt_fee_rate / 4.0

        # Cost basis proxy: sum of entry checks (could also include follow-ons, but keep simple)
        cost_basis = pd.Series(entry_checks)
        total_cost = float(cost_basis.sum()) if len(cost_basis) else 0.0

        if total_cost > 0:
            for dt in quarters:
                if not cfg.fee_starts_at_fund_start and dt < quarters[0]:
                    continue

                # Allocate fees only to deals that have entered by this quarter and not exited yet
                active_cids = []
                for cid, meta in deal_meta.items():
                    if cid not in entry_checks:
                        continue
                    entered = meta["entry_dt"] <= dt
                    not_exited = meta["exit_dt"] > dt  # still in fund
                    if entered and not_exited:
                        active_cids.append(cid)

                if not active_cids:
                    continue

                active_cost = float(cost_basis.loc[active_cids].sum())
                if active_cost <= 0:
                    continue

                total_fee_this_q = mm(active_cost * fee_q)

                for cid in active_cids:
                    alloc = mm(total_fee_this_q * float(cost_basis.loc[cid]) / active_cost)
                    capital_flows = pd.concat(
                        [
                            capital_flows,
                            pd.DataFrame(
                                [
                                    dict(
                                        company_id=cid,
                                        date=dt,
                                        flow_type="Contribution",
                                        amount=-alloc,
                                        security="Fee",
                                        notes="Mgmt fee allocation (synthetic)",
                                    )
                                ]
                            ),
                        ],
                        ignore_index=True,
                    )

    # -----------------------------
    # Clean types + sort for merge_asof friendliness
    # -----------------------------
    if not financials.empty:
        financials["period_end"] = pd.to_datetime(financials["period_end"])
    if not valuation.empty:
        valuation["as_of_date"] = pd.to_datetime(valuation["as_of_date"])
    if not capital_flows.empty:
        capital_flows["date"] = pd.to_datetime(capital_flows["date"])

    for c in ["revenue", "ebitda", "capex", "net_debt"]:
        if c in financials.columns:
            financials[c] = pd.to_numeric(financials[c], errors="coerce")

    for c in ["current_ev", "current_ebitda", "ev_to_ebitda", "equity_value"]:
        if c in valuation.columns:
            valuation[c] = pd.to_numeric(valuation[c], errors="coerce")

    if "amount" in capital_flows.columns:
        capital_flows["amount"] = pd.to_numeric(capital_flows["amount"], errors="coerce")

    companies = companies.sort_values(["company_id"]).reset_index(drop=True)
    financials = financials.sort_values(["company_id", "period_end"]).reset_index(drop=True)
    valuation = valuation.sort_values(["company_id", "as_of_date"]).reset_index(drop=True)
    capital_flows = capital_flows.sort_values(["company_id", "date"]).reset_index(drop=True)

    return companies, financials, valuation, capital_flows


def main():
    cfg = FundConfig()

    out_dir = os.path.join("data", "clean_fund")
    os.makedirs(out_dir, exist_ok=True)

    companies, financials, valuation, capital_flows = generate_fund_style(cfg)

    companies.to_csv(os.path.join(out_dir, "portfolio_companies.csv"), index=False)
    financials.to_csv(os.path.join(out_dir, "financials.csv"), index=False)
    valuation.to_csv(os.path.join(out_dir, "valuation.csv"), index=False)
    capital_flows.to_csv(os.path.join(out_dir, "capital_flows.csv"), index=False)

    print("Wrote fund-style synthetic data to:", out_dir)
    print("Rows:", len(companies), len(financials), len(valuation), len(capital_flows))
    print("Realized companies:", int((companies["status"] == "Realized").sum()))
    print("Active companies:", int((companies["status"] == "Active").sum()))


if __name__ == "__main__":
    main()
