from __future__ import annotations

import os
import numpy as np
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def q_ends(start: str, end: str) -> pd.DatetimeIndex:
    """Quarter-end dates inclusive."""
    return pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="QE")


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mm(x: float) -> float:
    """All currency in USD mm."""
    return float(x)


# -----------------------------
# Sector parameterization
# Tune these to make the dataset look/behave differently.
# -----------------------------
SECTOR_PARAMS = {
    "Technology": {
        "rev_growth_q_mean": 0.035,
        "rev_growth_q_sd": 0.020,
        "margin_start": 0.22,
        "margin_end": 0.34,
        "multiple_start": 12.0,
        "multiple_end": 10.0,   # mild compression
        "leverage_start": 4.5,
        "leverage_end": 2.8,
        "capex_pct_rev": 0.03,
    },
    "Healthcare": {
        "rev_growth_q_mean": 0.020,
        "rev_growth_q_sd": 0.012,
        "margin_start": 0.18,
        "margin_end": 0.26,
        "multiple_start": 11.5,
        "multiple_end": 11.0,
        "leverage_start": 5.0,
        "leverage_end": 3.5,
        "capex_pct_rev": 0.04,
    },
    "Industrials": {
        "rev_growth_q_mean": 0.015,
        "rev_growth_q_sd": 0.012,
        "margin_start": 0.14,
        "margin_end": 0.19,
        "multiple_start": 9.5,
        "multiple_end": 9.0,
        "leverage_start": 5.5,
        "leverage_end": 3.8,
        "capex_pct_rev": 0.05,
    },
    "Consumer": {
        "rev_growth_q_mean": 0.012,
        "rev_growth_q_sd": 0.015,
        "margin_start": 0.10,
        "margin_end": 0.14,
        "multiple_start": 10.0,
        "multiple_end": 9.0,
        "leverage_start": 5.0,
        "leverage_end": 4.0,
        "capex_pct_rev": 0.04,
    },
    "Infrastructure": {
        "rev_growth_q_mean": 0.018,
        "rev_growth_q_sd": 0.010,
        "margin_start": 0.28,
        "margin_end": 0.40,
        "multiple_start": 14.0,
        "multiple_end": 13.0,
        "leverage_start": 6.0,
        "leverage_end": 5.0,
        "capex_pct_rev": 0.08,
    },
    "Energy": {
        "rev_growth_q_mean": 0.010,
        "rev_growth_q_sd": 0.030,  # noisier
        "margin_start": 0.16,
        "margin_end": 0.18,
        "multiple_start": 6.5,
        "multiple_end": 6.0,
        "leverage_start": 3.5,
        "leverage_end": 2.8,
        "capex_pct_rev": 0.06,
    },
    "FinTech": {
        "rev_growth_q_mean": 0.030,
        "rev_growth_q_sd": 0.020,
        "margin_start": 0.15,
        "margin_end": 0.28,
        "multiple_start": 13.0,
        "multiple_end": 11.0,
        "leverage_start": 4.0,
        "leverage_end": 3.0,
        "capex_pct_rev": 0.03,
    },
}


def generate_portfolio(
    n_companies: int = 10,
    start_q: str = "2023-03-31",
    end_q: str = "2025-12-31",
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    quarters = q_ends(start_q, end_q)
    n_q = len(quarters)

    sectors = list(SECTOR_PARAMS.keys())

    # -----------------------------
    # companies
    # -----------------------------
    companies = []
    for i in range(n_companies):
        cid = f"C{i+1:03d}"
        sector = sectors[i % len(sectors)]
        companies.append(
            {
                "company_id": cid,
                "company_name": f"{['Atlas','Summit','Meridian','Evergrid','Bluepeak','Harborline','Pioneer','Northbridge','Riftline','Koru'][i % 10]} "
                                f"{['Software','Healthcare','Logistics','Renewables','Consumer Brands','Data Centers','Payments','Industrial Components','Mining Services','AgriTech Solutions'][i % 10]}",
                "platform_name": f"Platform {1 + (i // 3)}",
                "sector": sector,
                "subsector": "N/A",
                "region": rng.choice(["North America", "Europe", "Africa"], p=[0.75, 0.15, 0.10]),
                "entry_date": quarters[0],
                "ownership_pct": float(rng.uniform(0.55, 0.95)),
                "status": "Active",
                "investment_thesis": "Operational improvement + growth + prudent leverage.",
            }
        )
    companies = pd.DataFrame(companies)

    # -----------------------------
    # financials + valuation
    # -----------------------------
    fin_rows = []
    val_rows = []
    cf_rows = []

    for _, co in companies.iterrows():
        p = SECTOR_PARAMS[co["sector"]]

        # Starting scale
        rev0 = mm(rng.uniform(25, 90))                 # USD mm per quarter
        margin0 = p["margin_start"]
        ebitda0 = mm(rev0 * margin0)

        # Smooth improvement paths (with noise)
        margins = np.linspace(p["margin_start"], p["margin_end"], n_q)
        multiples = np.linspace(p["multiple_start"], p["multiple_end"], n_q)
        leverages = np.linspace(p["leverage_start"], p["leverage_end"], n_q)

        # Revenue series
        rev = [rev0]
        for t in range(1, n_q):
            g = rng.normal(p["rev_growth_q_mean"], p["rev_growth_q_sd"])
            g = clamp(g, -0.10, 0.12)  # avoid absurd drops/spikes
            rev.append(mm(rev[-1] * (1.0 + g)))

        for t, dt in enumerate(quarters):
            r = mm(rev[t])
            m = clamp(float(margins[t] + rng.normal(0, 0.01)), 0.03, 0.55)
            e = mm(r * m)

            # capex tied to revenue, with some variation
            capex = mm(r * p["capex_pct_rev"] * (1.0 + rng.normal(0, 0.15)))
            capex = max(0.0, capex)

            # leverage target → net debt
            lev = clamp(float(leverages[t] + rng.normal(0, 0.15)), 0.0, 12.0)
            net_debt = mm(e * lev)

            fin_rows.append(
                {
                    "company_id": co["company_id"],
                    "period_end": dt,
                    "fiscal_quarter": f"Q{((dt.quarter - 1) % 4) + 1} {dt.year}",
                    "revenue": r,
                    "ebitda": e,
                    "capex": capex,
                    "net_debt": net_debt,
                }
            )

            # valuation mark (EV = EBITDA * multiple)
            mult = clamp(float(multiples[t] + rng.normal(0, 0.35)), 3.5, 25.0)
            ev = mm(e * mult)
            equity = mm(ev - net_debt)

            # Prevent negative equity for demo unless you want distress scenarios
            equity = max(5.0, equity)

            val_rows.append(
                {
                    "company_id": co["company_id"],
                    "as_of_date": dt,
                    "current_ev": ev,
                    "current_ebitda": e,
                    "ev_to_ebitda": (ev / e) if e else np.nan,
                    "equity_value": equity,
                }
            )

        # -----------------------------
        # capital flows: (Contribution early, optional recap, exit late)
        # -----------------------------
        entry = quarters[0]
        mid = quarters[max(1, n_q // 2)]
        exit_dt = quarters[-1]

        # Entry contribution roughly linked to entry equity, ownership
        v_entry = [v for v in val_rows if v["company_id"] == co["company_id"]][0]
        entry_equity = float(v_entry["equity_value"])
        entry_check = mm(entry_equity * co["ownership_pct"] * rng.uniform(0.18, 0.35))

        cf_rows.append(
            {
                "company_id": co["company_id"],
                "date": entry,
                "flow_type": "Contribution",
                "amount": -entry_check,  # negative by convention
                "security": "Equity",
                "notes": "Entry investment",
            }
        )

        # Optional follow-on
        if rng.random() < 0.55:
            follow = mm(entry_check * rng.uniform(0.10, 0.40))
            cf_rows.append(
                {
                    "company_id": co["company_id"],
                    "date": quarters[min(3, n_q - 1)],
                    "flow_type": "Contribution",
                    "amount": -follow,
                    "security": "Equity",
                    "notes": "Follow-on / add-on",
                }
            )

        # Optional recap distribution (minor, mid-life)
        if rng.random() < 0.35:
            recap = mm(entry_check * rng.uniform(0.10, 0.30))
            cf_rows.append(
                {
                    "company_id": co["company_id"],
                    "date": mid,
                    "flow_type": "Distribution",
                    "amount": recap,
                    "security": "Dividend Recap",
                    "notes": "Partial recap",
                }
            )

        # Exit distribution linked to final equity value
        v_last = [v for v in val_rows if v["company_id"] == co["company_id"]][-1]
        exit_equity = float(v_last["equity_value"])
        exit_proceeds = mm(exit_equity * co["ownership_pct"] * rng.uniform(0.75, 1.05))

        cf_rows.append(
            {
                "company_id": co["company_id"],
                "date": exit_dt,
                "flow_type": "Distribution",
                "amount": exit_proceeds,
                "security": "Sale",
                "notes": "Exit proceeds",
            }
        )

    financials = pd.DataFrame(fin_rows)
    valuation = pd.DataFrame(val_rows)
    capital_flows = pd.DataFrame(cf_rows)

    # Ensure sorted order for merge_asof and your downstream logic
    financials = financials.sort_values(["company_id", "period_end"]).reset_index(drop=True)
    valuation = valuation.sort_values(["company_id", "as_of_date"]).reset_index(drop=True)
    capital_flows = capital_flows.sort_values(["company_id", "date"]).reset_index(drop=True)

    # Make sure types are clean
    for c in ["revenue", "ebitda", "capex", "net_debt"]:
        financials[c] = pd.to_numeric(financials[c], errors="coerce")
    for c in ["current_ev", "current_ebitda", "ev_to_ebitda", "equity_value"]:
        valuation[c] = pd.to_numeric(valuation[c], errors="coerce")
    capital_flows["amount"] = pd.to_numeric(capital_flows["amount"], errors="coerce")

    return companies, financials, valuation, capital_flows


def main():
    out_dir = os.path.join("data", "clean")
    os.makedirs(out_dir, exist_ok=True)

    companies, financials, valuation, capital_flows = generate_portfolio(
        n_companies=10,
        start_q="2023-03-31",
        end_q="2025-12-31",
        seed=7,
    )

    companies.to_csv(os.path.join(out_dir, "portfolio_companies.csv"), index=False)
    financials.to_csv(os.path.join(out_dir, "financials.csv"), index=False)
    valuation.to_csv(os.path.join(out_dir, "valuation.csv"), index=False)
    capital_flows.to_csv(os.path.join(out_dir, "capital_flows.csv"), index=False)

    print("Wrote clean synthetic data to:", out_dir)
    print("Rows:", len(companies), len(financials), len(valuation), len(capital_flows))


if __name__ == "__main__":
    main()
