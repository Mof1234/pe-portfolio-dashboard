PE Portfolio Performance Dashboard
Purpose

This project is a Private Equity-style portfolio monitoring dashboard that tracks fund-level performance, value creation, and operating/risk metrics across portfolio companies. It is built to resemble how PE teams validate portfolio data and monitor value drivers: cashflows, valuation marks, operating trends, and leverage.

What it includes
Portfolio-level KPIs (fund view)

Paid-in (Contributions)

Distributions

NAV (unrealized value from valuation marks)

TVPI / DPI / RVPI

Fund IRR (XIRR on dated cashflows + terminal NAV)

Company monitoring (asset view)

Company comparison table: latest revenue/EBITDA, margin, leverage, MOIC, IRR, growth (CAGR)

Operating performance trends: portfolio revenue + EBITDA time series

Value creation bridge: entry equity → EBITDA growth → multiple change → deleveraging → current equity

Risk view: leverage chart + risk flags (high leverage, low margin)

Key Insights: quick “leader/laggard” callouts (top MOIC, highest leverage, margin leader, etc.)

Data model

Inputs are modeled as separate tables (CSV):

portfolio_companies.csv: company master (sector, entry date, ownership, etc.)

financials.csv: quarterly revenue / EBITDA / capex / net debt

valuation.csv: valuation marks (equity value + optional EV/multiples)

capital_flows.csv: dated cashflows (for XIRR and MOIC)

Currency conventions

All currency values should be stored as USD (either absolute dollars or USD mm).

The app displays headline values in USD mm.

Cashflows follow PE convention:

Contributions are stored as negative cashflows (cash out)

Distributions are stored as positive cashflows (cash in)

Metric definitions
Fund-level

Paid-in = −Σ(Contributions)

Distributions = Σ(Distributions)

NAV = Σ(latest equity_value per company as-of date)

DPI = Distributions / Paid-in

RVPI = NAV / Paid-in

TVPI = (Distributions + NAV) / Paid-in

IRR = XIRR(cashflows + terminal NAV)

Company-level

MOIC = (Realized + Unrealized) / Invested (company basis)

IRR = XIRR(company cashflows + terminal equity value)

Revenue CAGR = annualized growth between first and last reported quarters

EBITDA Margin = EBITDA / Revenue (latest quarter)

Leverage = Net Debt / EBITDA (latest quarter)

Key implementation changes (important)

This dashboard includes several changes to make the results accurate and maintainable:

Single source-of-truth filtering

All charts/KPIs use the same filtered frames:

companies_f, financials_f, valuation_f, capital_flows_f

Schema normalization + aliasing in data_prep.py

Column names are normalized (lowercase, underscores).

Common aliases are mapped (e.g., period_end, as_of_date, equity_value, flow_type, amount).

Fund KPIs computed correctly

Fund KPIs use:

Cashflows only (Contribution/Distribution)

NAV from valuation marks (latest equity_value per company)

Fund IRR adds a terminal NAV row at the as-of date

Cashflow visuals cleaned

Operating Trends cashflow chart shows only Contribution/Distribution

(Optional enhancement in progress) A clearer version can plot Paid-in and Distributions as positive bars plus a net line.

Restored Overview sections

Risk View and Key Insights are rendered inside the Overview tab (fixed indentation/layout).

Unique Streamlit keys

All st.plotly_chart() and key widgets use unique keys to avoid DuplicateElement errors.

How to run locally
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit cache clear
streamlit run app.py

Troubleshooting

1. DuplicateElementId / DuplicateElementKey

Ensure each st.plotly_chart(..., key="...") has a unique key.

2. “No Contribution/Distribution cashflows”

Your flow_type values may not be normalized to exactly:

Contribution, Distribution

Fix in data_prep.py mapping, then rerun.

3. Data not updating after edits

Clear Streamlit cache:
streamlit cache clear
