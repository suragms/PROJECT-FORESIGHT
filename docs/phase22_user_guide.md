# Phase 22 — User Guide

A plain-language guide for business users, managers, and evaluators.

## What Does This System Do?

PROJECT FORESIGHT helps retail teams **see future demand** and **understand inventory risk** for individual products (SKUs). It answers:

- How much demand should we expect over the next 6 weeks?
- Which products are at risk of running out?
- Which products may have too much stock?
- What action should we consider?

**Important:** This is a **decision-support** tool. It does not automatically place orders or change inventory systems.

## Viewing Forecasts

1. Open the **Executive Dashboard**: `streamlit run dashboard/phase22_executive_dashboard.py`
2. Select a SKU from the dropdown
3. Review the 6-week bar chart — each bar is one week ahead (h1 = next week, h6 = six weeks out)

### Understanding the 6-Week Horizon

- **Weeks 1–6** are the validated production forecast horizon
- These are the forecasts the system was tested and promoted on
- **Weeks 7–8** (if shown) are labeled **EXTENDED / PARTIAL** — use with extra caution

## Understanding Risk Levels

| Level | Meaning |
|-------|---------|
| **LOW** | Stockout risk is low |
| **HIGH / CRITICAL** | Product may run out before replenishment |
| **OPTIMAL** | Overstock risk is acceptable |
| **HIGH / SEVERE** | Too much inventory may be tying up capital |

## Recommended Actions

| Action | When It Appears |
|--------|-----------------|
| **REORDER NOW** | Urgent replenishment may be needed |
| **WATCH / VOLATILE** | Demand or supply is uncertain — monitor closely |
| **HEALTHY** | Inventory position looks balanced |
| **MARKDOWN / CLEAR** | Excess stock — consider promotional clearance |

Always combine these recommendations with your own business judgment and supplier lead times.

## Monitoring Status

The system continuously checks (via Phase 21 monitoring):

- Data quality
- Feature correctness
- Drift in inputs and forecasts
- Model integrity (has the model file changed?)

Open the **Monitoring Dashboard**: `streamlit run dashboard/phase21_monitoring.py`

### Understanding Alerts

Alerts have three severity levels:

- **INFO** — informational
- **WARNING** — review recommended
- **CRITICAL** — immediate investigation needed

Alerts do **not** automatically change forecasts or recommendations.

## Performance Numbers

You will see WAPE figures (e.g., 13.96%). These are **validation / backtest** results from historical testing — **not live production performance**.

Live production performance is **PENDING ACTUALS** until real demand data is collected and compared to forecasts.

## Known Limitations

1. **Holiday periods (Nov–Dec)** — forecasts may be less accurate during holidays
2. **Weeks 7–8** — partial accuracy only
3. **UCI dataset** — used for research, not production forecasting
4. **No guaranteed savings** — the system supports decisions; it does not guarantee financial outcomes

## Getting Help

- Technical setup: `docs/phase22_deployment_guide.md`
- API details: `docs/phase22_api_documentation.md`
- Full project report: `docs/PROJECT_FORESIGHT_FINAL_REPORT.md`
