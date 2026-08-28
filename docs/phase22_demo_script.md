# Phase 22 — Demo Script

**Duration:** 5–10 minutes  
**Audience:** Zidio evaluator, HR, technical reviewer, business stakeholder

---

## 1. Introduction (1 min)

> "PROJECT FORESIGHT helps retail teams forecast weekly demand and understand inventory risk. It's a decision-support system — it recommends actions but does not automatically place orders."

**Business problem:** Stockouts lose sales; overstock ties up capital. We need 6-week visibility per SKU.

---

## 2. Executive Dashboard (2 min)

```bash
streamlit run dashboard/phase22_executive_dashboard.py
```

**Show:**
- System Status: PRODUCTION PROMOTION COMPLETE
- Monitoring Status: MONITORING READY
- Production Performance: **PENDING ACTUALS**
- Validation WAPE: 13.96% / 11.03% — labeled **VALIDATION / BACKTEST**

> "These numbers are from historical backtesting, not live production measurement."

---

## 3. Select a SKU (1 min)

- Pick a SKU from dropdown
- Point to 6-week bar chart (h1–h6)
- Explain each bar = one week ahead

> "Weeks 1 through 6 are our validated production horizon."

---

## 4. Inventory Risk (1 min)

- Show risk distribution pie chart
- Select same SKU — show stockout/overstock levels
- Explain recommended action (e.g., REORDER NOW vs HEALTHY)

---

## 5. Business Impact (30 sec)

- Show Sales at Risk, Locked Capital, At-Risk SKUs
- Note: values come from source data, not fabricated

---

## 6. Model Information (30 sec)

- Model: phase20_synthetic_lightgbm
- 45 features, 6-week horizon
- Known limitation: holiday bias in Nov–Dec

---

## 7. Monitoring Dashboard (1 min)

```bash
streamlit run dashboard/phase21_monitoring.py
```

- Show health score and component statuses
- Show any alerts
- Explain: monitoring observes, does not retrain

---

## 8. API (30 sec, optional)

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` — show `/phase20/forecast` and `/phase21/health`

---

## 9. Architecture (30 sec)

Open `docs/phase22_system_architecture.md` — walk through the Mermaid diagram:

Data → Features → Model → Forecast → Risk + Monitoring → Dashboard

---

## 10. Results & Limitations (1 min)

| Result | Value |
|--------|-------|
| Baseline WAPE | 25.51% |
| Production WAPE (validation) | 13.96% |
| h1–h6 WAPE | 11.03% |
| Tests | 214/214 PASS |

**Limitations to mention:**
- Holiday bias partially unresolved
- h7–h8 partial accuracy
- Live performance pending actuals
- UCI is research only
- Legacy artifact issue documented

---

## 11. Closing

> "PROJECT FORESIGHT is a complete, tested, documented forecasting and risk platform ready for evaluation. It supports better inventory decisions without claiming guaranteed financial outcomes."

**Q&A:** Refer to `docs/PROJECT_FORESIGHT_FINAL_REPORT.md` and `docs/phase22_user_guide.md`.
