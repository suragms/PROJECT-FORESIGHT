# Phase 23.3 — Final Demo & Submission Evidence Checklist

Capture real screenshots or terminal output for Zidio submission. Do not fabricate images.

---

## Frontend (Vercel)

- [ ] **1. Homepage** — https://foresight-project-green.vercel.app/ (login portal, live backend label)
- [ ] **2. Login page** — email/password form (empty fields), Sign In button
- [ ] **3. Registration page** — Register tab with confirm password
- [ ] **4. Main navigation** — sidebar after login (Executive, Forecasting, Inventory, Monitoring, etc.)
- [ ] **5. Executive dashboard** — KPI cards, validation WAPE labeled as backtest
- [ ] **6. Forecasting page** — horizon selector, model registry table
- [ ] **7. Inventory/risk page** — SKU risk table, stockout/overstock flags
- [ ] **8. Monitoring dashboard** — health/drift summary panels

---

## Backend (Render)

- [ ] **9. API root** — `curl https://project-foresight-api-tofn.onrender.com/` → `"status":"online"`
- [ ] **10. Swagger docs** — https://project-foresight-api-tofn.onrender.com/docs
- [ ] **11. Health endpoint** — `GET /health` → `"status":"ok"`
- [ ] **12. Forecast API response** — `POST /forecast` or `/phase20/forecast` (after redeploy or with API key)
- [ ] **13. Risk API response** — `POST /phase20/risk/explain`
- [ ] **14. Invalid request error** — empty features → 400, not 500/crash

---

## Repository & Quality

- [ ] **15. Test suite** — `python -m pytest tests -q` → **280/280 PASS**
- [ ] **16. Model integrity** — `docs/phase23_3_final_integrity_verification.json` → frozen 12/12 + Phase 20 unchanged
- [ ] **17. README Live Application section** — Vercel + Render URLs
- [ ] **18. Acceptance matrix** — `docs/phase23_3_zidio_final_acceptance_matrix.md`

---

## Labeling Reminder

When presenting metrics in demos or slides:
- ✅ "Validation/backtest WAPE: 13.96% / 11.03% h1–h6"
- ❌ Do not label backtest metrics as "live production performance"
- ✅ "Live production performance: PENDING ACTUALS"
