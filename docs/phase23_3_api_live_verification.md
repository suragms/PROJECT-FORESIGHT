# Phase 23.3 — API Live Verification

**Verified:** 2026-08-29  
**Backend:** https://project-foresight-api-tofn.onrender.com/  
**Frontend:** https://foresight-project-green.vercel.app/

---

## 1. Public Endpoint Verification (Live)

| Endpoint | Method | HTTP | Response Summary |
|----------|--------|------|------------------|
| `/` | GET | **200** | `{"status":"online","service":"Project FORESIGHT — Demand & Inventory Intelligence API","version":"0.13.0",...}` |
| `/health` | GET | **200** | `{"status":"ok","version":"0.13.0","timestamp":"..."}` |
| `/ready` | GET | **503** | `{"status":"not_ready","models_verified":true,"registry_verified":true,"config_valid":false,"config_errors":["production requires FORESIGHT_API_AUTH_ENABLED=true","production requires FORESIGHT_API_API_KEY"]}` |
| `/docs` | GET | **200** | Swagger UI loads |

---

## 2. Scoring Endpoints (Repository Implementation)

### Legacy Phase 11 forecast (daily horizons)

| Item | Value |
|------|-------|
| **Endpoint** | `POST /forecast` |
| **Auth** | API key (`X-API-Key`) when `FORESIGHT_API_AUTH_ENABLED=true`; JWT Bearer accepted after Phase 23.3 fix |
| **Datasets** | `UCI`, `SYNTHETIC` |
| **Horizons** | 1, 3, 7, 14, 30 |

**Request (SYNTHETIC h=1 example):**
```json
{
  "source_dataset": "SYNTHETIC",
  "horizon": 1,
  "date": "2025-08-07",
  "entity_id": "STORE_001",
  "product_key": "SYN_SKU_00001",
  "features": { "units_sold_lag_1": 0.0, "...": "..." }
}
```

**Response (200):**
```json
{
  "forecasts": [{ "prediction": 2.5, "horizon": 1, "model_name": "synthetic_h1_hurdle_th050", "...": "..." }],
  "metadata": { "model_id": "synthetic_h1_hurdle_th050", "...": "..." },
  "n": 1
}
```

### Phase 20 production forecast (weekly SKU, 6-week validated horizon)

| Item | Value |
|------|-------|
| **Endpoint** | `POST /phase20/forecast` |
| **Model** | `phase20_synthetic_lightgbm` |
| **Source** | `SYNTHETIC` only (UCI rejected) |
| **Features** | 45-feature contract — `GET /phase20/contract` |

**Request:**
```json
{
  "source_dataset": "SYNTHETIC",
  "forecast_origin": "2025-12-29",
  "include_extended": false,
  "records": [
    {
      "product_key": "SYN_00001",
      "features": { "lag_1": 120.5, "...": "..." }
    }
  ]
}
```

**Response (200):**
```json
{
  "forecasts": [{ "horizon": 1, "weekly_demand": 118.2, "forecast_status": "PRODUCTION", "...": "..." }],
  "metadata": { "model_id": "phase20_synthetic_lightgbm", "...": "..." },
  "n": 1,
  "generated_at": "..."
}
```

### Phase 20 inventory risk

| Item | Value |
|------|-------|
| **Endpoint** | `POST /phase20/risk/explain` |
| **Purpose** | Stockout/overstock levels, recommended action, projected balance |

**Request:**
```json
{
  "sku_id": "00001",
  "forecast_weekly_demand": 150.0,
  "on_hand_units": 200,
  "on_order_units": 50,
  "lead_time_weeks": 2,
  "safety_stock": 30,
  "reorder_point": 100
}
```

**Response (200):**
```json
{
  "sku_id": "00001",
  "stockout_risk_level": "LOW",
  "overstock_risk_level": "...",
  "recommended_action": "MONITOR",
  "forecast_demand": 150.0,
  "...": "..."
}
```

---

## 3. Error Handling

| Case | Status | Body |
|------|--------|------|
| Missing/invalid API key (auth enabled) | 401 | `{"detail":"Unauthorized"}` |
| Schema validation failure | 422 | `{"detail":"Invalid request schema"}` |
| Missing Phase 20 features | 400 | `{"detail":"Missing required features: [...]"}` |
| Invalid source dataset (Phase 20 + UCI) | 400 | `{"detail":"Phase 20 model only supports source_dataset=SYNTHETIC..."}` |
| Unhandled server error | 500 | `{"detail":"Internal server error"}` — no stack trace exposed |

**Invalid request example (verified locally):**
```bash
curl -X POST https://project-foresight-api-tofn.onrender.com/phase20/forecast \
  -H "Content-Type: application/json" \
  -d '{"source_dataset":"SYNTHETIC","records":[{"product_key":"X","features":{}}]}'
```
Expected after auth: **400** with missing-feature detail (not a crash).

---

## 4. Live Scoring Verification Status

| Check | Live (Render, pre–Phase 23.3 redeploy) | Local (post-fix, auth disabled) |
|-------|----------------------------------------|----------------------------------|
| Forecast `POST /forecast` | **401** — production auth enforced without matching API key | **200** |
| Forecast `POST /phase20/forecast` | **401** | **400** on invalid input (graceful) |
| Risk `POST /phase20/risk/explain` | **401** | **200** |
| Auth `POST /auth/register` | **200** | **200** |
| Auth `POST /auth/login` | **500** — JWT secret env mismatch (`FORESIGHT_API_JWT_SECRET` not read) | **200** after fix |

**Phase 23.3 repository fixes (require Render redeploy):**
1. CORS middleware for `https://foresight-project-green.vercel.app`
2. `auth_is_required()` respects `FORESIGHT_API_AUTH_ENABLED` (matches `.env.example` demo deployment)
3. JWT secret reads `FORESIGHT_API_JWT_SECRET` alias
4. Valid JWT Bearer tokens accepted as alternative to API key

---

## 5. Frontend Configuration

| Setting | Value |
|---------|-------|
| Vercel `vercel.json` | `FORESIGHT_API_URL` / `NEXT_PUBLIC_API_URL` → Render URL |
| `public/index.html` | `window.FORESIGHT_API_URL = "https://project-foresight-api-tofn.onrender.com"` |
| `public/js/app.js` fallback | Same Render URL (no localhost in production bundle) |

The Vercel dashboard uses **client-side demo metrics** for charts; live API calls are currently limited to `/auth/login` and `/auth/register`. Swagger link opens Render `/docs`.

---

## 6. Recommended Render Environment (Post-Redeploy)

```
FORESIGHT_ENV=production
FORESIGHT_API_AUTH_ENABLED=false
FORESIGHT_API_JWT_SECRET=<strong-secret>
FORESIGHT_CORS_ORIGINS=https://foresight-project-green.vercel.app,http://localhost:8501
FORESIGHT_AUTH_DB_PATH=/tmp/foresight_auth.db
```

For API-key–protected production: set `FORESIGHT_API_AUTH_ENABLED=true` and `FORESIGHT_API_API_KEY=<secret>`; `/ready` will then report `config_valid: true`.
