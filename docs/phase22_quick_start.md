# Phase 22 — Quick Start

Shortest valid workflow to run PROJECT FORESIGHT locally.

## 1. Clone Repository

```bash
git clone https://github.com/suragms/Demand-Inventory-Intelligence.git
cd Demand-Inventory-Intelligence
```

## 2. Create Environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Install Dependencies

Already covered in step 2. Verify:
```bash
python -c "import fastapi, streamlit, lightgbm; print('OK')"
```

## 4. Start API

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

## 5. Start Production Dashboard (new terminal)

```bash
streamlit run dashboard/phase20_production.py
```

## 6. Run Monitoring

```bash
python src/run_phase21.py
```

## 7. Start Executive Dashboard (new terminal)

```bash
streamlit run dashboard/phase22_executive_dashboard.py
```

## 8. Run Tests

```bash
python -m pytest tests -q
```

## 9. Final Audit (optional)

```bash
python src/phase22_final_audit.py
```

## What You Should See

- API docs at `http://127.0.0.1:8000/docs`
- Phase 20 production forecasts for 100 SKUs (6-week horizon)
- Phase 21 monitoring health and alerts
- Phase 22 executive view with validation metrics labeled as **VALIDATION / BACKTEST**

**Note:** Live production performance is **PENDING ACTUALS**.
