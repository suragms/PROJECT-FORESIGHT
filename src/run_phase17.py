"""Phase 17 — Run full pipeline."""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.phase17_dataset_ingestion import run_ingestion
from src.phase17_features import run_feature_engineering
from src.phase17_forecasting import run_forecasting
from src.phase17_risk_scoring import run_risk_scoring

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 17 — FULL PIPELINE")
    print("=" * 60 + "\n")

    ingestion = run_ingestion()
    print()

    features = run_feature_engineering()
    print()

    metrics, registry = run_forecasting()
    print()

    risk = run_risk_scoring()

    print("\n" + "=" * 60)
    print("PHASE 17 PIPELINE COMPLETE")
    print("=" * 60)
