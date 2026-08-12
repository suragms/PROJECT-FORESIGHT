"""
Phases 7 & 8 — Demand Forecasting Engine
==========================================
Project FORESIGHT: Demand & Inventory Intelligence

Implements Baseline models (Naive, Moving Averages, Seasonal Naive) and
Gradient Boosted ML models (LightGBM, XGBoost, Random Forest) for multi-horizon
retail demand forecasting with interactive scenario simulation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

from src.feature_engineering import build_forecasting_feature_matrix
from src.evaluation import calculate_metrics, compare_models

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "forecasts")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Key feature sets for tree-based ML models
FEATURE_COLS = [
    "month", "quarter", "day_of_month", "day_of_week", "is_weekend",
    "sin_day_of_week", "cos_day_of_week", "sin_month", "cos_month",
    "discount_pct", "promotion_flag",
    "units_sold_lag_1", "units_sold_lag_2", "units_sold_lag_3", "units_sold_lag_7",
    "units_sold_lag_14", "units_sold_lag_21", "units_sold_lag_28", "units_sold_lag_30",
    "units_sold_rolling_mean_7", "units_sold_rolling_std_7",
    "units_sold_rolling_mean_14", "units_sold_rolling_mean_30",
    "units_sold_ewm_7", "units_sold_ewm_28",
]


class BaselineForecaster:
    """Statistical & heuristic time series baseline models."""

    @staticmethod
    def naive_forecast(series: pd.Series, horizon: int = 30) -> np.ndarray:
        """Repeat the last observed value."""
        last_val = series.iloc[-1] if len(series) else 0.0
        return np.full(horizon, float(last_val))

    @staticmethod
    def moving_average_forecast(series: pd.Series, window: int = 7, horizon: int = 30) -> np.ndarray:
        """Mean of the last `window` observations."""
        mean_val = series.iloc[-window:].mean() if len(series) >= window else series.mean()
        return np.full(horizon, float(mean_val if not pd.isna(mean_val) else 0.0))

    @staticmethod
    def seasonal_naive_forecast(series: pd.Series, season_length: int = 7, horizon: int = 30) -> np.ndarray:
        """Repeat the last observed seasonal cycle (e.g. day of week)."""
        if len(series) < season_length:
            return BaselineForecaster.naive_forecast(series, horizon)
        recent_cycle = series.iloc[-season_length:].to_numpy()
        repeats = int(np.ceil(horizon / season_length))
        return np.tile(recent_cycle, repeats)[:horizon]


class MLDemandForecaster:
    """Machine Learning forecasting wrapper supporting LightGBM, XGBoost, and RF."""

    def __init__(self, model_type: str = "lightgbm", random_state: int = 42):
        self.model_type = model_type.lower()
        self.random_state = random_state
        self.model = None
        self.feature_names = FEATURE_COLS

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """Fit the chosen ML model."""
        X = X_train[self.feature_names].copy()
        y = np.asarray(y_train, dtype=float)

        if self.model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1,
                **kwargs
            )
            self.model.fit(X, y)
        elif self.model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=6,
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs
            )
            self.model.fit(X, y)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=self.random_state,
                n_jobs=-1,
                **kwargs
            )
            self.model.fit(X, y)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict non-negative demand values."""
        X_mat = X[self.feature_names].copy()
        preds = self.model.predict(X_mat)
        return np.maximum(0, preds)

    def get_feature_importances(self) -> pd.DataFrame:
        """Return a sorted DataFrame of feature importances."""
        if self.model is None:
            return pd.DataFrame()
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            return pd.DataFrame({
                "feature": self.feature_names,
                "importance": imp
            }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return pd.DataFrame()

    def save(self, filepath: str = None):
        """Save model to disk."""
        if filepath is None:
            filepath = os.path.join(MODELS_DIR, f"{self.model_type}_forecaster.joblib")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str):
        """Load model from disk."""
        return joblib.load(filepath)


def train_and_benchmark_models(
    feature_matrix: pd.DataFrame,
    test_days: int = 30,
) -> tuple[dict[str, MLDemandForecaster], pd.DataFrame, dict]:
    """
    Train and benchmark all models using time-series split.
    Returns trained models, evaluation leaderboard, and test predictions.
    """
    df = feature_matrix.sort_values(by="date").reset_index(drop=True)
    split_date = df["date"].max() - pd.Timedelta(days=test_days)

    train_df = df[df["date"] <= split_date].copy()
    test_df = df[df["date"] > split_date].copy()

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["units_sold"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["units_sold"].to_numpy()

    models = {}
    preds_dict = {}

    # 1. Train ML Models
    for model_type in ["lightgbm", "xgboost", "random_forest"]:
        forecaster = MLDemandForecaster(model_type=model_type)
        forecaster.fit(X_train, y_train)
        preds = forecaster.predict(X_test)
        forecaster.save()
        models[model_type] = forecaster
        preds_dict[model_type.upper().replace("_", " ")] = (y_test, preds)

    # 2. Baseline models on test set
    # Using 7-day moving average and 7-day seasonal lag from training tail
    naive_pred = np.full(len(y_test), float(y_train.iloc[-1]))
    ma_7_pred = np.full(len(y_test), float(y_train.iloc[-7:].mean()))
    seasonal_7_pred = np.tile(y_train.iloc[-7:].to_numpy(), int(np.ceil(len(y_test)/7)))[:len(y_test)]

    preds_dict["NAIVE"] = (y_test, naive_pred)
    preds_dict["MOVING AVERAGE (7D)"] = (y_test, ma_7_pred)
    preds_dict["SEASONAL NAIVE (7D)"] = (y_test, seasonal_7_pred)

    leaderboard = compare_models(preds_dict)

    # Save leaderboard report
    leaderboard.to_csv(os.path.join(OUTPUTS_DIR, "model_benchmark_leaderboard.csv"), index=False)

    return models, leaderboard, {
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "split_date": str(split_date)[:10],
        "test_df": test_df,
    }


def generate_multi_step_forecast(
    model: MLDemandForecaster,
    history_df: pd.DataFrame,
    horizon_days: int = 30,
    scenario_discount_pct: float = None,
    scenario_promo_flag: int = None,
    scenario_surge_factor: float = 1.0,
) -> pd.DataFrame:
    """
    Generate recursive multi-step future demand forecast with optional
    scenario simulation (pricing discount adjustment, promotion flag, surge multiplier).
    """
    hist = history_df.sort_values(by="date").copy()
    last_date = hist["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    forecast_records = []
    # Running buffer of recent sales units for recursive lag computation
    buffer_units = list(hist["units_sold"].iloc[-35:].values)

    for i, f_date in enumerate(future_dates):
        # Base calendar features
        dt = f_date
        dow = dt.dayofweek
        month = dt.month
        is_weekend = int(dow in [5, 6])
        sin_dow = np.sin(2 * np.pi * dow / 7)
        cos_dow = np.cos(2 * np.pi * dow / 7)
        sin_m = np.sin(2 * np.pi * month / 12)
        cos_m = np.cos(2 * np.pi * month / 12)

        # Lags from running buffer
        lag_1 = buffer_units[-1]
        lag_2 = buffer_units[-2] if len(buffer_units) >= 2 else lag_1
        lag_3 = buffer_units[-3] if len(buffer_units) >= 3 else lag_1
        lag_7 = buffer_units[-7] if len(buffer_units) >= 7 else lag_1
        lag_14 = buffer_units[-14] if len(buffer_units) >= 14 else lag_1
        lag_21 = buffer_units[-21] if len(buffer_units) >= 21 else lag_1
        lag_28 = buffer_units[-28] if len(buffer_units) >= 28 else lag_1
        lag_30 = buffer_units[-30] if len(buffer_units) >= 30 else lag_1

        roll_7 = float(np.mean(buffer_units[-7:]))
        std_7 = float(np.std(buffer_units[-7:]))
        roll_14 = float(np.mean(buffer_units[-14:]))
        roll_30 = float(np.mean(buffer_units[-30:]))
        ewm_7 = float(pd.Series(buffer_units[-14:]).ewm(span=7).mean().iloc[-1])
        ewm_28 = float(pd.Series(buffer_units[-35:]).ewm(span=28).mean().iloc[-1])

        discount_val = scenario_discount_pct if scenario_discount_pct is not None else float(hist["discount_pct"].iloc[-1] if "discount_pct" in hist.columns else 0.0)
        promo_val = scenario_promo_flag if scenario_promo_flag is not None else int(hist["promotion_flag"].iloc[-1] if "promotion_flag" in hist.columns else 0)

        feat_dict = {
            "month": month,
            "quarter": dt.quarter,
            "day_of_month": dt.day,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "sin_day_of_week": sin_dow,
            "cos_day_of_week": cos_dow,
            "sin_month": sin_m,
            "cos_month": cos_m,
            "discount_pct": discount_val,
            "promotion_flag": promo_val,
            "units_sold_lag_1": lag_1,
            "units_sold_lag_2": lag_2,
            "units_sold_lag_3": lag_3,
            "units_sold_lag_7": lag_7,
            "units_sold_lag_14": lag_14,
            "units_sold_lag_21": lag_21,
            "units_sold_lag_28": lag_28,
            "units_sold_lag_30": lag_30,
            "units_sold_rolling_mean_7": roll_7,
            "units_sold_rolling_std_7": std_7,
            "units_sold_rolling_mean_14": roll_14,
            "units_sold_rolling_mean_30": roll_30,
            "units_sold_ewm_7": ewm_7,
            "units_sold_ewm_28": ewm_28,
        }

        row_df = pd.DataFrame([feat_dict])
        pred_base = float(model.predict(row_df)[0])
        pred_val = float(pred_base * scenario_surge_factor)

        # Upper and lower confidence bounds (approx +/- 1.96 std error)
        std_est = max(1.0, std_7 * np.sqrt(1 + (i / 10.0)))
        lower_bound = max(0.0, pred_val - 1.96 * std_est)
        upper_bound = pred_val + 1.96 * std_est

        forecast_records.append({
            "date": f_date,
            "forecast_units": round(pred_val, 2),
            "forecast_lower": round(lower_bound, 2),
            "forecast_upper": round(upper_bound, 2),
            "step": i + 1,
            "discount_applied": discount_val,
            "promo_applied": promo_val,
        })

        # Append to buffer for next step
        buffer_units.append(pred_val)

    return pd.DataFrame(forecast_records)
