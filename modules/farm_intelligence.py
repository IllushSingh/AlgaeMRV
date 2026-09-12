"""One ATP3-trained prediction engine for 1-7 day dry-biomass forecasts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from .atp3_data import FEATURES, BOUNDS


def load_bundle(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = joblib.load(path)
    if bundle.get("schema_version") != 2:
        raise ValueError("Unsupported forecast model schema. Retrain with train_atp3_model.py.")
    if bundle.get("features") != FEATURES:
        raise ValueError("Forecast feature schema changed. Retrain the model.")
    return bundle


def _clean_state(bundle, state):
    row = {}
    warnings = []
    current_biomass = state.get("biomass_g_l", np.nan)
    try:
        current_biomass = float(current_biomass)
    except (TypeError, ValueError):
        raise ValueError("biomass_g_l must be numeric.") from None
    if not np.isfinite(current_biomass) or current_biomass <= 0:
        raise ValueError("Current dry biomass must be positive (g/L).")
    row["biomass_g_l"] = current_biomass

    for feature in FEATURES:
        if feature == "horizon_days":
            continue
        value = state.get(feature, np.nan)
        try:
            row[feature] = float(value) if value is not None else np.nan
        except (TypeError, ValueError):
            raise ValueError(f"{feature} must be numeric or empty.") from None
        if not np.isnan(row[feature]) and not np.isfinite(row[feature]):
            raise ValueError(f"{feature} must be finite.")

    for feature, (low, high) in BOUNDS.items():
        value = row.get(feature, state.get(feature, np.nan))
        if np.isfinite(value) and not low <= value <= high:
            raise ValueError(
                f"{feature}={value:g} is outside the supported screening range [{low:g}, {high:g}]."
            )

    missing = [
        feature
        for feature in FEATURES
        if feature != "horizon_days" and np.isnan(row.get(feature, np.nan))
    ]
    if missing:
        warnings.append("Missing inputs imputed from ATP3 training data: " + ", ".join(missing))

    unusual = []
    for feature, limits in bundle.get("reference_ranges", {}).items():
        value = row.get(feature, state.get(feature, np.nan))
        if np.isfinite(value) and (value < limits[0] or value > limits[1]):
            unusual.append(feature)
    if unusual:
        warnings.append(
            "Outside the central ATP3 training range: " + ", ".join(unusual)
        )
    return row, warnings


def forecast_week(bundle, state, observation_date):
    """Predict dry biomass for each of the next seven calendar days."""
    origin = pd.Timestamp(observation_date)
    if pd.isna(origin):
        raise ValueError("A valid observation date is required.")

    row, warnings = _clean_state(bundle, state)
    rows = []
    for horizon in range(1, 8):
        features = row.copy()
        features["horizon_days"] = float(horizon)
        frame = pd.DataFrame([features], columns=FEATURES)
        specific_growth = float(bundle["model"].predict(frame)[0])
        predicted_biomass = max(
            0.0,
            row["biomass_g_l"] * np.exp(specific_growth * horizon),
        )
        predicted_delta = predicted_biomass - row["biomass_g_l"]
        horizon_mae = bundle.get("mae_by_horizon_g_l", {}).get(str(horizon))
        rows.append(
            {
                "horizon_days": horizon,
                "date": origin + pd.Timedelta(days=horizon),
                "predicted_specific_growth_day": specific_growth,
                "predicted_change_g_l": predicted_delta,
                "predicted_biomass_g_l": predicted_biomass,
                "validation_mae_g_l": horizon_mae,
            }
        )

    return pd.DataFrame(rows), warnings


def model_summary(bundle):
    """Small metadata view for an optional technical details panel."""
    return {
        "model": bundle.get("model_name", "HistGradientBoostingRegressor"),
        "training_pairs": bundle.get("training_pairs"),
        "historical_runs": bundle.get("historical_runs"),
        "heldout_mae_g_l": bundle.get("evaluation", {}).get("mae_g_l"),
        "heldout_r2": bundle.get("evaluation", {}).get("r2"),
        "heldout_persistence_mae_g_l": bundle.get("evaluation", {}).get(
            "persistence_mae_g_l"
        ),
        "test_campaigns": bundle.get("test_campaigns"),
    }
