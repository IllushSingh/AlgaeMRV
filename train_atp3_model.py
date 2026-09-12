
from pathlib import Path
import argparse
import json
import platform
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modules.atp3_data import (
    FEATURES,
    prepare_forecast_pairs,
    train_test_split_campaigns,
)

ROOT = Path(__file__).resolve().parent
SEED = 42


def make_model():
    # Gradient Boost was chosen for the task due to best accuracy, rest were discarded. Missing historical measurements are median-imputed.
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=250,
                    max_depth=2,
                    learning_rate=0.03,
                    min_samples_leaf=15,
                    loss="huber",
                    random_state=SEED,
                ),
            ),
        ]
    )


def evaluate(model, frame):
    specific_growth = model.predict(frame[FEATURES])
    predicted = np.maximum(
        0.0,
        frame["biomass_g_l"].to_numpy()
        * np.exp(specific_growth * frame["horizon_days"].to_numpy()),
    )
    actual = frame["target_biomass_g_l"].to_numpy()
    persistence = frame["biomass_g_l"].to_numpy()
    metrics = {
        "n": int(len(frame)),
        "mae_g_l": float(mean_absolute_error(actual, predicted)),
        "rmse_g_l": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
        "persistence_mae_g_l": float(mean_absolute_error(actual, persistence)),
    }
    return predicted, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "atp3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    pairs, _, audit = prepare_forecast_pairs(args.data_dir, max_horizon=7)
    train, test, split_info = train_test_split_campaigns(pairs, test_campaign_count=2)

    evaluation_model = make_model()
    eval_low = float(train["target_specific_growth_day"].quantile(0.05))
    eval_high = float(train["target_specific_growth_day"].quantile(0.95))
    evaluation_target = train["target_specific_growth_day"].clip(eval_low, eval_high)
    evaluation_model.fit(train[FEATURES], evaluation_target)
    test_prediction, evaluation = evaluate(evaluation_model, test)

    per_horizon = {}
    prediction_rows = test[
        [
            "ExperimentID",
            "PondID",
            "StrainID",
            "date",
            "target_date",
            "horizon_days",
            "biomass_g_l",
            "target_biomass_g_l",
        ]
    ].copy()
    prediction_rows["predicted_biomass_g_l"] = test_prediction
    prediction_rows["error_g_l"] = (
        prediction_rows["predicted_biomass_g_l"]
        - prediction_rows["target_biomass_g_l"]
    )

    for horizon, group in prediction_rows.groupby("horizon_days"):
        actual = group["target_biomass_g_l"]
        pred = group["predicted_biomass_g_l"]
        persistence = group["biomass_g_l"]
        per_horizon[str(int(horizon))] = {
            "n": int(len(group)),
            "mae_g_l": float(mean_absolute_error(actual, pred)),
            "persistence_mae_g_l": float(
                mean_absolute_error(actual, persistence)
            ),
            "r2": float(r2_score(actual, pred)) if len(group) > 1 else None,
        }

    final_model = make_model()
    final_low = float(pairs["target_specific_growth_day"].quantile(0.05))
    final_high = float(pairs["target_specific_growth_day"].quantile(0.95))
    final_target = pairs["target_specific_growth_day"].clip(final_low, final_high)
    final_model.fit(pairs[FEATURES], final_target)

    reference_ranges = {}
    for feature in FEATURES:
        if feature == "horizon_days":
            continue
        series = pd.to_numeric(pairs[feature], errors="coerce").dropna()
        if len(series):
            reference_ranges[feature] = [
                float(series.quantile(0.01)),
                float(series.quantile(0.99)),
            ]

    metadata = {
        "schema_version": 2,
        "model_name": "GradientBoostingRegressor",
        "target": "specific dry-biomass growth rate per day, ln(B_future/B_current)/horizon",
        "forecast_horizons_days": [1, 2, 3, 4, 5, 6, 7],
        "features": FEATURES,
        "training_pairs": int(len(pairs)),
        "historical_runs": int(pairs["run_id"].nunique()),
        "reference_ranges": reference_ranges,
        "mae_by_horizon_g_l": {
            horizon: values["mae_g_l"] for horizon, values in per_horizon.items()
        },
        "evaluation": evaluation,
        "test_campaigns": split_info["test_campaigns"],
        "growth_target_clip_5_95": [final_low, final_high],
        "seed": SEED,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
    }

    bundle = dict(metadata)
    bundle["model"] = final_model
    model_path = args.output_dir / "atp3_forecast.joblib"
    joblib.dump(bundle, model_path, compress=3)
    (args.output_dir / "atp3_forecast.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    report = {
        "model": metadata["model_name"],
        "features": FEATURES,
        "evaluation": evaluation,
        "per_horizon": per_horizon,
        "split": split_info,
        "data_audit": audit,
        "note": (
            "Metrics come from the two latest held-out ATP3 experiment campaigns. "
            "The saved deployment model is then refit on all available ATP3 pairs."
        ),
    }
    (args.report_dir / "training_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    prediction_rows.to_csv(args.report_dir / "test_predictions.csv", index=False)

    print("\nATP3 forecast model trained")
    print("---------------------------")
    print(f"Historical pairs: {len(pairs)}")
    print(f"Historical runs:  {pairs['run_id'].nunique()}")
    print(f"Held-out MAE:     {evaluation['mae_g_l']:.4f} g/L")
    print(f"Held-out R2:      {evaluation['r2']:.3f}")
    print(f"Persistence MAE:  {evaluation['persistence_mae_g_l']:.4f} g/L")
    print("\nMAE by forecast horizon")
    for horizon in range(1, 8):
        row = per_horizon[str(horizon)]
        print(
            f"  +{horizon} day: {row['mae_g_l']:.4f} g/L "
            f"(n={row['n']}, persistence={row['persistence_mae_g_l']:.4f})"
        )
    print(f"\nSaved: {model_path}")
    print(f"Report: {args.report_dir / 'training_metrics.json'}")


if __name__ == "__main__":
    main()
