import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


def compute_ndci(df, b4_col="sentinel_B4_red_reflectance",
                 b5_col="sentinel_B5_rededge_reflectance"):
    denom = df[b5_col] + df[b4_col]
    return np.where(denom != 0, (df[b5_col] - df[b4_col]) / denom, np.nan)


def fit_biomass_model(df):
    cal = df.dropna(subset=["measured_dry_biomass_kg_m3", "ndci"]).copy()
    if len(cal) < 3:
        raise ValueError("At least 3 calibration samples are required.")
    X = cal[["ndci"]].values
    y = cal["measured_dry_biomass_kg_m3"].values
    model = LinearRegression().fit(X, y)
    pred = model.predict(X)
    r2 = r2_score(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    return model, cal, r2, rmse


def calculate_carbon(df, model, carbon_fraction=0.48,
                     grid_ef=0.70, transport_ef=0.18,
                     uncertainty_deduction=0.10,
                     permanence_factor=0.75):
    """Calculate chronological per-pond biomass and carbon accounting.

    potential_durable_removal_kgCO2e is cumulative-to-date, not a daily amount.
    This prevents negative operating days from disappearing when a period total is
    reported.
    """
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"])
    sort_cols = [c for c in ["pond_id", "date"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    out["estimated_biomass_kg_m3"] = model.predict(out[["ndci"]].values)
    out["pond_volume_m3"] = out["pond_area_m2"] * out["water_depth_m"]
    out["estimated_total_biomass_kg"] = (
        out["estimated_biomass_kg_m3"] * out["pond_volume_m3"]
    )

    if "pond_id" in out.columns:
        group = out.groupby("pond_id", sort=False)
        out["biomass_gain_kg"] = group["estimated_total_biomass_kg"].diff().fillna(0)
    else:
        out["biomass_gain_kg"] = out["estimated_total_biomass_kg"].diff().fillna(0)
    out["positive_biomass_gain_kg"] = out["biomass_gain_kg"].clip(lower=0)

    out["gross_CO2_fixed_kg"] = (
        out["positive_biomass_gain_kg"] * carbon_fraction * (44.0 / 12.0)
    )
    out["electricity_emissions_kgCO2e"] = out["electricity_kWh_day"] * grid_ef
    out["transport_emissions_kgCO2e"] = out["transport_km_day"] * transport_ef
    out["operational_emissions_kgCO2e"] = (
        out["electricity_emissions_kgCO2e"]
        + out["transport_emissions_kgCO2e"]
    )
    out["net_biological_balance_kgCO2e"] = (
        out["gross_CO2_fixed_kg"] - out["operational_emissions_kgCO2e"]
    )

    if "pond_id" in out.columns:
        group = out.groupby("pond_id", sort=False)
        out["cumulative_net_biological_balance_kgCO2e"] = group[
            "net_biological_balance_kgCO2e"
        ].cumsum()
    else:
        out["cumulative_net_biological_balance_kgCO2e"] = out[
            "net_biological_balance_kgCO2e"
        ].cumsum()

    out["potential_durable_removal_kgCO2e"] = (
        out["cumulative_net_biological_balance_kgCO2e"].clip(lower=0)
        * (1.0 - uncertainty_deduction)
        * permanence_factor
    )
    return out


def summarize_carbon(results):
    """Return period totals, using the final cumulative durable value per pond."""
    gross = float(results["gross_CO2_fixed_kg"].sum())
    operational = float(results["operational_emissions_kgCO2e"].sum())
    net = gross - operational
    if "pond_id" in results.columns:
        ordered = results.sort_values(["pond_id", "date"])
        durable = float(
            ordered.groupby("pond_id")["potential_durable_removal_kgCO2e"].last().sum()
        )
    else:
        durable = float(results["potential_durable_removal_kgCO2e"].iloc[-1])
    return {
        "gross_CO2_fixed_kg": gross,
        "operational_emissions_kgCO2e": operational,
        "net_biological_balance_kgCO2e": net,
        "potential_durable_removal_kgCO2e": durable,
    }


def data_quality_score(df, calibration_r2):
    required = [
        "sentinel_B4_red_reflectance", "sentinel_B5_rededge_reflectance",
        "temperature_C", "pH", "dissolved_oxygen_mg_L",
        "water_depth_m", "pond_area_m2"
    ]
    completeness = 1 - df[required].isna().mean().mean()

    plausible = (
        df["temperature_C"].between(20, 35)
        & df["pH"].between(6, 9)
        & df["dissolved_oxygen_mg_L"].between(0, 20)
        & df["water_depth_m"].between(0.05, 2.0)
    ).mean()

    ndci_valid = df["ndci"].between(-1, 1).mean()
    calibration_component = max(0.0, min(1.0, float(calibration_r2)))
    score = 100 * (
        0.35 * completeness
        + 0.25 * plausible
        + 0.15 * ndci_valid
        + 0.25 * calibration_component
    )
    return round(score, 1)
