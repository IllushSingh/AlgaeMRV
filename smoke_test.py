from pathlib import Path
import pandas as pd

from modules.model import compute_ndci, fit_biomass_model, calculate_carbon, summarize_carbon, data_quality_score
from modules.farm_intelligence import load_bundle, forecast_week

ROOT = Path(__file__).resolve().parent

df = pd.read_csv(ROOT / "data" / "algaemrv_test_data.csv")
df["date"] = pd.to_datetime(df["date"])
df["ndci"] = compute_ndci(df)
model, cal, r2, rmse = fit_biomass_model(df)
res = calculate_carbon(df, model)
summary = summarize_carbon(res)
dq = data_quality_score(res, r2)

print("AlgaeMRV smoke test")
print("-------------------")
print(f"Rows: {len(res)}")
print(f"Calibration samples: {len(cal)}")
print(f"Calibration R2: {r2:.3f}")
print(f"Gross CO2 fixed: {summary['gross_CO2_fixed_kg']:.2f} kg")
print(f"Net biological balance: {summary['net_biological_balance_kgCO2e']:.2f} kgCO2e")
print(f"Potential durable removal: {summary['potential_durable_removal_kgCO2e']:.2f} kgCO2e")
print(f"Data quality score: {dq}/100")

assert r2 > 0.90
assert res["ndci"].between(-1, 1).all()
assert summary["potential_durable_removal_kgCO2e"] >= 0

model_path = ROOT / "models" / "atp3_forecast.joblib"
if model_path.exists():
    bundle = load_bundle(model_path)
    row = res.iloc[6]
    prev = res.iloc[5]
    state = {
        "biomass_g_l": float(row["estimated_biomass_kg_m3"]),
        "recent_growth_g_l_day": float(row["estimated_biomass_kg_m3"] - prev["estimated_biomass_kg_m3"]),
        "ph": float(row["pH"]),
        "temperature_C": float(row["temperature_C"]),
        "dissolved_oxygen_mg_L": float(row["dissolved_oxygen_mg_L"]),
        "water_depth_m": float(row["water_depth_m"]),
        "culture_age_days": 6.0,
    }
    fc, _ = forecast_week(bundle, state, row["date"])
    assert len(fc) == 7
    assert fc["predicted_biomass_g_l"].ge(0).all()
    print(f"7-day forecast: {fc.iloc[-1]['predicted_biomass_g_l']:.3f} g/L")
else:
    print("Forecast model missing: run python train_atp3_model.py")

print("PASS")
