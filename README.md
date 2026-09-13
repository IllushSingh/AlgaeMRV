# AlgaeMRV

Satellite-verified carbon accounting for algae farms. Monitor, predict, account.

AlgaeMRV is a Streamlit prototype for the measurement, reporting and verification (MRV)
layer that algae carbon projects need but rarely have. It turns Sentinel-2 red and
red-edge reflectance into a calibrated biomass estimate, forecasts the next seven days
with a model trained on real open pond data, and produces a conservative,
machine-readable carbon ledger that an auditor can pick apart.

Built for HackOut'26.

## What it does

**Monitor.** Each daily observation carries Sentinel-2 B4 (red) and B5 (red-edge)
reflectance. The app computes NDCI, fits a linear NDCI to dry-biomass calibration against
the lab samples in the same file, and reports R², RMSE and a 0 to 100 data quality score
so you can see how much the numbers are worth.

**Predict.** A gradient boosting model trained on the ATP3 Unified Field Studies dataset
forecasts dry biomass one to seven days ahead from the current farm state (recent growth,
pH, temperature, dissolved oxygen, depth, culture age). It predicts a specific growth rate
rather than absolute biomass, which keeps it honest across ponds of different sizes.

**Account.** Biomass gain becomes gross CO₂ fixation at 44/12 times the carbon fraction,
electricity and transport are subtracted as operational emissions, and what is left is
discounted by an uncertainty deduction and a permanence factor chosen from the biomass
fate. Only positive cumulative balance ever converts into a durable removal claim, so a
bad week cannot be netted away by a good one.

The three tabs are **Remote sensing** (NDCI surfaces, pixel distributions, calibration
fit), **Carbon MRV** (period ledger, waterfall, daily breakdown) and **Audit** (the full
verification record plus JSON and CSV exports).

## Quickstart

Requires Python 3.9 or newer.

```bash
git clone https://github.com/IllushSingh/AlgaeMRV.git
cd AlgaeMRV

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m streamlit run app.py
```

The repository ships with a trained model at `models/atp3_forecast.joblib`, so the app
works immediately. On Windows you can skip the steps above and double-click
`run_windows.bat`, which activates the `semlab` conda environment, installs
requirements, trains the model if it is missing, and launches Streamlit.

Check the whole pipeline without the UI:

```bash
python smoke_test.py
```

## Command line tools

Forecast seven days from a single farm-state record:

```bash
python predict_farm.py --input data/farm_state_example.json
python predict_farm.py --input data/farm_state_example.json --output forecast.csv
```

Retrain the forecast model from the bundled ATP3 data (`retrain_windows.bat` on Windows):

```bash
python train_atp3_model.py
```

Training writes `models/atp3_forecast.joblib`, a metadata sidecar at
`models/atp3_forecast.json`, held-out metrics to `reports/training_metrics.json` and
per-row test predictions to `reports/test_predictions.csv`.

## The forecast model

| | |
|---|---|
| Algorithm | `GradientBoostingRegressor` with huber loss, median imputation |
| Target | specific growth rate, `ln(B_future / B_current) / horizon` |
| Features | recent growth, pH, temperature, dissolved oxygen, water depth, culture age, horizon |
| Horizons | 1 to 7 days |
| Training pairs | 6,233 origin-to-future pairs across 403 pond runs and 11 campaigns |
| Validation | two latest whole campaigns held out (MAY052015, JUN172015), 1,065 rows |
| Held-out MAE | 0.057 g/L, against a persistence baseline of 0.116 g/L |

Accuracy by horizon, held-out campaigns only:

| Horizon | MAE (g/L) | Persistence MAE (g/L) | n |
|---|---|---|---|
| +1 day | 0.030 | 0.043 | 248 |
| +2 days | 0.041 | 0.079 | 285 |
| +3 days | 0.053 | 0.111 | 179 |
| +4 days | 0.082 | 0.165 | 99 |
| +5 days | 0.087 | 0.185 | 127 |
| +6 days | 0.077 | 0.261 | 36 |
| +7 days | 0.107 | 0.227 | 91 |

The model beats persistence at every horizon, roughly halving the error, but R² decays
badly past day three and is negative at day seven. Treat days one to three as a planning
signal and the rest as a trend, not a number to bank on.

Two choices matter for whether you trust this. The split holds out entire experiment
campaigns rather than random rows, so nothing from a test pond leaks into training
through a neighbouring day. And any pair spanning a recorded harvest, crash or reset is
dropped, which excluded 10,301 candidate pairs. Without that, the model would have learnt
to predict harvests it cannot see coming.

## Repository layout

```
app.py                      Streamlit dashboard, all styling included
train_atp3_model.py         Trains and evaluates the ATP3 forecast model
predict_farm.py             CLI 7-day forecast from one JSON farm state
smoke_test.py               End-to-end check with assertions
modules/
  model.py                  NDCI, calibration, carbon accounting, data quality
  atp3_data.py              ATP3 ingestion, pair building, campaign split
  farm_intelligence.py      Model loading, input validation, 7-day forecast
data/
  algaemrv_test_data.csv    14 days of demo pond observations
  algaemrv_spatial_pixels.csv  24 x 24 pixel NDCI captures
  farm_state_example.json   Example input for predict_farm.py
  atp3/                     ATP3 UFS summary, instrumentation and harvest data
models/                     Trained model bundle and metadata
reports/                    Training metrics and test predictions
```

## Bringing your own data

The sidebar takes any CSV with the same schema as
[algaemrv_test_data.csv](data/algaemrv_test_data.csv):

| Column | Meaning |
|---|---|
| `date`, `pond_id` | Observation day and pond identifier |
| `sentinel_B4_red_reflectance`, `sentinel_B5_rededge_reflectance` | Sentinel-2 bands used for NDCI |
| `temperature_C`, `pH`, `dissolved_oxygen_mg_L`, `EC_mS_cm` | Water chemistry |
| `water_depth_m`, `pond_area_m2` | Geometry, used for pond volume |
| `electricity_kWh_day`, `transport_km_day` | Operational emission drivers |
| `measured_dry_biomass_kg_m3` | Lab samples, at least 3 needed to calibrate |

Rows without a lab measurement are fine and expected. Calibration uses the days that have
one; every day gets an estimate.

Carbon assumptions are all adjustable in the sidebar: carbon fraction (default 0.48),
grid factor (0.70 kg CO₂e/kWh), transport factor (0.18 kg CO₂e/km), uncertainty deduction
(10%) and biomass fate, which sets permanence from 0.90 for long-term storage down to
0.05 for biofuel.

## Limitations

- Demo satellite imagery is interpolated between weekly captures, not live Sentinel-2 downloads.
- The NDCI to biomass calibration is linear and pond-specific. It does not transfer between sites.
- Forecast accuracy past day three is weak, as the table above shows.
- Nothing here is validated against a registry methodology. It is a prototype of the workflow, not a certified one.

If `smoke_test.py` prints scikit-learn `InconsistentVersionWarning` messages, the bundled
model was pickled with a newer scikit-learn than your environment has. Results still
match, and `python train_atp3_model.py` regenerates the bundle locally to silence it.

## Data

Pond data comes from the [ATP3 Unified Field Studies](https://atp3.org/)
dataset, an open multi-site outdoor algae cultivation campaign run from 2013 to 2015.
Column definitions are in [data/definitions/](data/definitions/). Monitoring and spatial
CSVs in `data/` are synthetic demonstration data.
