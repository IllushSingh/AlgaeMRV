from pathlib import Path
import numpy as np
import pandas as pd

KEY = ["SiteID", "ExperimentID", "PondID", "StrainID"]
RUN = KEY + ["BatchID"]

FEATURES = [
    "recent_growth_g_l_day",
    "ph",
    "temperature_C",
    "dissolved_oxygen_mg_L",
    "water_depth_m",
    "culture_age_days",
    "horizon_days",
]

BOUNDS = {
    "biomass_g_l": (0.0, 10.0),
    "ph": (0.0, 14.0),
    "temperature_C": (-5.0, 60.0),
    "dissolved_oxygen_mg_L": (0.0, 60.0),
    "water_depth_m": (0.0, 3.0),
    "culture_age_days": (0.0, 365.0),
}


def _require(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def prepare_forecast_pairs(data_dir, max_horizon=7):
    """Return historical origin->future-growth pairs for horizons 1..max_horizon.

    The target is future dry-biomass change (g/L), not future absolute biomass.
    Pairs crossing a recorded harvest/crash/reset are excluded.
    """
    data_dir = Path(data_dir)
    summary_path = data_dir / "ATP3-UFS-SummaryCombinedData.csv"
    sensor_path = data_dir / "ATP3-UFS-Instrumentation-daily.csv"
    harvest_path = data_dir / "ATP3-UFS-HarvestData.csv"

    s = pd.read_csv(summary_path, low_memory=False).drop_duplicates().copy()
    sensors = pd.read_csv(sensor_path, low_memory=False).copy()
    harvest = pd.read_csv(harvest_path, low_memory=False).copy()

    _require(
        s,
        RUN + ["DateTime", "DW.g.L", "Depth.cm", "Harvest.Vol..L.", "crash"],
        summary_path.name,
    )
    _require(
        sensors,
        KEY
        + [
            "Date",
            "pH",
            "Temp.avg (C)",
            "DO (mg.L)",
        ],
        sensor_path.name,
    )
    _require(harvest, KEY + ["Date"], harvest_path.name)

    for frame in (s, sensors, harvest):
        for col in KEY:
            frame[col] = frame[col].astype("string").str.strip()

    s["timestamp"] = pd.to_datetime(s["DateTime"], errors="coerce")
    s["date"] = s["timestamp"].dt.normalize()
    sensors["date"] = pd.to_datetime(
        sensors["Date"], format="%m-%d-%Y", errors="coerce"
    )
    harvest["timestamp"] = pd.to_datetime(
        harvest["Date"].astype(str).str.strip(),
        format="%m-%d-%Y %H:%M",
        errors="coerce",
    )
    harvest["date"] = harvest["timestamp"].dt.normalize()

    if sensors["date"].isna().any() or harvest["date"].isna().any():
        raise ValueError("ATP3 instrumentation/harvest contains unparseable dates.")
    if sensors.duplicated(KEY + ["date"]).any():
        raise ValueError("ATP3 daily sensor keys are not unique.")

    valid = s[
        s["PondID"].str.fullmatch(r"P\d+").fillna(False)
        & s["date"].notna()
        & s["BatchID"].notna()
    ].copy()
    valid["biomass_g_l"] = pd.to_numeric(valid["DW.g.L"], errors="coerce")
    valid["depth_cm"] = pd.to_numeric(valid["Depth.cm"], errors="coerce")
    valid["sample_ph"] = pd.to_numeric(valid["pH"], errors="coerce")
    valid["sample_temp_c"] = pd.to_numeric(valid["Temp.C"], errors="coerce")
    valid = valid[valid["biomass_g_l"].gt(0)].copy()

    latest = valid.groupby(KEY + ["date"])["timestamp"].transform("max")
    valid = valid[valid["timestamp"].eq(latest)]
    batch_conflict = valid.groupby(KEY + ["date"])["BatchID"].nunique()
    if batch_conflict.gt(1).any():
        raise ValueError("More than one batch is present at the final same-day sample.")

    day = (
        valid.groupby(RUN + ["date"], as_index=False)
        .agg(
            biomass_g_l=("biomass_g_l", "mean"),
            depth_cm=("depth_cm", "mean"),
            sample_ph=("sample_ph", "mean"),
            sample_temp_c=("sample_temp_c", "mean"),
        )
        .copy()
    )

    sensor_daily = sensors[
        KEY + ["date", "pH", "Temp.avg (C)", "DO (mg.L)"]
    ].rename(
        columns={
            "pH": "sensor_ph",
            "Temp.avg (C)": "sensor_temp_c",
            "DO (mg.L)": "dissolved_oxygen_mg_L",
        }
    )
    day = day.merge(sensor_daily, on=KEY + ["date"], how="left", validate="many_to_one")
    day["ph"] = pd.to_numeric(day["sensor_ph"], errors="coerce").combine_first(
        pd.to_numeric(day["sample_ph"], errors="coerce")
    )
    day["temperature_C"] = pd.to_numeric(day["sensor_temp_c"], errors="coerce").combine_first(
        pd.to_numeric(day["sample_temp_c"], errors="coerce")
    )
    day["water_depth_m"] = day["depth_cm"] / 100.0
    day = day.sort_values(RUN + ["date"]).reset_index(drop=True)

    for col in [
        "biomass_g_l",
        "ph",
        "temperature_C",
        "dissolved_oxygen_mg_L",
            "water_depth_m",
    ]:
        day[col] = pd.to_numeric(day[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if col in BOUNDS:
            lo, hi = BOUNDS[col]
            day.loc[day[col].notna() & ~day[col].between(lo, hi), col] = np.nan

    grouped = day.groupby(RUN, sort=False)
    day["previous_biomass_g_l"] = grouped["biomass_g_l"].shift(1)
    day["previous_date"] = grouped["date"].shift(1)
    day["previous_gap_days"] = (day["date"] - day["previous_date"]).dt.days
    day["recent_growth_g_l_day"] = (
        day["biomass_g_l"] - day["previous_biomass_g_l"]
    ) / day["previous_gap_days"]
    day["culture_age_days"] = grouped["date"].transform(lambda x: (x - x.min()).dt.days)

    events = pd.concat(
        [
            harvest[KEY + ["date"]],
            s.loc[s["Harvest.Vol..L."].notna() | s["crash"].notna(), KEY + ["date"]],
        ],
        ignore_index=True,
    ).dropna().drop_duplicates()
    event_lookup = {key: set(g["date"]) for key, g in events.groupby(KEY)}

    def has_event(row, start, end):
        if pd.isna(start) or pd.isna(end):
            return False
        dates = event_lookup.get(tuple(row[k] for k in KEY), set())
        return any(start <= event_date <= end for event_date in dates)

    stale_history = day["previous_gap_days"].gt(7) | day.apply(
        lambda r: has_event(r, r["previous_date"], r["date"]), axis=1
    )
    day.loc[
        stale_history,
        ["previous_biomass_g_l", "previous_gap_days", "recent_growth_g_l_day"],
    ] = np.nan

    targets = day[RUN + ["date", "biomass_g_l"]].rename(
        columns={"date": "target_date", "biomass_g_l": "target_biomass_g_l"}
    )

    pair_frames = []
    excluded_events = 0
    horizon_counts = {}
    for horizon in range(1, max_horizon + 1):
        origins = day.copy()
        origins["horizon_days"] = horizon
        origins["target_date"] = origins["date"] + pd.Timedelta(days=horizon)
        pairs = origins.merge(
            targets,
            on=RUN + ["target_date"],
            how="inner",
            validate="one_to_one",
        )
        crosses_event = pairs.apply(
            lambda r: has_event(r, r["date"], r["target_date"]), axis=1
        )
        excluded_events += int(crosses_event.sum())
        pairs = pairs[~crosses_event].copy()
        pairs["target_specific_growth_day"] = (
            np.log(pairs["target_biomass_g_l"] / pairs["biomass_g_l"])
            / pairs["horizon_days"]
        )
        horizon_counts[str(horizon)] = int(len(pairs))
        pair_frames.append(pairs)

    all_pairs = pd.concat(pair_frames, ignore_index=True)
    all_pairs["run_id"] = all_pairs[RUN].astype(str).agg("|".join, axis=1)

    audit = {
        "eligible_pairs": int(len(all_pairs)),
        "eligible_runs": int(all_pairs["run_id"].nunique()),
        "campaigns": int(all_pairs["ExperimentID"].nunique()),
        "pairs_by_horizon": horizon_counts,
        "pairs_excluded_for_harvest_or_crash": excluded_events,
        "feature_missing_fraction": all_pairs[FEATURES].isna().mean().to_dict(),
        "date_min": str(all_pairs["date"].min().date()),
        "date_max": str(all_pairs["target_date"].max().date()),
    }
    return all_pairs, day, audit


def train_test_split_campaigns(pairs, test_campaign_count=2):
    """Hold out the latest whole ATP3 experiment campaigns for evaluation."""
    campaigns = (
        pairs.groupby("ExperimentID")["date"].min().sort_values().index.tolist()
    )
    if len(campaigns) <= test_campaign_count + 2:
        raise ValueError("Not enough ATP3 campaigns for a whole-campaign holdout.")

    test_campaigns = campaigns[-test_campaign_count:]
    train_campaigns = campaigns[:-test_campaign_count]
    train = pairs[pairs["ExperimentID"].isin(train_campaigns)].copy()
    test = pairs[pairs["ExperimentID"].isin(test_campaigns)].copy()

    test_start = test["date"].min()
    train = train[train["target_date"] < test_start].copy()
    if len(train) < 200 or len(test) < 100:
        raise ValueError("Too few rows remain after the campaign split.")

    return train, test, {
        "train_campaigns": train_campaigns,
        "test_campaigns": test_campaigns,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
    }
