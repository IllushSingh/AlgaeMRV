"""Generate a seven-day forecast from one JSON farm-state record.

Example:
    python predict_farm.py --input data/farm_state_example.json --output forecast.csv
"""
from pathlib import Path
import argparse
import json
from modules.farm_intelligence import load_bundle, forecast_week

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models" / "atp3_forecast.joblib"
    )
    args = parser.parse_args()

    record = json.loads(args.input.read_text(encoding="utf-8"))
    forecast, warnings = forecast_week(
        load_bundle(args.model), record["state"], record["date"]
    )
    if warnings:
        for warning in warnings:
            print("WARNING:", warning)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix.lower() == ".json":
            payload = forecast.copy()
            payload["date"] = payload["date"].dt.strftime("%Y-%m-%d")
            args.output.write_text(
                json.dumps(payload.to_dict("records"), indent=2), encoding="utf-8"
            )
        else:
            forecast.to_csv(args.output, index=False)
        print(f"Saved {args.output}")
    else:
        print(forecast.to_string(index=False))


if __name__ == "__main__":
    main()
