import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from retail_forecast.data import Dataset
from retail_forecast.model import ForecastModel, ModelConfig, train_and_evaluate


def main():
    parser = argparse.ArgumentParser(description="Train or score the Kaggle retail panel locally")
    parser.add_argument("command", choices=["train", "predict"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-dir", help="Trusted model artifact directory for predict")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--trees", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    data = Dataset.from_directory(args.data_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "train":
        config = ModelConfig(
            folds=args.folds, horizon=args.horizon, n_estimators=args.trees, n_jobs=args.threads
        )
        model, metadata, backtests = train_and_evaluate(data, config)
        metadata["input_sha256"] = {
            file.name: hashlib.file_digest(file.open("rb"), "sha256").hexdigest()
            for file in sorted(Path(args.data_dir).glob("*.csv"))
        }
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
        metadata["git_revision"] = revision.stdout.strip() or "uncommitted"
        model.save(output / "model", metadata)
        backtests.to_csv(output / "backtests.csv", index=False)
        print(
            json.dumps({"selected_model": model.model_name, "mean_rmsle": metadata["mean_rmsle"]})
        )
    else:
        if not args.model_dir:
            parser.error("predict requires --model-dir")
        model = ForecastModel.load(args.model_dir)
    forecasts = model.predict(data.train, data.test)
    if len(forecasts) != len(data.test) or set(forecasts.id) != set(data.test.id):
        raise RuntimeError("forecast coverage does not match test ids")
    forecasts.to_csv(output / "forecasts.csv", index=False)
    forecasts[["id", "sales"]].to_csv(output / "submission.csv", index=False)
    print(f"Wrote {len(forecasts)} non-negative predictions to {output / 'submission.csv'}")


if __name__ == "__main__":
    main()
