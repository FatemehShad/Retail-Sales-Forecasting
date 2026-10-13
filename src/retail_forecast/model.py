import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from retail_forecast.data import KEYS, Dataset
from retail_forecast.features import CATEGORICAL, FeatureBuilder, lookup, target_matrix


def rmsle(actual, predicted) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("RMSLE needs equally sized, non-empty inputs")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("RMSLE inputs must be finite")
    if (actual < 0).any() or (predicted < 0).any():
        raise ValueError("RMSLE inputs must be non-negative")
    return float(np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)))


@dataclass
class ModelConfig:
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    n_jobs: int = 4
    random_state: int = 42
    horizon: int = 16
    folds: int = 3
    minimum_history: int = 56

    def __post_init__(self):
        for name in [
            "n_estimators",
            "num_leaves",
            "min_child_samples",
            "n_jobs",
            "horizon",
            "folds",
        ]:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.minimum_history < 28 or not 0 < self.learning_rate <= 1:
            raise ValueError("minimum_history must be >=28 and learning_rate in (0,1]")


@dataclass
class ForecastModel:
    builder: FeatureBuilder
    estimator: object | None
    model_name: str

    def predict(self, history: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
        """One day at a time; later lags use predictions, never holdout actuals."""
        history = history[[*KEYS, "sales"]].copy()
        future = future.copy()
        history["date"] = pd.to_datetime(history.date)
        future["date"] = pd.to_datetime(future.date)
        cutoff = history.date.max()
        if future.empty or future.date.min() != cutoff + pd.Timedelta(days=1):
            raise ValueError("future must start one day after history")
        # Prediction only needs 28 days of target history, even for long training panels.
        history = history[history.date >= cutoff - pd.Timedelta(days=27)]
        known_pairs = set(zip(history.store_nbr, history.family, strict=True))
        outputs = []
        for date in pd.date_range(future.date.min(), future.date.max()):
            day = future[future.date.eq(date)].copy().reset_index(drop=True)
            if day.empty or day.duplicated(KEYS).any():
                raise ValueError("future must have unique series on every calendar day")
            if set(zip(day.store_nbr, day.family, strict=True)) != known_pairs:
                raise ValueError("future contains incomplete or unseen series")
            if self.model_name == "weekly_naive":
                matrix = target_matrix(history)
                shifted = day.copy()
                shifted["date"] -= pd.Timedelta(days=7)
                prediction = lookup(matrix, shifted)
                fallback = matrix.tail(7).mean().to_numpy()
                cols = matrix.columns.get_indexer(
                    pd.MultiIndex.from_frame(day[["store_nbr", "family"]])
                )
                if (cols < 0).any():
                    raise ValueError("future contains unseen series")
                prediction = np.where(np.isnan(prediction), fallback[cols], prediction)
                prediction = np.nan_to_num(prediction, nan=0.0)
            else:
                features = self.builder.transform(day, history, oil_cutoff=cutoff)
                # Avoid overflow from corrupt models while keeping predictions non-negative.
                prediction = np.expm1(np.clip(self.estimator.predict(features), 0, 20))
            if not np.isfinite(prediction).all():
                raise ValueError("non-finite forecast")
            day["sales"] = prediction
            outputs.append(day[["id", *KEYS, "sales"]])
            history = pd.concat([history, day[[*KEYS, "sales"]]], ignore_index=True)
            history = history[history.date >= date - pd.Timedelta(days=27)]
        return pd.concat(outputs, ignore_index=True).sort_values("id").reset_index(drop=True)

    def save(self, directory: str | Path, metadata: dict) -> None:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "model.joblib")
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> "ForecastModel":
        # joblib is executable serialization: load only artifacts from your trusted jobs.
        return joblib.load(Path(directory) / "model.joblib")


def fit(data: Dataset, history: pd.DataFrame, config: ModelConfig) -> ForecastModel:
    builder = FeatureBuilder.from_dataset(data)
    features = builder.transform(history, history)
    estimator = lgb.LGBMRegressor(
        objective="regression",
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )
    estimator.fit(features, np.log1p(history.sales), categorical_feature=CATEGORICAL)
    return ForecastModel(builder, estimator, "lightgbm")


def train_and_evaluate(
    data: Dataset, config: ModelConfig
) -> tuple[ForecastModel, dict, pd.DataFrame]:
    last = data.train.date.max()
    first = data.train.date.min()
    if (last - first).days + 1 < config.minimum_history + config.folds * config.horizon:
        raise ValueError("not enough history for the requested rolling folds")
    results, predictions = [], []
    for fold in range(config.folds):
        end = last - pd.Timedelta(days=(config.folds - 1 - fold) * config.horizon)
        start = end - pd.Timedelta(days=config.horizon - 1)
        history = data.train[data.train.date < start]
        actual = data.train[data.train.date.between(start, end)]
        if actual.date.nunique() != config.horizon:
            raise ValueError("backtest window has missing calendar days; choose another window")
        pairs = history[["store_nbr", "family"]].drop_duplicates()
        if len(actual) != config.horizon * len(pairs):
            raise ValueError("backtest window has incomplete series")
        future = actual.drop(columns="sales")
        candidate = fit(data, history, config)
        baseline = ForecastModel(candidate.builder, None, "weekly_naive")
        for model in [candidate, baseline]:
            forecast = model.predict(history, future)
            aligned = actual[["id", "sales"]].merge(
                forecast[["id", "sales"]],
                on="id",
                validate="one_to_one",
                suffixes=("_actual", "_pred"),
            )
            score = rmsle(aligned.sales_actual, aligned.sales_pred)
            results.append(
                {
                    "fold": fold + 1,
                    "model": model.model_name,
                    "cutoff": start - pd.Timedelta(days=1),
                    "start": start,
                    "end": end,
                    "rmsle": score,
                    "rows": len(aligned),
                }
            )
            forecast["fold"] = fold + 1
            forecast["model"] = model.model_name
            forecast = forecast.merge(
                actual[["id", "sales"]].rename(columns={"sales": "actual"}), on="id"
            )
            predictions.append(forecast)
    metrics = pd.DataFrame(results)
    scores = metrics.groupby("model").rmsle.mean().to_dict()
    # Retain the simpler baseline unless the model improves average out-of-time RMSLE.
    selected = "lightgbm" if scores["lightgbm"] < scores["weekly_naive"] else "weekly_naive"
    model = (
        fit(data, data.train, config)
        if selected == "lightgbm"
        else ForecastModel(FeatureBuilder.from_dataset(data), None, selected)
    )
    metadata = {
        "selected_model": selected,
        "mean_rmsle": scores,
        "folds": metrics.to_dict(orient="records"),
        "config": asdict(config),
        "train_rows": len(data.train),
        "train_start": str(first.date()),
        "train_end": str(last.date()),
        "feature_columns": list(
            model.builder.transform(
                data.train.tail(1), data.train[data.train.date >= last - pd.Timedelta(days=28)]
            ).columns
        ),
        "validation_note": (
            "Rolling folds select the model; a separate final holdout is needed "
            "for an unbiased report."
        ),
    }
    return model, metadata, pd.concat(predictions, ignore_index=True)
