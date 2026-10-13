from dataclasses import dataclass

import numpy as np
import pandas as pd

from retail_forecast.data import Dataset

LAGS = (1, 7, 14, 28)
WINDOWS = (7, 28)
CATEGORICAL = ["store_nbr", "family_code", "city_code", "state_code", "type_code", "cluster"]


def target_matrix(history: pd.DataFrame) -> pd.DataFrame:
    """Missing calendar dates remain unknown, never shifted into the wrong day."""
    matrix = history.pivot(index="date", columns=["store_nbr", "family"], values="sales")
    return matrix.reindex(pd.date_range(matrix.index.min(), matrix.index.max()))


def lookup(matrix: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    dates = matrix.index.get_indexer(rows.date)
    pairs = pd.MultiIndex.from_frame(rows[["store_nbr", "family"]])
    columns = matrix.columns.get_indexer(pairs)
    valid = (dates >= 0) & (columns >= 0)
    result = np.full(len(rows), np.nan, dtype=np.float32)
    result[valid] = matrix.to_numpy()[dates[valid], columns[valid]]
    return result


@dataclass
class FeatureBuilder:
    stores: pd.DataFrame
    oil: pd.DataFrame
    holidays: pd.DataFrame
    vocabularies: dict
    origin: pd.Timestamp

    @classmethod
    def from_dataset(cls, data: Dataset) -> "FeatureBuilder":
        vocab = {"family": sorted(data.train.family.unique())}
        for column in ["city", "state", "type"]:
            vocab[column] = sorted(data.stores[column].unique())
        return cls(data.stores, data.oil, data.holidays_events, vocab, data.train.date.min())

    def transform(
        self, rows: pd.DataFrame, history: pd.DataFrame, oil_cutoff: pd.Timestamp | None = None
    ) -> pd.DataFrame:
        rows = rows.reset_index(drop=True)
        merged = rows.merge(self.stores, on="store_nbr", how="left", validate="many_to_one")
        dates = rows.date.dt
        features = pd.DataFrame(index=rows.index)
        for column in ["store_nbr", "cluster"]:
            features[column] = merged[column].astype(np.float32)
        for column, vocabulary in self.vocabularies.items():
            features[f"{column}_code"] = pd.Categorical(
                merged[column], categories=vocabulary
            ).codes.astype(np.float32)
        features["onpromotion"] = np.log1p(rows.onpromotion).astype(np.float32)
        features["dow"] = dates.dayofweek
        features["month"] = dates.month
        features["day"] = dates.day
        features["year"] = dates.year
        features["week"] = dates.isocalendar().week.astype(int)
        features["trend"] = (rows.date - self.origin).dt.days
        features["payday"] = ((dates.day == 15) | dates.is_month_end).astype(int)
        # Only previous/current published oil observations; never backward fill.
        oil = self.oil.set_index("date").dcoilwtico.sort_index()
        if oil_cutoff is not None:
            oil = oil.loc[oil.index <= oil_cutoff]
        observed_dates = rows.date - pd.Timedelta(days=1)
        oil = (
            oil.reindex(oil.index.union(pd.DatetimeIndex(observed_dates.unique())))
            .sort_index()
            .ffill()
        )
        features["oil"] = observed_dates.map(oil)
        active = self.holidays.loc[~self.holidays.transferred]
        for label, types in {
            "holiday": ["Holiday", "Transfer", "Additional", "Bridge"],
            "event": ["Event"],
            "workday": ["Work Day"],
        }.items():
            selected = active[active.type.isin(types)]
            flag = np.zeros(len(rows), dtype=bool)
            for locale, place in [("National", None), ("Regional", "state"), ("Local", "city")]:
                part = selected[selected.locale.eq(locale)]
                if place is None:
                    flag |= rows.date.isin(part.date).to_numpy()
                else:
                    keys = pd.MultiIndex.from_frame(part[["date", "locale_name"]])
                    row_keys = pd.MultiIndex.from_arrays([rows.date, merged[place]])
                    flag |= row_keys.isin(keys)
            features[label] = flag.astype(int)
        matrix = target_matrix(history)
        # Extend only the index: the next prediction day has no observed target.
        matrix = matrix.reindex(
            pd.date_range(matrix.index.min(), max(matrix.index.max(), rows.date.max()))
        )
        for lag in LAGS:
            features[f"lag_{lag}"] = np.log1p(lookup(matrix.shift(lag), rows))
        previous = matrix.shift(1)
        for window in WINDOWS:
            features[f"mean_{window}"] = np.log1p(
                lookup(previous.rolling(window, min_periods=1).mean(), rows)
            )
            features[f"std_{window}"] = np.log1p(
                lookup(previous.rolling(window, min_periods=2).std(), rows)
            )
        return features.astype(np.float32)
