from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["date", "store_nbr", "family"]
REQUIRED = {
    "train": ["id", *KEYS, "sales", "onpromotion"],
    "test": ["id", *KEYS, "onpromotion"],
    "stores": ["store_nbr", "city", "state", "type", "cluster"],
    "oil": ["date", "dcoilwtico"],
    "holidays_events": ["date", "type", "locale", "locale_name", "description", "transferred"],
    "transactions": ["date", "store_nbr", "transactions"],
}


@dataclass
class Dataset:
    train: pd.DataFrame
    test: pd.DataFrame
    stores: pd.DataFrame
    oil: pd.DataFrame
    holidays_events: pd.DataFrame
    transactions: pd.DataFrame

    @classmethod
    def from_directory(cls, directory: str | Path) -> "Dataset":
        path = Path(directory)
        frames = {name: pd.read_csv(path / f"{name}.csv") for name in REQUIRED}
        return cls.from_frames(**frames)

    @classmethod
    def from_frames(cls, **frames: pd.DataFrame) -> "Dataset":
        for name, columns in REQUIRED.items():
            missing = set(columns) - set(frames[name].columns)
            if missing:
                raise ValueError(f"{name}: missing columns {sorted(missing)}")
            frames[name] = frames[name].copy()
            if "date" in frames[name]:
                frames[name]["date"] = pd.to_datetime(frames[name]["date"], errors="raise")
                dates = frames[name]["date"]
                if dates.isna().any() or (dates != dates.dt.normalize()).any():
                    raise ValueError(f"{name}: dates must be non-null calendar dates")
        result = cls(**frames)
        result.validate()
        return result

    def validate(self) -> None:
        for name, frame, key in [
            ("train", self.train, KEYS),
            ("test", self.test, KEYS),
            ("stores", self.stores, ["store_nbr"]),
            ("oil", self.oil, ["date"]),
            ("transactions", self.transactions, ["date", "store_nbr"]),
        ]:
            if frame.empty and name != "transactions":
                raise ValueError(f"{name}: empty data")
            if frame[key].isna().any().any() or frame.duplicated(key).any():
                raise ValueError(f"{name}: null or duplicate keys")
        for name, frame in [("train", self.train), ("test", self.test)]:
            if frame.id.isna().any() or frame.id.duplicated().any():
                raise ValueError(f"{name}: null or duplicate id")
            for column in ["onpromotion", *(["sales"] if name == "train" else [])]:
                values = pd.to_numeric(frame[column], errors="raise")
                if not np.isfinite(values).all() or (values < 0).any():
                    raise ValueError(f"{name}: {column} must be finite and non-negative")
            if not set(frame.store_nbr).issubset(set(self.stores.store_nbr)):
                raise ValueError(f"{name}: missing store metadata")
        if self.stores[["city", "state", "type", "cluster"]].isna().any().any():
            raise ValueError("stores: missing metadata")
        oil = pd.to_numeric(self.oil.dcoilwtico.dropna(), errors="raise")
        if not np.isfinite(oil).all():
            raise ValueError("oil: observations must be finite or missing")
        if (
            self.holidays_events[["type", "locale", "locale_name", "transferred"]]
            .isna()
            .any()
            .any()
        ):
            raise ValueError("holidays_events: missing metadata")
        transactions = pd.to_numeric(self.transactions.transactions, errors="raise")
        if not np.isfinite(transactions).all() or (transactions < 0).any():
            raise ValueError("transactions: values must be finite and non-negative")
        if set(self.train.id) & set(self.test.id):
            raise ValueError("train/test ids overlap")
        if self.test.date.min() != self.train.date.max() + pd.Timedelta(days=1):
            raise ValueError("test must begin the day after training ends")
        train_pairs = set(zip(self.train.store_nbr, self.train.family, strict=True))
        for date, day in self.test.groupby("date"):
            if set(zip(day.store_nbr, day.family, strict=True)) != train_pairs:
                raise ValueError(f"test: incomplete or unseen series on {date.date()}")
        expected = pd.date_range(self.test.date.min(), self.test.date.max())
        if not expected.equals(pd.DatetimeIndex(sorted(self.test.date.unique()))):
            raise ValueError("test: horizon must contain consecutive calendar days")
        transferred = self.holidays_events.transferred.astype(str).str.lower()
        if not transferred.isin(["true", "false"]).all():
            raise ValueError("holidays_events: transferred must be True or False")
        if not self.holidays_events.locale.isin(["National", "Regional", "Local"]).all():
            raise ValueError("holidays_events: unknown locale")
        self.holidays_events["transferred"] = transferred.eq("true")
