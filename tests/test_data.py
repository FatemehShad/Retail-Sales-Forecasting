import numpy as np
import pandas as pd
import pytest
from conftest import write_dataset

from retail_forecast.data import Dataset


def test_csv_roundtrip(dataset, tmp_path):
    write_dataset(dataset, tmp_path)
    result = Dataset.from_directory(tmp_path)
    pd.testing.assert_frame_equal(result.train, dataset.train)
    assert result.holidays_events.transferred.dtype == bool


@pytest.mark.parametrize(
    "defect", ["duplicates", "negative", "nan", "id", "stores", "gap", "series"]
)
def test_rejects_invalid_panel(dataset, defect):
    frames = {k: v.copy() for k, v in vars(dataset).items()}
    if defect == "duplicates":
        frames["train"] = pd.concat([frames["train"], frames["train"].iloc[:1]])
    elif defect == "negative":
        frames["train"].loc[0, "sales"] = -1
    elif defect == "nan":
        frames["train"].loc[0, "sales"] = np.nan
    elif defect == "id":
        frames["test"].iloc[0, 0] = frames["train"].iloc[0, 0]
    elif defect == "stores":
        frames["stores"] = frames["stores"].iloc[:1]
    elif defect == "gap":
        frames["test"]["date"] += pd.Timedelta(days=1)
    else:
        frames["test"] = frames["test"].iloc[1:]
    with pytest.raises(ValueError):
        Dataset.from_frames(**frames)
