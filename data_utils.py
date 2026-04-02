from pathlib import Path
from typing import Dict

import pandas as pd


def load_metric_dataframe(config: Dict) -> pd.DataFrame:
    """Load the raw metric data file and standardize DATE dtype."""
    raw_path = Path(config["paths"]["raw_data"])
    if raw_path.suffix.lower() == ".csv":
        df = pd.read_csv(raw_path)
    elif raw_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(raw_path)
    else:
        raise ValueError(f"Unsupported file type: {raw_path.suffix}")

    date_col = config["columns"]["date"]
    # Keep monthly DatetimeIndex-compatible timestamps for downstream SARIMAX.
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp("M")
    return df


def build_series(config: Dict, df: pd.DataFrame, metric_name: str) -> pd.Series:
    """Filter by portfolio/sub-portfolio/forecast_type/metric and return date-indexed series."""
    c = config["columns"]
    mask = (
        (df[c["portfolio"]] == config["portfolio"])
        & (df[c["sub_portfolio"]] == config["sub_portfolio"])
        & (df[c["forecast_type"]] == config["forecast_type"])
        & (df[c["metric"]] == metric_name)
    )
    series = (
        df.loc[mask, [c["date"], c["value"]]]
        .groupby(c["date"], as_index=True)[c["value"]]
        .sum()
        .sort_index()
        .astype(float)
    )
    series.index = pd.DatetimeIndex(series.index)
    inferred = pd.infer_freq(series.index)
    if inferred is not None:
        series = series.asfreq(inferred)
    series.name = metric_name
    return series


def split_by_dates(series: pd.Series, start_date: str, train_end: str, test_end: str | None = None):
    """Split a series into train/test windows using ISO date strings."""
    s = series.copy()
    start_ts = pd.Timestamp(start_date)
    train_end_ts = pd.Timestamp(train_end)
    s = s[s.index >= start_ts]
    train = s[s.index <= train_end_ts]
    if test_end is None:
        test = s[s.index > train_end_ts]
    else:
        test_end_ts = pd.Timestamp(test_end)
        test = s[
            (s.index > train_end_ts)
            & (s.index <= test_end_ts)
        ]
    return train, test
