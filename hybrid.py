from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import LinearRegressionModel
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from .modeling import fit_sarimax, forecast_sarimax


def fit_base_sarimax(
    y_train: pd.Series,
    exog_train: pd.DataFrame | None,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
):
    """Train base structural SARIMAX model."""
    return fit_sarimax(y_train, exog_train, order, seasonal_order)


def historical_one_step_predictions(
    y_train: pd.Series,
    exog_train: pd.DataFrame | None,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    start_ratio: float = 0.4,
    min_train_points: int = 24,
) -> pd.Series:
    """Generate one-step-ahead historical predictions without look-ahead leakage."""
    if len(y_train) < 3:
        return pd.Series(dtype=float)

    start_idx = max(int(len(y_train) * start_ratio), min_train_points)
    start_idx = min(start_idx, len(y_train) - 1)

    init_y = y_train.iloc[:start_idx]
    init_x = None if exog_train is None else exog_train.loc[init_y.index]

    state = fit_sarimax(init_y, init_x, order, seasonal_order)

    preds = []
    pred_idx = []
    for dt in y_train.index[start_idx:]:
        x_next = None if exog_train is None else exog_train.loc[[dt]]
        y_hat = float(state.get_forecast(steps=1, exog=x_next).predicted_mean.iloc[0])
        preds.append(y_hat)
        pred_idx.append(dt)

        y_new = pd.Series([float(y_train.loc[dt])], index=[dt], name=y_train.name)
        state = state.append(endog=y_new, exog=x_next, refit=False)

    return pd.Series(preds, index=pred_idx, name="base_hist_pred")


def compute_base_residuals_with_historical_forecasts(
    y_train: pd.Series,
    exog_train: pd.DataFrame | None,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    start_ratio: float = 0.4,
    min_train_points: int = 24,
) -> pd.Series:
    """Compute residual series as actual minus one-step historical predictions."""
    hist_pred = historical_one_step_predictions(
        y_train=y_train,
        exog_train=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        start_ratio=start_ratio,
        min_train_points=min_train_points,
    )
    if hist_pred.empty:
        return pd.Series(dtype=float, name="base_residual")

    residual = y_train.loc[hist_pred.index] - hist_pred
    residual.name = "base_residual"
    return residual


def fit_residual_lr_model(residual_series: pd.Series):
    """Fit explainable Darts Linear Regression on residual momentum only."""
    ts_residual = TimeSeries.from_series(residual_series.astype(float))
    lr = LinearRegressionModel(lags=3, output_chunk_length=1)
    lr.fit(ts_residual)
    return lr


def forecast_hybrid(
    base_fit,
    residual_lr,
    residual_series: pd.Series,
    exog_test: pd.DataFrame | None,
    test_index,
):
    """Combine structural base forecast and residual correction forecast."""
    base_fc = forecast_sarimax(base_fit, len(test_index), exog_test, test_index)

    ts_res = TimeSeries.from_series(residual_series.astype(float))
    res_fc = residual_lr.predict(n=len(test_index), series=ts_res).pd_series()
    res_fc = pd.Series(res_fc.values.astype(float), index=list(test_index), name="residual_correction")

    hybrid_fc = pd.Series(base_fc.values + res_fc.values, index=list(test_index), name="hybrid_forecast")
    return base_fc, res_fc, hybrid_fc


def error_diagnostics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Metrics and simple residual diagnostics for forecast error analysis."""
    e = y_true.values.astype(float) - y_pred.values.astype(float)
    mape = mean_absolute_percentage_error(y_true.values.astype(float), y_pred.values.astype(float)) * 100.0
    mae = mean_absolute_error(y_true.values.astype(float), y_pred.values.astype(float))
    rmse = np.sqrt(mean_squared_error(y_true.values.astype(float), y_pred.values.astype(float)))

    bias = float(np.mean(e))
    err_std = float(np.std(e))
    lag1_corr = float(pd.Series(e).autocorr(lag=1)) if len(e) > 2 else np.nan

    return {
        "MAPE": float(mape),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "BIAS": bias,
        "ERR_STD": err_std,
        "ERR_LAG1_AUTOCORR": lag1_corr,
    }
