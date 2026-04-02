from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.utils.statistics import check_seasonality
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


def adf_test(series: pd.Series, alpha: float = 0.05) -> Dict:
    """Run ADF test and return summary dictionary."""
    clean = series.dropna().astype(float)
    stat, pvalue, lags, nobs, crit, _ = adfuller(clean)
    return {
        "adf_stat": float(stat),
        "p_value": float(pvalue),
        "n_lags": int(lags),
        "n_obs": int(nobs),
        "critical_values": crit,
        "is_stationary": bool(pvalue < alpha),
    }


def make_stationary(series: pd.Series, alpha: float = 0.05) -> Dict:
    """Apply progressive differencing: none -> diff(1) -> diff(1)+diff(12)."""
    stage_0 = series.copy()
    adf_0 = adf_test(stage_0, alpha=alpha)
    if adf_0["is_stationary"]:
        return {"series": stage_0, "method": "none", "adf": adf_0}

    stage_1 = stage_0.diff().dropna()
    adf_1 = adf_test(stage_1, alpha=alpha)
    if adf_1["is_stationary"]:
        return {"series": stage_1, "method": "diff_1", "adf": adf_1}

    stage_2 = stage_0.diff().diff(12).dropna()
    adf_2 = adf_test(stage_2, alpha=alpha)
    return {"series": stage_2, "method": "diff_1_diff_12", "adf": adf_2}


def plot_stationary_diagnostics(series: pd.Series, max_acf_lag: int = 36, max_pacf_lag: int = 24) -> None:
    """Plot stationary series, ACF, and PACF for parameter-bound inspection."""
    y = series.dropna().astype(float)
    if len(y) < 3:
        raise ValueError("Need at least 3 data points to plot ACF/PACF diagnostics.")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    axes[0].plot(y.index.astype(str), y.values, marker="o")
    axes[0].set_title("Stationary Series")
    axes[0].tick_params(axis="x", rotation=90)

    max_safe_acf = max(1, min(max_acf_lag, len(y) - 1))
    plot_acf(y.values, lags=max_safe_acf, ax=axes[1])
    axes[1].set_title("ACF")

    max_safe_pacf = max(1, min(max_pacf_lag, len(y) // 2 - 1))
    plot_pacf(y.values, lags=max_safe_pacf, ax=axes[2], method="ywm")
    axes[2].set_title("PACF")

    plt.tight_layout()
    plt.show()


def detect_seasonality_with_darts(series: pd.Series, max_lag: int = 36):
    """Use Darts seasonality checker and return (has_seasonality, period)."""
    y = series.dropna().astype(float).copy()

    # Darts requires a DatetimeIndex/RangeIndex; normalize to monthly DatetimeIndex.
    y.index = pd.to_datetime(y.index).to_period("M").to_timestamp("M")
    y = y.sort_index()

    # Guard against duplicate months after normalization.
    if y.index.has_duplicates:
        y = y.groupby(level=0).mean()

    ts = TimeSeries.from_series(y, fill_missing_dates=True, freq="M")
    has_seasonality, period = check_seasonality(ts, max_lag=max_lag)
    return bool(has_seasonality), int(period) if period else None


def suggest_pq_bounds_from_acf_pacf(stationary_series: pd.Series, max_lag: int = 24) -> Dict:
    """Simple significance-based p/q bound suggestion for grid initialization."""
    y = stationary_series.dropna().astype(float).values
    n = len(y)
    if n < 10:
        return {"p_max": 2, "q_max": 2}

    conf = 1.96 / np.sqrt(n)

    from statsmodels.tsa.stattools import acf, pacf

    acf_vals = acf(y, nlags=max_lag, fft=False)
    pacf_vals = pacf(y, nlags=max_lag, method="ywm")

    q_sig = np.where(np.abs(acf_vals[1:]) > conf)[0] + 1
    p_sig = np.where(np.abs(pacf_vals[1:]) > conf)[0] + 1

    q_max = int(min(max(q_sig) if len(q_sig) else 2, 5))
    p_max = int(min(max(p_sig) if len(p_sig) else 2, 5))
    return {"p_max": max(1, p_max), "q_max": max(1, q_max)}
