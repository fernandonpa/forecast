from __future__ import annotations

import itertools
import warnings
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning


def fit_sarimax(
    train_y: pd.Series,
    train_exog: pd.DataFrame | None,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    maxiter: int = 35,
):
    """Fit a SARIMAX model with stable defaults."""
    exog_obj = None if train_exog is None else train_exog.astype(float)
    model = SARIMAX(
        endog=train_y.astype(float),
        exog=exog_obj,
        order=order,
        seasonal_order=seasonal_order,
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ValueWarning)
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        result = model.fit(disp=False, maxiter=maxiter, method="lbfgs", cov_type="none")
    return result


def forecast_sarimax(
    result_obj,
    steps: int,
    future_exog: pd.DataFrame | None,
    future_index: Iterable,
) -> pd.Series:
    """Forecast SARIMAX and return aligned pd.Series."""
    exog_obj = None if future_exog is None else future_exog.astype(float)
    fc = result_obj.get_forecast(steps=steps, exog=exog_obj).predicted_mean
    fc = pd.Series(np.asarray(fc, dtype=float), index=list(future_index), name="sarimax_forecast")
    return fc


def build_expanding_validation_splits(
    train_start: str,
    valid_start: str,
    valid_end: str,
    step_months: int = 3,
    enable_fixed_window_tuning: bool = False,
    window_months: int = 12,
) -> List[Dict[str, str]]:
    """Build validation splits in either expanding or fixed-window mode.

    Expanding mode (default):
    - train_end grows with each split
    - validation window is [curr_valid_start, valid_end]

    Fixed-window mode:
    - train_end grows with each split
    - validation window has fixed size `window_months`
    """
    tr_start = pd.Timestamp(train_start)
    va_start = pd.Timestamp(valid_start)
    va_end = pd.Timestamp(valid_end)

    if step_months < 1:
        raise ValueError("step_months must be >= 1")
    if window_months < 1:
        raise ValueError("window_months must be >= 1")

    splits: List[Dict[str, str]] = []
    i = 1
    curr = va_start

    while curr <= va_end:
        tr_end = curr - pd.offsets.MonthEnd(1)

        if enable_fixed_window_tuning:
            curr_valid_end = curr + pd.offsets.MonthEnd(window_months - 1)
            if curr_valid_end > va_end:
                break
        else:
            curr_valid_end = va_end

        splits.append(
            {
                "name": f"split_{i}",
                "train_start": tr_start.strftime("%Y-%m-%d"),
                "train_end": tr_end.strftime("%Y-%m-%d"),
                "valid_start": curr.strftime("%Y-%m-%d"),
                "valid_end": curr_valid_end.strftime("%Y-%m-%d"),
            }
        )
        curr = curr + pd.offsets.MonthEnd(step_months)
        i += 1

    return splits


def _generate_grid(grid_cfg: Dict) -> List[Tuple[int, int, int, int, int, int, int]]:
    """Build deterministic parameter grid from config entries."""
    p_list = list(grid_cfg["p"])
    d_list = list(grid_cfg["d"])
    q_list = list(grid_cfg["q"])
    P_list = list(grid_cfg["P"])
    D_list = list(grid_cfg["D"])
    Q_list = list(grid_cfg["Q"])
    S = int(grid_cfg["S"])
    combos = list(itertools.product(p_list, d_list, q_list, P_list, D_list, Q_list))
    return [(p, d, q, P, D, Q, S) for p, d, q, P, D, Q in combos]


def run_multi_objective_grid_search(
    train_y_full: pd.Series,
    exog_full: pd.DataFrame | None,
    splits: List[Dict],
    grid_cfg: Dict,
    score_cfg: Dict,
):
    """Run grid search and rank by configured objective.

    Supported objectives:
    - mape_only: rank only by AVG_MAPE (AIC/BIC kept for diagnostics).
    - equal_weight: rank by mean(minmax(AIC), minmax(AVG_MAPE)).
    """
    rows = []

    for p, d, q, P, D, Q, S in _generate_grid(grid_cfg):
        order = (p, d, q)
        seasonal_order = (P, D, Q, S)

        if not _is_viable_sarimax_setup(len(train_y_full), order, seasonal_order):
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ValueWarning)
                warnings.simplefilter("ignore", FutureWarning)
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", RuntimeWarning)

                base_x = None if exog_full is None else exog_full.loc[train_y_full.index]
                base_fit = fit_sarimax(train_y_full, base_x, order, seasonal_order)
                aic_value = float(base_fit.aic)
                bic_value = float(base_fit.bic)

            split_mapes = []
            for sp in splits:
                tr_start = pd.Timestamp(sp["train_start"])
                tr_end = pd.Timestamp(sp["train_end"])
                va_start = pd.Timestamp(sp["valid_start"])
                va_end = pd.Timestamp(sp["valid_end"])

                tr_y = train_y_full[(train_y_full.index >= tr_start) & (train_y_full.index <= tr_end)]
                va_y = train_y_full[(train_y_full.index >= va_start) & (train_y_full.index <= va_end)]

                tr_x = None if exog_full is None else exog_full.loc[tr_y.index]
                va_x = None if exog_full is None else exog_full.loc[va_y.index]

                if len(tr_y) < 18 or len(va_y) == 0:
                    continue
                if not _is_viable_sarimax_setup(len(tr_y), order, seasonal_order):
                    continue

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ValueWarning)
                    warnings.simplefilter("ignore", FutureWarning)
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    warnings.simplefilter("ignore", UserWarning)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    fit = fit_sarimax(tr_y, tr_x, order, seasonal_order)
                    pred = forecast_sarimax(fit, len(va_y), va_x, va_y.index)
                mape = mean_absolute_percentage_error(va_y.values.astype(float), pred.values.astype(float)) * 100.0
                split_mapes.append(float(mape))

            if not split_mapes:
                continue

            rows.append(
                {
                    "p": p,
                    "d": d,
                    "q": q,
                    "P": P,
                    "D": D,
                    "Q": Q,
                    "S": S,
                    "AIC": aic_value,
                    "BIC": bic_value,
                    "AVG_MAPE": float(np.mean(split_mapes)),
                    "n_splits": int(len(split_mapes)),
                }
            )
        except Exception:
            continue

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        return leaderboard

    leaderboard = leaderboard.copy()
    objective = str(score_cfg.get("objective", "mape_only")).lower()

    if objective == "equal_weight":
        leaderboard["scaled_AIC"] = _min_max_scale(leaderboard["AIC"])
        leaderboard["scaled_AVG_MAPE"] = _min_max_scale(leaderboard["AVG_MAPE"])
        leaderboard["FINAL_SCORE"] = (leaderboard["scaled_AIC"] + leaderboard["scaled_AVG_MAPE"]) / 2.0
        leaderboard = leaderboard.sort_values(["FINAL_SCORE", "AVG_MAPE"]).reset_index(drop=True)
    else:
        # MAPE-only tuning: AIC/BIC are retained for visibility but not used in ranking.
        leaderboard["FINAL_SCORE"] = leaderboard["AVG_MAPE"]
        leaderboard = leaderboard.sort_values(["AVG_MAPE", "AIC"]).reset_index(drop=True)

    return leaderboard


def _min_max_scale(s: pd.Series) -> pd.Series:
    """Scale a numeric series to [0, 1]."""
    s = s.astype(float)
    min_v = float(s.min())
    max_v = float(s.max())
    if max_v - min_v < 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_v) / (max_v - min_v)


def _is_viable_sarimax_setup(
    n_obs: int,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> bool:
    """Quick guardrail for unstable seasonal fits on short windows."""
    p, d, q = order
    P, D, Q, S = seasonal_order

    if n_obs < 12:
        return False
    if S < 0:
        return False

    # If seasonal terms exist, require enough data to estimate seasonal effects.
    seasonal_terms = P + Q
    dynamic_terms = p + q + seasonal_terms

    # Keep model complexity proportional to sample size to avoid pathological optimizations.
    if n_obs < 48 and dynamic_terms > 3:
        return False
    if n_obs < 60 and dynamic_terms > 4:
        return False

    if S > 1 and seasonal_terms > 0:
        if n_obs <= S * 2:
            return False
        if D > 0 and n_obs <= S * 3:
            return False

    if d + D > 2:
        return False

    required = 8 + d + (D * max(S, 1)) + p + q + (seasonal_terms * max(S, 1))
    return n_obs > required
