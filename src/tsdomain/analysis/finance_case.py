"""Phase 7 (P1): Finance / Volatility Structural-Break Case Study (`src/tsdomain/analysis/finance_case.py`)."""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ruptures as rpt
from arch import arch_model
from scipy import stats
from scipy.stats import norm

from tsdomain.metrics.point import mase, rmse
from tsdomain.metrics.probabilistic import coverage_and_width, crps_weighted_quantile_loss
from tsdomain.models.chronos_cli import ChronosAdapter
from tsdomain.models.moirai_cli import MoiraiAdapter
from tsdomain.utils import PROJECT_ROOT, QUANTILE_COLS, QUANTILE_LEVELS, ensure_dir

LABELLED_CRISIS_EVENTS: List[Tuple[str, str]] = [
    ("1997-10-27", "Asian_Financial_Crisis_MiniCrash"),
    ("2001-09-17", "Sept11_Market_Reopening"),
    ("2008-09-15", "Lehman_Brothers_GFC"),
    ("2011-08-08", "US_Debt_Downgrade_Eurozone"),
    ("2020-03-09", "COVID19_Crash"),
    ("2022-03-16", "Fed_Rate_Hike_Cycle_Start"),
]


def kupiec_pof_test(
    y_true: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
    alpha: float = 0.20,
) -> Dict[str, float]:
    """Kupiec (1995) Proportion-of-Failures (POF) likelihood-ratio test for interval coverage."""
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(q_low, dtype=float)
    hi = np.asarray(q_high, dtype=float)
    violations = ((y < lo) | (y > hi)).astype(int)
    n = int(len(violations))
    x = int(np.sum(violations))
    if n == 0:
        return {"violation_rate": np.nan, "kupiec_lr_stat": np.nan, "kupiec_p_value": np.nan}

    p_hat = np.clip(x / float(n), 1e-6, 1.0 - 1e-6)
    p0 = float(alpha)
    ll_null = (n - x) * np.log(1.0 - p0) + x * np.log(p0)
    ll_alt = (n - x) * np.log(1.0 - p_hat) + x * np.log(p_hat)
    lr_pof = float(max(0.0, -2.0 * (ll_null - ll_alt)))
    p_val = float(stats.chi2.sf(lr_pof, df=1))
    return {
        "violation_rate": float(x / float(n)),
        "empirical_coverage": float(1.0 - x / float(n)),
        "kupiec_lr_stat": lr_pof,
        "kupiec_p_value": p_val,
    }


def qlike_loss(var_true: np.ndarray, var_pred: np.ndarray) -> float:
    """Patton (2011) QLIKE volatility loss on variance scale: sigma^2/hat_sigma^2 - ln(sigma^2/hat_sigma^2) - 1."""
    vt = np.maximum(np.asarray(var_true, dtype=float), 1e-10)
    vp = np.maximum(np.asarray(var_pred, dtype=float), 1e-10)
    ratio = vt / vp
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def detect_spx_volatility_breaks(df_spx: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
    """Run PELT structural break detection on S&P 500 log Garman-Klass volatility and match 6 crisis dates."""
    log_vol = (0.5 * df_spx["log_gk_var"].to_numpy(dtype=float))
    # Smooth with a 5-day rolling mean for change-point localization against daily noise spikes
    signal = pd.Series(log_vol).rolling(5, min_periods=1).mean().to_numpy(dtype=float)
    z_sig = (signal - np.mean(signal)) / max(np.std(signal, ddof=1), 1e-8)

    algo = rpt.Pelt(model="l2", min_size=15, jump=1).fit(z_sig)
    # Calibrate penalty so PELT segments major volatility regime shifts (~25-35 change points over 36 years)
    bkps = [b for b in algo.predict(pen=12.0) if b < len(df_spx)]

    dates = pd.to_datetime(df_spx["Date"]).reset_index(drop=True)
    rows = []
    for date_str, label in LABELLED_CRISIS_EVENTS:
        target_dt = pd.Timestamp(date_str)
        # Trading index nearest to target_dt
        diffs_all = np.abs((dates - target_dt).dt.days.to_numpy())
        target_idx = int(np.argmin(diffs_all))
        actual_trading_dt = dates.iloc[target_idx]

        if bkps:
            td_dists = [abs(int(b) - target_idx) for b in bkps]
            best_pos = int(np.argmin(td_dists))
            nearest_bkp_idx = int(bkps[best_pos])
            nearest_bkp_dt = dates.iloc[nearest_bkp_idx]
            trading_day_dist = int(td_dists[best_pos])
        else:
            nearest_bkp_idx = target_idx
            nearest_bkp_dt = actual_trading_dt
            trading_day_dist = 999

        rows.append(
            {
                "event_label": label,
                "labelled_date": date_str,
                "labelled_trading_idx": target_idx,
                "detected_break_date": nearest_bkp_dt.strftime("%Y-%m-%d"),
                "detected_break_idx": nearest_bkp_idx,
                "trading_day_distance": trading_day_dist,
                "recovered_within_30d": bool(trading_day_dist <= 30),
            }
        )

    breaks_df = pd.DataFrame(rows)
    return breaks_df, bkps


def _forecast_ewma_log_vol(
    log_vol_train: np.ndarray,
    horizon: int,
    lam: float = 0.94,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """RiskMetrics EWMA(lambda=0.94) forecast mapped to log-volatility scale with analytical quantiles."""
    var_series = np.exp(2.0 * np.asarray(log_vol_train, dtype=float))
    ewma_var = float(var_series[0])
    ewma_path = np.empty_like(var_series)
    for t, v in enumerate(var_series):
        ewma_var = lam * ewma_var + (1.0 - lam) * float(v)
        ewma_path[t] = ewma_var

    # Mean reversion toward long-run log-vol at rate 0.985^h on log scale
    long_run_lv = float(np.mean(log_vol_train[-252:])) if len(log_vol_train) >= 252 else float(np.mean(log_vol_train))
    cur_lv = 0.5 * np.log(max(ewma_var, 1e-10))
    steps = np.arange(1, horizon + 1, dtype=float)
    phi = 0.97 ** steps
    mean_pred = phi * cur_lv + (1.0 - phi) * long_run_lv

    resid = log_vol_train[1:] - (0.5 * np.log(np.maximum(ewma_path[:-1], 1e-10)))
    resid_std = float(np.std(resid[-252:], ddof=1)) if len(resid) > 10 else 0.35
    sigma_h = resid_std * np.sqrt(1.0 + 0.15 * (steps - 1.0))
    q_dict = {
        col: mean_pred + float(norm.ppf(q)) * sigma_h
        for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS)
    }
    return mean_pred, q_dict


def _forecast_garch_log_vol(
    log_returns_train: np.ndarray,
    log_vol_train: np.ndarray,
    horizon: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Fit GARCH(1,1) with Student-t innovations via `arch` and map variance forecasts to log-GK-vol scale."""
    ret_pct = np.asarray(log_returns_train[-512:], dtype=float) * 100.0
    lv_tr = np.asarray(log_vol_train[-512:], dtype=float)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(ret_pct, mean="Zero", vol="Garch", p=1, q=1, dist="t", rescale=False)
            res = am.fit(disp="off", show_warning=False)
            fc = res.forecast(horizon=int(horizon), reindex=False)
            cond_var_pct2 = fc.variance.values[-1, :horizon]
            # Convert percentage return variance to daily decimal log-vol and align level with GK log-vol
            garch_in_sample_lv = 0.5 * np.log(np.maximum(res.conditional_volatility**2 / 1e4, 1e-10))
            level_offset = float(np.median(lv_tr - garch_in_sample_lv))
            mean_pred = 0.5 * np.log(np.maximum(cond_var_pct2 / 1e4, 1e-10)) + level_offset
            resid_std = float(np.std(lv_tr - (garch_in_sample_lv + level_offset), ddof=1))
    except Exception:
        return _forecast_ewma_log_vol(log_vol_train, horizon=horizon, lam=0.94)

    steps = np.arange(1, horizon + 1, dtype=float)
    sigma_h = max(resid_std, 0.20) * np.sqrt(1.0 + 0.12 * (steps - 1.0))
    q_dict = {
        col: mean_pred + float(norm.ppf(q)) * sigma_h
        for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS)
    }
    return mean_pred, q_dict


def stationary_block_bootstrap_diff(
    diff_series: np.ndarray,
    block_length: int = 20,
    n_boot: int = 1000,
    seed: int = 20260410,
) -> Tuple[float, float, float]:
    """Politis & Romano (1994) stationary block bootstrap for mean difference and 95% CI."""
    arr = np.asarray(diff_series, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        val = float(arr[0])
        return val, val, val

    rng = np.random.default_rng(seed)
    p_geom = 1.0 / max(1.0, float(block_length))
    boot_means = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sample = []
        while len(sample) < n:
            start = int(rng.integers(0, n))
            blen = int(rng.geometric(p_geom))
            for k in range(blen):
                if len(sample) >= n:
                    break
                sample.append(arr[(start + k) % n])
        boot_means[b] = float(np.mean(sample))

    return (
        float(np.mean(arr)),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def run_phase7_finance(device: str = "auto") -> Dict[str, pd.DataFrame]:
    """Execute Phase 7 Finance / Volatility Structural-Break Case Study (H3)."""
    spx_path = PROJECT_ROOT / "data" / "processed" / "spx_daily.parquet"
    if not spx_path.exists():
        from tsdomain.data.finance import fetch_spx_daily_ohlc

        df_spx = fetch_spx_daily_ohlc()
    else:
        df_spx = pd.read_parquet(spx_path)

    analysis_dir = ensure_dir(PROJECT_ROOT / "results" / "analysis")
    breaks_df, bkps = detect_spx_volatility_breaks(df_spx)
    breaks_df.to_csv(analysis_dir / "finance_breaks.csv", index=False)

    log_vol = 0.5 * df_spx["log_gk_var"].to_numpy(dtype=float)
    log_ret = df_spx["log_return"].to_numpy(dtype=float)
    dates = pd.to_datetime(df_spx["Date"]).to_numpy()
    n_total = len(df_spx)

    # Mark post-break indices: within [0, 63] trading days after each labelled or major PELT break
    major_break_indices = sorted(
        set(breaks_df["labelled_trading_idx"].tolist() + breaks_df["detected_break_idx"].tolist())
    )
    is_post_break = np.zeros(n_total, dtype=bool)
    for b_idx in major_break_indices:
        is_post_break[b_idx : min(n_total, b_idx + 63)] = True

    # Build 24 post-break origin indices (4 offsets per crisis: +2, +10, +22, +38 trading days)
    post_break_origins: List[int] = []
    for _, r in breaks_df.iterrows():
        b0 = int(r["labelled_trading_idx"])
        for offset in (2, 10, 22, 38):
            cand = b0 + offset
            if 520 <= cand < n_total - 25:
                post_break_origins.append(cand)

    # Build 24 stable origin indices (> 90 trading days away from any break)
    stable_candidates = [
        idx
        for idx in range(600, n_total - 30, 60)
        if not np.any(is_post_break[max(0, idx - 30) : min(n_total, idx + 25)])
    ]
    rng = np.random.default_rng(20260410)
    if len(stable_candidates) > len(post_break_origins):
        chosen_pos = np.linspace(0, len(stable_candidates) - 1, len(post_break_origins), dtype=int)
        stable_origins = [stable_candidates[i] for i in chosen_pos]
    else:
        stable_origins = stable_candidates

    chronos = ChronosAdapter(device=device, num_samples=10, batch_size=64)
    moirai = MoiraiAdapter(device=device, num_samples=20, batch_size=64)
    chronos.load_model()
    moirai.load_model()

    horizons = [1, 5, 20]
    all_window_rows: List[Dict] = []

    for h in horizons:
        origin_specs = (
            [(idx, "post_break") for idx in post_break_origins]
            + [(idx, "stable") for idx in stable_origins]
        )

        # Prepare TSFM batch windows for horizon h
        tsfm_windows = []
        for w_id, (orig_idx, regime) in enumerate(origin_specs):
            tsfm_windows.append(
                {
                    "item_id": f"SPX_{regime}_{w_id}",
                    "window_id": w_id,
                    "y_train": log_vol[max(0, orig_idx - 512) : orig_idx],
                    "y_test": log_vol[orig_idx : orig_idx + h],
                    "test_timestamps": [pd.Timestamp(t) for t in dates[orig_idx : orig_idx + h]],
                    "prediction_length": h,
                    "season_length": 1,
                }
            )

        chronos_fc, _ = chronos.forecast_windows(tsfm_windows, scaling_mode="common_zscore")
        moirai_fc, _ = moirai.forecast_windows(tsfm_windows, scaling_mode="common_zscore")

        for w_id, (orig_idx, regime) in enumerate(origin_specs):
            lv_train = log_vol[max(0, orig_idx - 512) : orig_idx]
            lr_train = log_ret[max(0, orig_idx - 512) : orig_idx]
            lv_test = log_vol[orig_idx : orig_idx + h]
            var_test = np.exp(2.0 * lv_test)

            # 1. Naive (random walk on log-volatility)
            naive_mean = np.repeat(lv_train[-1], h)
            naive_std = float(np.std(np.diff(lv_train[-252:]), ddof=1))
            steps = np.arange(1, h + 1, dtype=float)
            naive_q = {
                col: naive_mean + float(norm.ppf(q)) * naive_std * np.sqrt(steps)
                for q, col in zip(QUANTILE_LEVELS, QUANTILE_COLS)
            }

            # 2. EWMA(0.94)
            ewma_mean, ewma_q = _forecast_ewma_log_vol(lv_train, horizon=h, lam=0.94)

            # 3. GARCH(1,1)-t
            garch_mean, garch_q = _forecast_garch_log_vol(lr_train, lv_train, horizon=h)

            # 4. Chronos_small
            c_sub = chronos_fc[chronos_fc["window_id"] == w_id].sort_values("horizon_step")
            chronos_mean = c_sub["mean"].to_numpy(dtype=float)
            chronos_q = {col: c_sub[col].to_numpy(dtype=float) for col in QUANTILE_COLS}

            # 5. Moirai_small
            m_sub = moirai_fc[moirai_fc["window_id"] == w_id].sort_values("horizon_step")
            moirai_mean = m_sub["mean"].to_numpy(dtype=float)
            moirai_q = {col: m_sub[col].to_numpy(dtype=float) for col in QUANTILE_COLS}

            model_preds = [
                ("Naive", naive_mean, naive_q),
                ("EWMA_094", ewma_mean, ewma_q),
                ("GARCH_1_1_t", garch_mean, garch_q),
                ("Chronos_small", chronos_mean, chronos_q),
                ("Moirai_small", moirai_mean, moirai_q),
            ]

            for m_name, mp, qd in model_preds:
                var_pred = np.exp(2.0 * np.clip(mp, -15.0, 5.0))
                q_map = {q: qd[f"q{q:.1f}"] for q in QUANTILE_LEVELS}
                cov_info = coverage_and_width(lv_test, qd["q0.1"], qd["q0.9"], y_train=lv_train)
                all_window_rows.append(
                    {
                        "horizon": h,
                        "window_id": w_id,
                        "origin_idx": orig_idx,
                        "origin_date": str(pd.Timestamp(dates[orig_idx]).date()),
                        "regime": regime,
                        "model": m_name,
                        "mase": mase(lv_test, mp, lv_train, m=1),
                        "rmse": rmse(lv_test, mp),
                        "qlike": qlike_loss(var_test, var_pred),
                        "crps": crps_weighted_quantile_loss(lv_test, q_map),
                        "coverage_80": float(cov_info["coverage"]),
                        "width_80": float(cov_info["scaled_width"]),
                        "violations_80": int(np.sum((lv_test < qd["q0.1"]) | (lv_test > qd["q0.9"]))),
                        "n_points": int(h),
                    }
                )

    win_df = pd.DataFrame(all_window_rows)
    win_df.to_parquet(analysis_dir / "finance_window_metrics.parquet", index=False)

    # Aggregate by (horizon, regime, model) + Kupiec POF test
    regime_rows: List[Dict] = []
    for (h, regime, m_name), grp in win_df.groupby(["horizon", "regime", "model"], sort=True):
        tot_pts = int(grp["n_points"].sum())
        tot_viol = int(grp["violations_80"].sum())
        p_hat = np.clip(tot_viol / float(max(1, tot_pts)), 1e-6, 1.0 - 1e-6)
        p0 = 0.20
        ll_null = (tot_pts - tot_viol) * np.log(1.0 - p0) + tot_viol * np.log(p0)
        ll_alt = (tot_pts - tot_viol) * np.log(1.0 - p_hat) + tot_viol * np.log(p_hat)
        lr_pof = float(max(0.0, -2.0 * (ll_null - ll_alt)))
        kup_p = float(stats.chi2.sf(lr_pof, df=1))

        regime_rows.append(
            {
                "horizon": int(h),
                "regime": str(regime),
                "model": str(m_name),
                "n_windows": int(len(grp)),
                "mase": float(grp["mase"].mean()),
                "mase_median": float(grp["mase"].median()),
                "rmse": float(grp["rmse"].mean()),
                "qlike": float(grp["qlike"].mean()),
                "crps": float(grp["crps"].mean()),
                "coverage_80": float(1.0 - tot_viol / float(max(1, tot_pts))),
                "width_80": float(grp["width_80"].mean()),
                "kupiec_lr_stat": lr_pof,
                "kupiec_p_value": kup_p,
            }
        )

    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(analysis_dir / "finance_metrics_by_regime.csv", index=False)

    # H3 Test: Compare (MASE_Chronos - MASE_GARCH) vs (MASE_Moirai - MASE_GARCH) in stable vs post_break
    h3_rows: List[Dict] = []
    for h in horizons:
        sub_h = win_df[win_df["horizon"] == h]
        piv = sub_h.pivot(index=["window_id", "regime"], columns="model", values="mase").reset_index()
        piv["chronos_minus_garch"] = piv["Chronos_small"] - piv["GARCH_1_1_t"]
        piv["moirai_minus_garch"] = piv["Moirai_small"] - piv["GARCH_1_1_t"]
        piv["chronos_minus_moirai"] = piv["Chronos_small"] - piv["Moirai_small"]

        for reg in ["stable", "post_break"]:
            r_sub = piv[piv["regime"] == reg]
            c_mean, c_lo, c_hi = stationary_block_bootstrap_diff(
                r_sub["chronos_minus_garch"].to_numpy(), block_length=4, n_boot=1000
            )
            m_mean, m_lo, m_hi = stationary_block_bootstrap_diff(
                r_sub["moirai_minus_garch"].to_numpy(), block_length=4, n_boot=1000
            )
            cm_mean, cm_lo, cm_hi = stationary_block_bootstrap_diff(
                r_sub["chronos_minus_moirai"].to_numpy(), block_length=4, n_boot=1000
            )
            h3_rows.append(
                {
                    "horizon": int(h),
                    "regime": reg,
                    "chronos_minus_garch_mean": c_mean,
                    "chronos_minus_garch_ci95_low": c_lo,
                    "chronos_minus_garch_ci95_high": c_hi,
                    "moirai_minus_garch_mean": m_mean,
                    "moirai_minus_garch_ci95_low": m_lo,
                    "moirai_minus_garch_ci95_high": m_hi,
                    "chronos_minus_moirai_mean": cm_mean,
                    "chronos_minus_moirai_ci95_low": cm_lo,
                    "chronos_minus_moirai_ci95_high": cm_hi,
                }
            )

    h3_df = pd.DataFrame(h3_rows)
    h3_df.to_csv(analysis_dir / "finance_h3_contrast.csv", index=False)

    return {
        "breaks": breaks_df,
        "regime_metrics": regime_df,
        "h3_contrast": h3_df,
    }
