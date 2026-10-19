"""Phase 6: Leave-One-Base-Dataset-Out (LODO) Cross-Validation (`src/tsdomain/analysis/grouped_cv.py`)."""

from __future__ import annotations

import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.model_selection import LeaveOneGroupOut

from tsdomain.analysis.mixed_models import (
    MODEL_FORMULAS,
    build_phase6_analysis_dataset,
)
from tsdomain.utils import PROJECT_ROOT, ensure_dir


def _fit_and_predict_lodo_fold(
    model_name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    """Fit a candidate predictor on `train_df` and predict `delta_mase` on held-out `test_df`."""
    y_train = train_df["delta_mase"].to_numpy(dtype=float)

    if model_name == "Constant":
        return np.full(len(test_df), float(np.mean(y_train)), dtype=float)

    if model_name == "TanJerome_Simon_Heuristic":
        # Tan Jerome & Simon (2026) two-feature heuristic: length < 700 and strong seasonality
        seas_med = float(train_df["seasonal_strength"].median())
        tr_short = (train_df["length"] < 700).astype(int)
        tr_seas = (train_df["seasonal_strength"] >= seas_med).astype(int)
        cell_means: Dict[tuple, float] = {}
        grand_mean = float(np.mean(y_train))
        for s_flag in (0, 1):
            for z_flag in (0, 1):
                mask = (tr_short == s_flag) & (tr_seas == z_flag)
                cell_means[(s_flag, z_flag)] = (
                    float(train_df.loc[mask, "delta_mase"].mean()) if mask.any() else grand_mean
                )
        te_short = (test_df["length"] < 700).astype(int).to_numpy()
        te_seas = (test_df["seasonal_strength"] >= seas_med).astype(int).to_numpy()
        return np.array(
            [cell_means[(int(s), int(z))] for s, z in zip(te_short, te_seas)],
            dtype=float,
        )

    # For M1, M2a, M2b, M2, M3: fit fixed effects + model_family random/shrunk intercept on train_df
    # Note: held-out test_df has an unseen base_dataset, so base_dataset random effect is 0 out-of-sample,
    # while model_family random effect transfers across datasets!
    formula = MODEL_FORMULAS[model_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Fit MixedLM grouped by model_family so model_family random intercepts are used for OOS prediction
            md = smf.mixedlm(formula, data=train_df, groups=train_df["model_family"])
            res = md.fit(reml=True, method="lbfgs", maxiter=300)
            fe_pred = np.asarray(res.predict(exog=test_df), dtype=float)
            re_map = {
                fam: float(np.asarray(val).ravel()[0])
                for fam, val in res.random_effects.items()
            }
            re_pred = np.array(
                [re_map.get(fam, 0.0) for fam in test_df["model_family"]],
                dtype=float,
            )
            return fe_pred + re_pred
        except Exception:
            ols_res = smf.ols(formula + " + C(model_family)", data=train_df).fit()
            return np.asarray(ols_res.predict(test_df), dtype=float)


def run_phase6_grouped_cv(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run Leave-One-Base-Dataset-Out (LODO) cross-validation across all candidate predictors."""
    out_dir = ensure_dir(PROJECT_ROOT / "results" / "analysis")
    if df is None:
        pq_path = out_dir / "phase6_analysis_dataset.parquet"
        df = pd.read_parquet(pq_path) if pq_path.exists() else build_phase6_analysis_dataset()

    logo = LeaveOneGroupOut()
    groups = df["base_dataset"].to_numpy()

    candidates = [
        "Constant",
        "TanJerome_Simon_Heuristic",
        "M1",
        "M2a",
        "M2b",
        "M2",
        "M3",
    ]

    pred_records: List[pd.DataFrame] = []

    for train_idx, test_idx in logo.split(df, groups=groups):
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        held_out_ds = str(test_df["base_dataset"].iloc[0])

        fold_df = test_df[
            [
                "model",
                "model_family",
                "config",
                "base_dataset",
                "domain",
                "sim_stratum",
                "sim_binary",
                "sim_distance",
                "delta_mase",
                "delta_mase_raw",
                "tsfm_wins",
            ]
        ].copy()
        fold_df["held_out_base_dataset"] = held_out_ds

        for cand in candidates:
            fold_df[f"pred_{cand}"] = _fit_and_predict_lodo_fold(cand, train_df, test_df)

        pred_records.append(fold_df)

    all_preds_df = pd.concat(pred_records, ignore_index=True)
    all_preds_df.to_parquet(out_dir / "lodo_cv_predictions.parquet", index=False)

    y_true = all_preds_df["delta_mase"].to_numpy(dtype=float)
    true_pos = (y_true > 0.0).astype(int)

    summary_rows: List[Dict] = []
    for cand in candidates:
        y_hat = all_preds_df[f"pred_{cand}"].to_numpy(dtype=float)
        err = y_hat - y_true
        rmse_val = float(np.sqrt(np.mean(err**2)))
        mae_val = float(np.mean(np.abs(err)))

        if np.std(y_hat) < 1e-12:
            rho, rho_p = 0.0, 1.0
        else:
            sp = stats.spearmanr(y_true, y_hat)
            rho, rho_p = float(sp.statistic), float(sp.pvalue)

        pred_pos = (y_hat > 0.0).astype(int)
        sign_acc = float(np.mean(pred_pos == true_pos))

        # Per-base-dataset fold RMSE for paired Wilcoxon / t-test vs M1
        fold_rmses = (
            all_preds_df.assign(_sq_err=(all_preds_df[f"pred_{cand}"] - all_preds_df["delta_mase"]) ** 2)
            .groupby("base_dataset")["_sq_err"]
            .apply(lambda s: float(np.sqrt(np.mean(s))))
        )

        summary_rows.append(
            {
                "predictor": cand,
                "n_pairs": int(len(y_true)),
                "n_base_datasets": int(all_preds_df["base_dataset"].nunique()),
                "oos_rmse": rmse_val,
                "oos_mae": mae_val,
                "fold_median_rmse": float(fold_rmses.median()),
                "spearman_rho": rho,
                "spearman_p_value": rho_p,
                "sign_accuracy": sign_acc,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    m1_rmse = float(summary_df.loc[summary_df["predictor"] == "M1", "oos_rmse"].iloc[0])
    summary_df["rmse_change_vs_M1_pct"] = ((summary_df["oos_rmse"] - m1_rmse) / m1_rmse) * 100.0
    summary_df.to_csv(out_dir / "lodo_cv_summary.csv", index=False)
    return summary_df
