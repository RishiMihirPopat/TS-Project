"""Unified Command-Line Interface for the TSFM Domain-Shift & Compute-Efficiency Benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

from tsdomain.config import load_and_validate_configs


def run_phase1_data() -> None:
    """Execute Phase 1: Config validation, GIFT-Eval leaderboard, GIFT-Eval subset, unseen datasets, and SPX."""
    print("[Phase 1] Validating configuration schemas...")
    cfgs = load_and_validate_configs()
    exp_cfg = cfgs["experiment"]

    from tsdomain.data.finance import fetch_spx_daily_ohlc
    from tsdomain.data.gift_eval import (
        build_gift_eval_catalog,
        download_and_load_gift_eval_series,
    )
    from tsdomain.data.leaderboard import ingest_gift_eval_leaderboard
    from tsdomain.data.unseen_mandi import build_unseen_mandi_dataset
    from tsdomain.data.unseen_recency import build_unseen_recency_dataset

    print(f"[Phase 1b] Ingesting GIFT-Eval leaderboard ({len(exp_cfg.leaderboard_models)} models)...")
    lb_df = ingest_gift_eval_leaderboard(model_folders=exp_cfg.leaderboard_models)
    print(f"  -> Saved leaderboard.parquet: {len(lb_df)} rows across {lb_df['model'].nunique()} models.")

    print("[Phase 1a] Building 97-configuration GIFT-Eval catalog...")
    cat_df = build_gift_eval_catalog()
    print(f"  -> Saved gift_eval_catalog.parquet: {len(cat_df)} configurations.")

    print("[Phase 1a] Downloading stratified GIFT-Eval subset from Hugging Face...")
    for cfg_name in exp_cfg.local_eval_configs:
        if not cfg_name.startswith("unseen_"):
            series_list, min_len = download_and_load_gift_eval_series(
                cfg_name, max_raw_series=exp_cfg.max_series_per_config
            )
            print(f"  -> Cached {cfg_name}: {len(series_list)} univariate series (min_len={min_len})")

    print("[Phase 1c] Building Unseen Dataset 1: Indian Mandi Commodity Prices (Agmarknet)...")
    df_mandi = build_unseen_mandi_dataset()
    print(
        f"  -> Saved unseen_mandi.parquet: {df_mandi['item_id'].nunique()} series, {len(df_mandi)} rows."
    )

    print("[Phase 1c] Building Unseen Dataset 2: Post-Cutoff 2025-2026 Daily Weather (Open-Meteo)...")
    df_rec = build_unseen_recency_dataset()
    print(
        f"  -> Saved unseen_recency.parquet: {df_rec['item_id'].nunique()} series, {len(df_rec)} rows."
    )

    print("[Phase 1d] Fetching S&P 500 Daily OHLC (1990-2026) and Garman-Klass Volatility...")
    df_spx = fetch_spx_daily_ohlc()
    print(f"  -> Saved spx_daily.parquet: {len(df_spx)} trading days ({df_spx['Date'].min().date()} to {df_spx['Date'].max().date()}).")


def main() -> None:
    target = sys.argv[1].lower() if len(sys.argv) > 1 else "all"
    if target == "data":
        run_phase1_data()
    elif target == "baselines":
        from tsdomain.baselines.run_baselines import run_phase2_baselines

        run_phase2_baselines()
    elif target == "features":
        from tsdomain.features.run_features import run_phase3_features

        run_phase3_features()
    elif target == "models":
        from tsdomain.models.run_models import run_phase4_models

        run_phase4_models()
    elif target == "uncertainty":
        from tsdomain.metrics.conformal import run_phase5_uncertainty

        run_phase5_uncertainty()
    elif target == "mixed-models":
        from tsdomain.analysis.mixed_models import run_phase6_mixed_models

        run_phase6_mixed_models()
    elif target == "finance":
        from tsdomain.analysis.finance_case import run_phase7_finance

        run_phase7_finance()
    elif target == "ablations":
        from tsdomain.analysis.ablations import run_phase8_ablations

        run_phase8_ablations()
    elif target == "report":
        from tsdomain.analysis.report_assets import run_phase9_report

        run_phase9_report()
    elif target == "all":
        run_phase1_data()
        from tsdomain.baselines.run_baselines import run_phase2_baselines

        run_phase2_baselines()
        from tsdomain.features.run_features import run_phase3_features

        run_phase3_features()
        from tsdomain.models.run_models import run_phase4_models

        run_phase4_models()
        from tsdomain.metrics.conformal import run_phase5_uncertainty

        run_phase5_uncertainty()
        from tsdomain.analysis.mixed_models import run_phase6_mixed_models

        run_phase6_mixed_models()
        from tsdomain.analysis.finance_case import run_phase7_finance

        run_phase7_finance()
        from tsdomain.analysis.ablations import run_phase8_ablations

        run_phase8_ablations()
        from tsdomain.analysis.report_assets import run_phase9_report

        run_phase9_report()
    else:
        raise SystemExit(f"Unknown CLI target: {target}")


if __name__ == "__main__":
    main()
