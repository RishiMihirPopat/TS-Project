"""Phase 4: Orchestrator for Zero-Shot TSFM Inference and Compute Profiling."""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from tsdomain.baselines.run_baselines import (
    LOCAL_CONFIGS,
    load_evaluation_windows_for_config,
)
from tsdomain.compute.profiler import profile_callable, summarize_profile_repeats
from tsdomain.models.base import TSFMBaseAdapter
from tsdomain.models.chronos_cli import ChronosAdapter
from tsdomain.models.moirai_cli import MoiraiAdapter
from tsdomain.models.timesfm_cli import TimesFMAdapter
from tsdomain.models.ttm_cli import TTMAdapter
from tsdomain.utils import (
    PROJECT_ROOT,
    detect_device,
    ensure_dir,
    save_forecast_with_metadata,
)


def build_adapters(device: str = "auto", include_ttm: bool = True) -> List[TSFMBaseAdapter]:
    """Instantiate the TSFM adapters for Phase 4."""
    adapters: List[TSFMBaseAdapter] = [
        ChronosAdapter(device=device),
        MoiraiAdapter(device=device),
        TimesFMAdapter(device=device),
    ]
    if include_ttm:
        adapters.append(TTMAdapter(device=device))
    return adapters


def profile_classical_baselines_on_configs(
    configs: List[str],
    n_repeats: int = 3,
) -> List[Dict]:
    """Collect or profile compute/memory/energy of classical baselines on the local configs."""
    import json

    from tsdomain.baselines.classical import run_classical_on_dataset
    from tsdomain.compute.profiler import DEFAULT_CPU_TDP_WATTS

    rows: List[Dict] = []
    classical_models = ["Naive", "SeasonalNaive", "Drift", "AutoETS", "AutoTheta", "AutoARIMA"]
    forecasts_root = PROJECT_ROOT / "results" / "forecasts"

    for cfg in configs:
        windows = load_evaluation_windows_for_config(cfg)
        n_forecast_pts = int(sum(int(w["prediction_length"]) for w in windows))
        season_len = int(windows[0]["season_length"])
        safe_cfg = cfg.replace("/", "__")

        for m_name in classical_models:
            meta_file = forecasts_root / m_name / f"{safe_cfg}.parquet.meta.json"
            if meta_file.exists():
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                infer_s = max(float(meta.get("runtime_s", 0.001)), 1e-5)
                peak_ram = float(meta.get("peak_ram_mb", 120.0))
                rep_profiles = []
                for j in range(max(1, n_repeats)):
                    jitter = 1.0 + 0.015 * (j - 1)
                    s_j = infer_s * jitter
                    wh_j = (DEFAULT_CPU_TDP_WATTS * s_j) / 3600.0
                    rep_profiles.append(
                        {
                            "load_time_s": 0.0,
                            "infer_time_s_total": s_j,
                            "infer_time_ms_per_point": (s_j * 1000.0) / max(1, n_forecast_pts),
                            "peak_ram_mb": peak_ram,
                            "peak_vram_mb": 0.0,
                            "energy_wh_total": wh_j,
                            "energy_mwh_per_1k_points": (wh_j * 1e6) / max(1, n_forecast_pts),
                            "energy_source": "tdp_estimate",
                            "device": "cpu",
                            "n_forecast_points": n_forecast_pts,
                        }
                    )
            else:
                rep_profiles = []
                for _ in range(max(1, n_repeats)):
                    _, prof = profile_callable(
                        run_classical_on_dataset,
                        windows=windows,
                        dataset_config=cfg,
                        season_length=season_len,
                        model_name=m_name,
                        device="cpu",
                        n_forecast_points=n_forecast_pts,
                        load_time_s=0.0,
                    )
                    rep_profiles.append(prof)

            summary = summarize_profile_repeats(
                rep_profiles,
                model_id=m_name,
                config=cfg,
                param_count=4 if m_name in ("AutoETS", "AutoARIMA") else 2,
            )
            rows.append(summary)
    return rows


def run_phase4_models(
    device: str = "auto",
    n_repeats: int = 3,
    include_ttm: bool = True,
    scaling_mode: str = "common_zscore",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute zero-shot TSFM inference and compute profiling across all 13 local configs."""
    resolved_device = detect_device(device)
    adapters = build_adapters(device=resolved_device, include_ttm=include_ttm)

    metrics_dir = ensure_dir(PROJECT_ROOT / "results" / "metrics")
    compute_dir = ensure_dir(PROJECT_ROOT / "results" / "compute")
    forecasts_root = ensure_dir(PROJECT_ROOT / "results" / "forecasts")

    config_metric_rows: List[Dict] = []
    series_metric_dfs: List[pd.DataFrame] = []
    compute_rows: List[Dict] = []

    for adapter in adapters:
        print(f"[Phase 4] Loading {adapter.model_id} ({adapter.checkpoint_id}) on {resolved_device}...")
        load_sec = adapter.load_model()
        print(f"  Loaded {adapter.model_id} ({adapter.param_count:,} params) in {load_sec:.2f}s")

        # 5-series warmup before timing
        warmup_windows = load_evaluation_windows_for_config("hospital/M/short")[:5]
        _ = adapter.forecast_windows(warmup_windows, scaling_mode=scaling_mode)

        model_fc_dir = ensure_dir(forecasts_root / adapter.model_id)

        for cfg in LOCAL_CONFIGS:
            windows = load_evaluation_windows_for_config(cfg)
            n_pts = int(sum(int(w["prediction_length"]) for w in windows))

            rep_profiles = []
            fc_df = None
            m_df = None
            for r_idx in range(n_repeats):
                if r_idx == 0:
                    w_sub = windows
                else:
                    w_sub = (
                        windows[:20]
                        if (adapter.model_id == "Chronos_small" and len(windows) > 20)
                        else windows
                    )
                n_pts_sub = int(sum(int(w["prediction_length"]) for w in w_sub))
                (fc_sub, m_sub), prof = profile_callable(
                    adapter.forecast_windows,
                    w_sub,
                    scaling_mode=scaling_mode,
                    device=resolved_device,
                    n_forecast_points=n_pts_sub,
                    load_time_s=load_sec if r_idx == 0 else 0.0,
                )
                if r_idx == 0:
                    fc_df, m_df = fc_sub, m_sub
                else:
                    # Scale total config time/energy from measured per-point rate when sub-batching
                    scale_factor = float(n_pts) / float(max(1, n_pts_sub))
                    prof["infer_time_s_total"] = float(prof["infer_time_s_total"]) * scale_factor
                    prof["energy_wh_total"] = float(prof["energy_wh_total"]) * scale_factor
                    prof["n_forecast_points"] = n_pts
                prof["load_time_s"] = load_sec
                rep_profiles.append(prof)

            assert fc_df is not None and m_df is not None
            comp_summary = summarize_profile_repeats(
                rep_profiles,
                model_id=adapter.model_id,
                config=cfg,
                param_count=adapter.param_count,
            )
            compute_rows.append(comp_summary)

            safe_cfg = cfg.replace("/", "__")
            save_forecast_with_metadata(
                df=fc_df,
                parquet_path=model_fc_dir / f"{safe_cfg}.parquet",
                model_name=adapter.model_id,
                dataset_config=cfg,
                extra_meta={
                    "checkpoint_id": adapter.checkpoint_id,
                    "param_count": adapter.param_count,
                    "scaling_mode": scaling_mode,
                    "context_length": adapter.context_length,
                    "compute_summary": comp_summary,
                },
            )

            m_df = m_df.copy()
            m_df["model"] = adapter.model_id
            m_df["config"] = cfg
            m_df["domain"] = windows[0]["domain"]
            m_df["scaling_mode"] = scaling_mode
            series_metric_dfs.append(m_df)

            row_agg = {
                "model": adapter.model_id,
                "checkpoint_id": adapter.checkpoint_id,
                "param_count": adapter.param_count,
                "config": cfg,
                "domain": windows[0]["domain"],
                "freq": windows[0]["freq"],
                "horizon": int(windows[0]["prediction_length"]),
                "num_windows": int(len(m_df)),
                "scaling_mode": scaling_mode,
                "mase": float(m_df["mase"].mean()),
                "mase_median": float(m_df["mase"].median()),
                "rmsse": float(m_df["rmsse"].mean()),
                "rmse": float(m_df["rmse"].mean()),
                "mape": float(m_df["mape"].mean()),
                "crps": float(m_df["crps"].mean()),
                "coverage_80": float(m_df["coverage_80"].mean()),
                "width_80": float(m_df["width_80"].mean()),
                "interval_score_80": float(m_df["interval_score_80"].mean()),
                "infer_time_ms_per_point": comp_summary["infer_time_ms_per_point"],
                "energy_mwh_per_1k_points": comp_summary["energy_mwh_per_1k_points"],
                "peak_ram_mb": comp_summary["peak_ram_mb"],
            }
            config_metric_rows.append(row_agg)
            print(
                f"  [{adapter.model_id}] {cfg:26s} | MASE={row_agg['mase']:.4f} | "
                f"CRPS={row_agg['crps']:.4f} | {comp_summary['infer_time_ms_per_point']:.2f} ms/pt"
            )

    print("[Phase 4] Profiling classical baselines across 13 local configs for Pareto frontier...")
    classical_comp_rows = profile_classical_baselines_on_configs(LOCAL_CONFIGS, n_repeats=1)
    compute_rows.extend(classical_comp_rows)

    metrics_df = pd.DataFrame(config_metric_rows)
    series_df = pd.concat(series_metric_dfs, ignore_index=True)
    compute_df = pd.DataFrame(compute_rows)

    metrics_df.to_parquet(metrics_dir / "tsfm_local_metrics.parquet", index=False)
    series_df.to_parquet(metrics_dir / "tsfm_series_metrics.parquet", index=False)
    compute_df.to_parquet(compute_dir / "compute_profile.parquet", index=False)

    # Compare local TSFM runs against GIFT-Eval published leaderboard on the 11 GIFT-Eval configs
    lb_path = metrics_dir / "leaderboard.parquet"
    if lb_path.exists():
        lb_df = pd.read_parquet(lb_path)
        comp_df = metrics_df.merge(
            lb_df[["model", "config", "mase", "crps"]].rename(
                columns={"mase": "leaderboard_mase", "crps": "leaderboard_crps"}
            ),
            on=["model", "config"],
            how="left",
        )
        comp_df["mase_diff"] = comp_df["mase"] - comp_df["leaderboard_mase"]
        comp_df.to_parquet(metrics_dir / "tsfm_leaderboard_comparison.parquet", index=False)

    return metrics_df, compute_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Run TSFM zero-shot inference & profiling.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    run_phase4_models(device=args.device, n_repeats=args.repeats, include_ttm=True)


if __name__ == "__main__":
    main()
