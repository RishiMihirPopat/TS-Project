"""Phase 9 Acceptance Tests: Compute-Efficiency Pareto Frontier & Report Assets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsdomain.utils import PROJECT_ROOT

EXPECTED_FIGURES = [
    "fig1_domain_shift_heatmap",
    "fig2_similarity_vs_gain",
    "fig3_lodo_cv_calibration",
    "fig4_pareto_compute_accuracy",
    "fig5_conformal_calibration",
    "fig6_finance_structural_breaks",
]

EXPECTED_TABLES = [
    "table1_dataset_summary",
    "table2_pretraining_registry",
    "table3_mixed_effects_ladder",
    "table4_lodo_cv_comparison",
    "table5_compute_pareto",
    "table6_finance_and_ablations",
]


def test_phase9_pareto_and_h4_outputs() -> None:
    """Verify Pareto frontier tables and H4 decision boundary exist and have valid metrics."""
    analysis_dir = PROJECT_ROOT / "results" / "analysis"
    pt_path = analysis_dir / "pareto_table.csv"
    pm_path = analysis_dir / "pareto_summary_by_model.csv"
    h4_path = analysis_dir / "h4_decision_boundary.csv"

    assert pt_path.exists(), "Missing pareto_table.csv"
    assert pm_path.exists(), "Missing pareto_summary_by_model.csv"
    assert h4_path.exists(), "Missing h4_decision_boundary.csv"

    pt_df = pd.read_csv(pt_path)
    assert "energy_per_1pct_mase_gain_mwh" in pt_df.columns
    assert "pareto_energy" in pt_df.columns
    assert pt_df["pareto_energy"].any(), "At least one model must be Pareto-optimal per config"

    pm_df = pd.read_csv(pm_path)
    assert pm_df["global_pareto_energy"].any(), "Global Pareto energy frontier must not be empty"
    assert np.isfinite(pm_df["mean_mase"].to_numpy()).all()
    assert np.isfinite(pm_df["mean_energy_mwh_per_1k_pts"].to_numpy()).all()


def test_phase9_figures_tables_and_report_md() -> None:
    """Verify all 6 figures (PNG+PDF), 6 tables (CSV+MD), and report.md exist with no placeholders."""
    report_dir = PROJECT_ROOT / "report"
    fig_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"

    for stem in EXPECTED_FIGURES:
        png_p = fig_dir / f"{stem}.png"
        pdf_p = fig_dir / f"{stem}.pdf"
        assert png_p.exists() and png_p.stat().st_size > 5_000, f"Missing or empty {png_p}"
        assert pdf_p.exists() and pdf_p.stat().st_size > 2_000, f"Missing or empty {pdf_p}"

    for stem in EXPECTED_TABLES:
        csv_p = tables_dir / f"{stem}.csv"
        md_p = tables_dir / f"{stem}.md"
        assert csv_p.exists() and csv_p.stat().st_size > 50, f"Missing or empty {csv_p}"
        assert md_p.exists() and md_p.stat().st_size > 50, f"Missing or empty {md_p}"

    report_md = report_dir / "report.md"
    assert report_md.exists(), "Missing report/report.md"
    txt = report_md.read_text(encoding="utf-8")
    assert len(txt) > 5_000
    assert "TODO" not in txt
    assert "TBD" not in txt
