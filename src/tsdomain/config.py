"""Schema validation and loaders for YAML configuration files under configs/."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class DatasetConfigEntry(BaseModel):
    config: str
    base_dataset: str
    source: str
    hf_subdir: Optional[str] = None
    local_path: Optional[str] = None
    domain: str
    freq: str
    frequency_group: str
    term: Literal["short", "medium", "long"]
    horizon: int = Field(gt=0)
    season_length: int = Field(ge=1)
    stl_period: Optional[int] = None
    source_url: str
    licence: str
    retrieval_date: str


class DatasetsFileSchema(BaseModel):
    datasets: List[DatasetConfigEntry]


class ModelConfigEntry(BaseModel):
    model_id: str
    family: str
    package: str
    checkpoint_id: Optional[str] = None
    leaderboard_name: Optional[str] = None
    env_file: Optional[str] = None
    context_length: int = Field(gt=0)
    device: Literal["auto", "cpu", "cuda"]
    quantiles: List[float]
    param_count: Optional[int] = None
    tier: Literal["P0", "P1", "P2"]


class ModelsFileSchema(BaseModel):
    models: List[ModelConfigEntry]


class DomainCoverageClaim(BaseModel):
    in_pretraining: Literal["true", "false", "unknown"]
    verified: bool
    source_url: str
    notes: Optional[str] = None


class PretrainingModelEntry(BaseModel):
    checkpoint_id: str
    family: Optional[str] = None
    inherits_from: Optional[str] = None
    pretraining_corpus_name: Optional[str] = None
    training_cutoff: str
    finance_included: bool
    agriculture_mandi_included: Optional[bool] = False
    domains_covered: Optional[Dict[str, DomainCoverageClaim]] = None


class PretrainingRegistrySchema(BaseModel):
    models: Dict[str, PretrainingModelEntry]


class ExperimentSchema(BaseModel):
    seed: int
    context_length: int
    max_series_per_config: int
    max_windows_per_series: int
    meta_feature_max_series: int
    profile_repeats: int
    profile_warmup: int
    quantiles: List[float]
    local_eval_configs: List[str]
    leaderboard_crosscheck_configs: List[str]
    leaderboard_models: List[str]


def load_and_validate_configs(config_dir: Path = Path("configs")) -> Dict[str, Any]:
    """Load and validate all four YAML configuration files against Pydantic schemas."""
    ds_raw = yaml.safe_load((config_dir / "datasets.yaml").read_text(encoding="utf-8"))
    mod_raw = yaml.safe_load((config_dir / "models.yaml").read_text(encoding="utf-8"))
    reg_raw = yaml.safe_load((config_dir / "pretraining_registry.yaml").read_text(encoding="utf-8"))
    exp_raw = yaml.safe_load((config_dir / "experiment.yaml").read_text(encoding="utf-8"))

    ds_cfg = DatasetsFileSchema.model_validate(ds_raw)
    mod_cfg = ModelsFileSchema.model_validate(mod_raw)
    reg_cfg = PretrainingRegistrySchema.model_validate(reg_raw)
    exp_cfg = ExperimentSchema.model_validate(exp_raw)

    # Resolve inheritance in pretraining_registry
    for model_name, entry in reg_cfg.models.items():
        if entry.inherits_from:
            parent = reg_cfg.models.get(entry.inherits_from)
            if parent is None:
                raise ValueError(f"Model {model_name} inherits_from unknown parent {entry.inherits_from}")
            if entry.domains_covered is None and parent.domains_covered is not None:
                entry.domains_covered = parent.domains_covered.copy()
            if entry.family is None:
                entry.family = parent.family
            if entry.pretraining_corpus_name is None:
                entry.pretraining_corpus_name = parent.pretraining_corpus_name

    return {
        "datasets": ds_cfg,
        "models": mod_cfg,
        "pretraining_registry": reg_cfg,
        "experiment": exp_cfg,
    }
