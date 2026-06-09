from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import yaml
from pydantic import BaseModel, field_validator, model_validator

if TYPE_CHECKING:
    pass


class GenerateConfig(BaseModel):
    backend: Literal["rfdiffusion", "stub"] = "stub"
    n_designs: int = 10
    seed: int = 42
    stub_dir: Path = Path("examples/backbones")
    rfdiffusion_weights: Optional[Path] = None
    target_pdb: Optional[Path] = None
    hotspot_residues: list[str] = []

    @field_validator("n_designs")
    @classmethod
    def validate_n_designs(cls, v: int) -> int:
        if not (1 <= v <= 1000):
            raise ValueError("n_designs must be between 1 and 1000 (inclusive)")
        return v


class DesignConfig(BaseModel):
    backend: Literal["proteinmpnn", "stub"] = "proteinmpnn"
    n_seqs_per_backbone: int = 8
    temperature: float = 0.1
    seed: int = 42
    stub_dir: Path = Path("examples/sequences")

    @field_validator("n_seqs_per_backbone")
    @classmethod
    def validate_n_seqs_per_backbone(cls, v: int) -> int:
        if not (1 <= v <= 64):
            raise ValueError("n_seqs_per_backbone must be between 1 and 64 (inclusive)")
        return v


class FoldConfig(BaseModel):
    backend: Literal["esmfold_api", "stub"] = "stub"
    esmfold_api_url: str = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    stub_dir: Path = Path("examples/structures")
    seed: int = 42


class FilterConfig(BaseModel):
    rmsd_threshold: float = 2.0
    plddt_threshold: float = 70.0
    min_binder_length: int = 50
    max_binder_length: int = 150

    @field_validator("rmsd_threshold")
    @classmethod
    def validate_rmsd_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("rmsd_threshold must be greater than 0")
        return v

    @field_validator("plddt_threshold")
    @classmethod
    def validate_plddt_threshold(cls, v: float) -> float:
        if not (0 < v <= 100):
            raise ValueError("plddt_threshold must be in the range (0, 100]")
        return v

    @model_validator(mode="after")
    def validate_binder_length_range(self) -> "FilterConfig":
        if self.min_binder_length >= self.max_binder_length:
            raise ValueError(
                "min_binder_length must be strictly less than max_binder_length"
            )
        return self


class ScoreConfig(BaseModel):
    contact_distance: float = 8.0
    min_interface_contacts: int = 10
    rosetta_enabled: bool = False
    foldx_enabled: bool = False


class OutputConfig(BaseModel):
    output_dir: Path = Path("output")
    scoreboard_format: Literal["csv", "parquet"] = "parquet"
    save_passing_structures: bool = True
    plot_success_rate: bool = True


class BinderBenchConfig(BaseModel):
    target_name: str
    target_pdb: Path
    generate: GenerateConfig = GenerateConfig()
    design: DesignConfig = DesignConfig()
    fold: FoldConfig = FoldConfig()
    filter: FilterConfig = FilterConfig()
    score: ScoreConfig = ScoreConfig()
    output: OutputConfig = OutputConfig()

    @classmethod
    def from_yaml(cls, path: Path | str) -> "BinderBenchConfig":
        with open(path, "r") as fh:
            data: dict[str, Any] = yaml.safe_load(fh)
        return cls.model_validate(data)

    def save_yaml(self, path: Path | str) -> None:
        data = self.model_dump(mode="json")
        with open(path, "w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
