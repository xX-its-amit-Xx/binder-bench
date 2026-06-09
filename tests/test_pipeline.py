"""End-to-end tests for the stub pipeline."""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Inject a stub binder_bench.design module before any pipeline import so that
# ``from binder_bench.design import run_design`` does not raise ImportError.
# ---------------------------------------------------------------------------
_stub_design_module = types.ModuleType("binder_bench.design")
_stub_design_module.run_design = MagicMock(return_value={})  # type: ignore[attr-defined]
sys.modules.setdefault("binder_bench.design", _stub_design_module)

from binder_bench.config import (  # noqa: E402
    BinderBenchConfig,
    DesignConfig,
    FilterConfig,
    FoldConfig,
    GenerateConfig,
    OutputConfig,
    ScoreConfig,
)
from binder_bench.filter import FilterResult  # noqa: E402
from binder_bench.pipeline import run_pipeline  # noqa: E402
from binder_bench.score import ScoreResult  # noqa: E402


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_stub_pdb(path: Path, n_ca: int = 70) -> None:
    """Write a minimal valid PDB file with *n_ca* CA-only ALA residues.

    Coordinates follow a simple helix-like parametric curve so that the file
    is geometrically non-degenerate and passes Kabsch RMSD without SVD
    singularities.  The B-factor column is set to 75.0 (stub pLDDT).
    """
    deg_to_rad = math.pi / 180.0
    angle_step = 100.0 * deg_to_rad
    lines: list[str] = []
    for i in range(n_ca):
        serial = i + 1
        res_seq = i + 1
        x = 1.5 * math.cos(i * angle_step)
        y = 1.5 * math.sin(i * angle_step)
        z = 1.5 * i
        line = (
            f"ATOM  {serial:5d}  CA  ALA A{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 75.00           C  "
        )
        lines.append(line)
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Shared fixture data builders
# ---------------------------------------------------------------------------

def _make_backbones(tmp_path: Path, n: int = 3) -> list[Path]:
    """Write *n* backbone PDB files and return their paths."""
    backbones_dir = tmp_path / "backbones"
    backbones_dir.mkdir(exist_ok=True)
    paths: list[Path] = []
    for i in range(n):
        p = backbones_dir / f"backbone_{i:03d}.pdb"
        make_stub_pdb(p, n_ca=70)
        paths.append(p)
    return paths


def _make_sequences(backbones: list[Path], n_seqs: int = 2) -> dict:
    """Return a {Path: [str, ...]} mapping with synthetic sequences."""
    seq_map: dict = {}
    for bb in backbones:
        seqs = [f"ACDEFGHIKLMNPQRSTVWY" * 4 for _ in range(n_seqs)]
        seq_map[bb] = seqs
    return seq_map


def _make_folded(tmp_path: Path, sequences: dict) -> dict[str, Path]:
    """Write stub folded PDB files and return a design_id -> Path mapping."""
    fold_dir = tmp_path / "folded"
    fold_dir.mkdir(exist_ok=True)
    folded: dict[str, Path] = {}
    for bb_path, seqs in sequences.items():
        bb_stem = bb_path.stem
        for idx in range(len(seqs)):
            design_id = f"{bb_stem}_seq{idx:03d}"
            p = fold_dir / f"{design_id}.pdb"
            make_stub_pdb(p, n_ca=70)
            folded[design_id] = p
    return folded


def _make_filter_results(
    backbones: list[Path],
    folded: dict[str, Path],
    rmsd: float = 1.0,
    plddt: float = 75.0,
) -> list[FilterResult]:
    """Build FilterResult objects that pass the lenient stub thresholds."""
    results: list[FilterResult] = []
    for bb_path, folded_path in zip(
        [b for b in backbones for _ in range(2)],
        folded.values(),
    ):
        design_id = folded_path.stem
        results.append(
            FilterResult(
                design_id=design_id,
                backbone_path=bb_path,
                folded_path=folded_path,
                rmsd=rmsd,
                plddt=plddt,
                passes=True,
                fail_reason="",
            )
        )
    return results


def _make_score_results(filter_results: list[FilterResult]) -> list[ScoreResult]:
    """Build ScoreResult objects from FilterResults, ranked in order."""
    score_results: list[ScoreResult] = []
    for rank, fr in enumerate(filter_results, start=1):
        score_results.append(
            ScoreResult(
                design_id=fr.design_id,
                backbone_path=fr.backbone_path,
                folded_path=fr.folded_path,
                rmsd=fr.rmsd,
                plddt=fr.plddt,
                passes_filter=fr.passes,
                n_interface_contacts=5,
                interface_score=50.0,
                rosetta_score=None,
                foldx_score=None,
                rank=rank,
            )
        )
    return score_results


def _make_scoreboard(score_results: list[ScoreResult]) -> pd.DataFrame:
    """Build the scoreboard DataFrame from ScoreResults."""
    from binder_bench.score import results_to_scoreboard

    return results_to_scoreboard(score_results)


def _make_stats(score_results: list[ScoreResult]) -> dict:
    """Return aggregate stats dict for the given score results."""
    df = _make_scoreboard(score_results)
    n_total = len(df)
    passing = df[df["passes_filter"]]
    n_pass = len(passing)
    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "success_rate": n_pass / n_total if n_total > 0 else 0.0,
        "mean_plddt_pass": float(passing["plddt"].mean()) if n_pass > 0 else float("nan"),
        "mean_rmsd_pass": float(passing["rmsd"].mean()) if n_pass > 0 else float("nan"),
        "mean_contacts_pass": float(passing["n_interface_contacts"].mean())
        if n_pass > 0
        else float("nan"),
        "best_design_id": str(df.loc[df["rank"] == 1, "design_id"].iloc[0])
        if n_total > 0
        else None,
    }


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_config(tmp_path: Path) -> BinderBenchConfig:
    """Minimal BinderBenchConfig that exercises the stub pipeline."""
    backbones_dir = tmp_path / "backbones"
    sequences_dir = tmp_path / "sequences"
    structures_dir = tmp_path / "structures"
    for d in [backbones_dir, sequences_dir, structures_dir]:
        d.mkdir()

    # Write the target PDB used by the score step
    target_pdb = tmp_path / "target.pdb"
    make_stub_pdb(target_pdb, n_ca=70)

    config = BinderBenchConfig(
        target_name="test_target",
        target_pdb=target_pdb,
        generate=GenerateConfig(
            backend="stub",
            n_designs=3,
            stub_dir=backbones_dir,
            seed=42,
        ),
        design=DesignConfig(
            backend="stub",
            n_seqs_per_backbone=2,
            stub_dir=sequences_dir,
        ),
        fold=FoldConfig(backend="stub", stub_dir=structures_dir),
        filter=FilterConfig(
            rmsd_threshold=5.0,
            plddt_threshold=50.0,
            min_binder_length=50,
            max_binder_length=150,
        ),
        score=ScoreConfig(),
        output=OutputConfig(
            output_dir=tmp_path / "output",
            scoreboard_format="csv",
        ),
    )
    return config


# ---------------------------------------------------------------------------
# Helpers used by all tests to build consistent mock pipeline state
# ---------------------------------------------------------------------------

def _build_pipeline_mocks(tmp_path: Path, stub_config: BinderBenchConfig):
    """Return (backbones, sequences, folded, filter_results, score_results, scoreboard, stats)."""
    backbones = _make_backbones(tmp_path, n=3)

    # Populate the generate stub_dir so StubGenerator has files to load
    for bb in backbones:
        dest = stub_config.generate.stub_dir / bb.name
        if not dest.exists():
            dest.write_bytes(bb.read_bytes())

    sequences = _make_sequences(backbones, n_seqs=2)
    folded = _make_folded(tmp_path, sequences)
    filter_results = _make_filter_results(backbones, folded)
    score_results = _make_score_results(filter_results)
    scoreboard = _make_scoreboard(score_results)
    stats = _make_stats(score_results)
    return backbones, sequences, folded, filter_results, score_results, scoreboard, stats


def _patch_pipeline(
    backbones,
    sequences,
    folded,
    filter_results,
    score_results,
    scoreboard,
    stats,
):
    """Return a list of patch context managers that replace all pipeline steps."""
    return [
        patch("binder_bench.pipeline.run_generate", return_value=backbones),
        patch("binder_bench.pipeline.run_design", return_value=sequences),
        patch("binder_bench.pipeline.run_fold", return_value=folded),
        patch("binder_bench.pipeline.run_filter", return_value=filter_results),
        patch("binder_bench.pipeline.run_score", return_value=score_results),
        patch("binder_bench.pipeline.results_to_scoreboard", return_value=scoreboard),
        patch("binder_bench.pipeline.compute_aggregate_stats", return_value=stats),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestStubPipeline:
    """Full end-to-end stub pipeline tests."""

    def test_stub_pipeline_runs(self, tmp_path: Path, stub_config: BinderBenchConfig) -> None:
        """run_pipeline completes without raising an exception."""
        (
            backbones, sequences, folded,
            filter_results, score_results, scoreboard, stats,
        ) = _build_pipeline_mocks(tmp_path, stub_config)

        patches = _patch_pipeline(
            backbones, sequences, folded,
            filter_results, score_results, scoreboard, stats,
        )

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            run = run_pipeline(stub_config)

        assert run is not None, "run_pipeline must return a PipelineRun"
        assert run.elapsed_seconds >= 0.0

    def test_stub_pipeline_scoreboard(
        self, tmp_path: Path, stub_config: BinderBenchConfig
    ) -> None:
        """Scoreboard DataFrame has the expected columns."""
        expected_columns = {"design_id", "rmsd", "plddt", "passes_filter", "rank"}

        (
            backbones, sequences, folded,
            filter_results, score_results, scoreboard, stats,
        ) = _build_pipeline_mocks(tmp_path, stub_config)

        patches = _patch_pipeline(
            backbones, sequences, folded,
            filter_results, score_results, scoreboard, stats,
        )

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            run = run_pipeline(stub_config)

        assert run.scoreboard is not None, "run.scoreboard must not be None"
        assert not run.scoreboard.empty, "scoreboard must not be empty"
        missing = expected_columns - set(run.scoreboard.columns)
        assert not missing, f"scoreboard is missing columns: {missing}"

    def test_stub_pipeline_output_files(
        self, tmp_path: Path, stub_config: BinderBenchConfig
    ) -> None:
        """Output directory contains scoreboard.csv and stats.json after a run."""
        (
            backbones, sequences, folded,
            filter_results, score_results, scoreboard, stats,
        ) = _build_pipeline_mocks(tmp_path, stub_config)

        patches = _patch_pipeline(
            backbones, sequences, folded,
            filter_results, score_results, scoreboard, stats,
        )

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            run = run_pipeline(stub_config)

        output_dir = stub_config.output.output_dir
        # The pipeline writes into output_dir / run_id /
        run_dirs = list(output_dir.iterdir()) if output_dir.exists() else []
        assert run_dirs, f"output_dir {output_dir} should contain at least one run sub-directory"

        run_dir = run_dirs[0]
        scoreboard_csv = run_dir / "scoreboard.csv"
        stats_json = run_dir / "stats.json"

        assert scoreboard_csv.exists(), f"scoreboard.csv not found in {run_dir}"
        assert stats_json.exists(), f"stats.json not found in {run_dir}"

        # Sanity-check that stats.json is valid JSON
        with open(stats_json) as fh:
            data = json.load(fh)
        assert isinstance(data, dict), "stats.json should contain a JSON object"

    def test_seed_determinism(self, tmp_path: Path) -> None:
        """Two runs with identical seed and stub data produce identical scoreboards."""
        target_pdb = tmp_path / "target.pdb"
        make_stub_pdb(target_pdb, n_ca=70)

        def _make_config(output_subdir: str) -> BinderBenchConfig:
            backbones_dir = tmp_path / "backbones"
            sequences_dir = tmp_path / "sequences"
            structures_dir = tmp_path / "structures"
            for d in [backbones_dir, sequences_dir, structures_dir]:
                d.mkdir(exist_ok=True)
            return BinderBenchConfig(
                target_name="test_target",
                target_pdb=target_pdb,
                generate=GenerateConfig(
                    backend="stub",
                    n_designs=3,
                    stub_dir=backbones_dir,
                    seed=42,
                ),
                design=DesignConfig(
                    backend="stub",
                    n_seqs_per_backbone=2,
                    stub_dir=sequences_dir,
                ),
                fold=FoldConfig(backend="stub", stub_dir=structures_dir),
                filter=FilterConfig(
                    rmsd_threshold=5.0,
                    plddt_threshold=50.0,
                    min_binder_length=50,
                    max_binder_length=150,
                ),
                score=ScoreConfig(),
                output=OutputConfig(
                    output_dir=tmp_path / output_subdir,
                    scoreboard_format="csv",
                ),
            )

        config_a = _make_config("output_a")
        config_b = _make_config("output_b")

        # Build backbones once and share via the stub_dir (same path for both configs).
        backbones = _make_backbones(tmp_path, n=3)
        shared_backbones_dir = tmp_path / "backbones"
        for bb in backbones:
            dest = shared_backbones_dir / bb.name
            if not dest.exists():
                dest.write_bytes(bb.read_bytes())

        sequences = _make_sequences(backbones, n_seqs=2)
        folded = _make_folded(tmp_path, sequences)
        filter_results = _make_filter_results(backbones, folded)
        score_results = _make_score_results(filter_results)
        scoreboard = _make_scoreboard(score_results)
        stats = _make_stats(score_results)

        patches = _patch_pipeline(
            backbones, sequences, folded,
            filter_results, score_results, scoreboard, stats,
        )

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            run_a = run_pipeline(config_a)

        # Reset patches for second run (new patch context managers needed)
        patches2 = _patch_pipeline(
            backbones, sequences, folded,
            filter_results, score_results, scoreboard, stats,
        )

        with (
            patches2[0],
            patches2[1],
            patches2[2],
            patches2[3],
            patches2[4],
            patches2[5],
            patches2[6],
        ):
            run_b = run_pipeline(config_b)

        assert run_a.scoreboard is not None, "run_a scoreboard must not be None"
        assert run_b.scoreboard is not None, "run_b scoreboard must not be None"

        # Compare sorted scoreboards (order by design_id for stability)
        sb_a = run_a.scoreboard.sort_values("design_id").reset_index(drop=True)
        sb_b = run_b.scoreboard.sort_values("design_id").reset_index(drop=True)

        pd.testing.assert_frame_equal(
            sb_a,
            sb_b,
            check_like=False,
            obj="scoreboard from two identical-seed runs",
        )
