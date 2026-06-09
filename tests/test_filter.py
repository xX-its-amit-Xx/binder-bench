"""Tests for binder_bench.filter module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from binder_bench.config import FilterConfig
from binder_bench.filter import (
    FilterResult,
    filter_design,
    results_to_dataframe,
    run_filter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> FilterConfig:
    """FilterConfig with known thresholds: rmsd < 2.0, plddt > 70."""
    return FilterConfig(
        rmsd_threshold=2.0,
        plddt_threshold=70.0,
        min_binder_length=50,
        max_binder_length=150,
    )


@pytest.fixture
def dummy_paths(tmp_path: Path):
    """Return a pair of (backbone_path, folded_path) that exist on disk."""
    backbone = tmp_path / "backbone.pdb"
    folded = tmp_path / "folded.pdb"
    backbone.touch()
    folded.touch()
    return backbone, folded


# ---------------------------------------------------------------------------
# Helper: build a passing FilterResult
# ---------------------------------------------------------------------------


def _make_result(
    design_id: str,
    rmsd: float,
    plddt: float,
    passes: bool,
    fail_reason: str,
    tmp_path: Path,
) -> FilterResult:
    return FilterResult(
        design_id=design_id,
        backbone_path=tmp_path / f"{design_id}_backbone.pdb",
        folded_path=tmp_path / f"{design_id}_folded.pdb",
        rmsd=rmsd,
        plddt=plddt,
        passes=passes,
        fail_reason=fail_reason,
    )


# ---------------------------------------------------------------------------
# Tests for filter_design logic
# ---------------------------------------------------------------------------


class TestFilterDesign:
    """Unit tests for filter_design with mocked metrics."""

    def _run(
        self,
        backbone_path: Path,
        folded_path: Path,
        config: FilterConfig,
        rmsd_return: float,
        plddt_return: float,
        ca_len: int = 100,
    ) -> FilterResult:
        """Invoke filter_design with patched metrics."""
        with (
            patch(
                "binder_bench.filter.self_consistency_rmsd",
                return_value=rmsd_return,
            ),
            patch(
                "binder_bench.filter.compute_plddt",
                return_value=plddt_return,
            ),
            patch(
                "binder_bench.metrics.extract_ca_coords",
                return_value=[None] * ca_len,
            ),
            patch(
                "binder_bench.filter.extract_ca_coords",
                return_value=[None] * ca_len,
            ),
        ):
            return filter_design("d0", backbone_path, folded_path, config)

    # 1. Design passes both thresholds
    def test_passes_when_both_metrics_within_thresholds(
        self, config: FilterConfig, dummy_paths
    ):
        backbone, folded = dummy_paths
        result = self._run(backbone, folded, config, rmsd_return=1.0, plddt_return=85.0)

        assert result.passes is True
        assert result.fail_reason == ""
        assert result.rmsd == pytest.approx(1.0)
        assert result.plddt == pytest.approx(85.0)

    # 2. RMSD exceeds threshold
    def test_fails_when_rmsd_exceeds_threshold(
        self, config: FilterConfig, dummy_paths
    ):
        backbone, folded = dummy_paths
        result = self._run(backbone, folded, config, rmsd_return=3.0, plddt_return=85.0)

        assert result.passes is False
        assert result.fail_reason == "rmsd"

    # 3. pLDDT below threshold
    def test_fails_when_plddt_below_threshold(
        self, config: FilterConfig, dummy_paths
    ):
        backbone, folded = dummy_paths
        result = self._run(backbone, folded, config, rmsd_return=1.0, plddt_return=55.0)

        assert result.passes is False
        assert result.fail_reason == "plddt"

    # 4. Both metrics fail
    def test_fails_with_both_when_both_metrics_fail(
        self, config: FilterConfig, dummy_paths
    ):
        backbone, folded = dummy_paths
        result = self._run(backbone, folded, config, rmsd_return=3.0, plddt_return=55.0)

        assert result.passes is False
        assert result.fail_reason == "both"

    # 7. Missing file raises an exception -> graceful fallback
    def test_missing_file_returns_failing_result_with_rmsd_minus_one(
        self, config: FilterConfig, tmp_path: Path
    ):
        backbone = tmp_path / "nonexistent_backbone.pdb"
        folded = tmp_path / "nonexistent_folded.pdb"
        # Do NOT touch the files so they remain missing.
        # Patch metrics to raise FileNotFoundError as real code would.
        with (
            patch(
                "binder_bench.filter.self_consistency_rmsd",
                side_effect=FileNotFoundError("file not found"),
            ),
            patch(
                "binder_bench.filter.compute_plddt",
                side_effect=FileNotFoundError("file not found"),
            ),
        ):
            result = filter_design("missing", backbone, folded, config)

        assert result.passes is False
        assert result.rmsd == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Tests for run_filter
# ---------------------------------------------------------------------------


class TestRunFilter:
    """Tests for run_filter collecting multiple designs."""

    def test_run_filter_returns_correct_pass_count(
        self, config: FilterConfig, tmp_path: Path
    ):
        """5 designs (3 pass, 2 fail) -> run_filter returns 3 passing results."""
        # Build a design_map with 5 entries.
        design_ids = [f"d{i}" for i in range(5)]
        design_map = {}
        for did in design_ids:
            bb = tmp_path / f"{did}_backbone.pdb"
            fo = tmp_path / f"{did}_folded.pdb"
            bb.touch()
            fo.touch()
            design_map[did] = (bb, fo)

        # Mock filter_design so the first 3 pass and the last 2 fail.
        call_count = 0

        def mock_filter_design(design_id, backbone_path, folded_path, cfg):
            nonlocal call_count
            idx = call_count
            call_count += 1
            passes = idx < 3
            fail_reason = "" if passes else "rmsd"
            rmsd = 1.0 if passes else 3.0
            return FilterResult(
                design_id=design_id,
                backbone_path=backbone_path,
                folded_path=folded_path,
                rmsd=rmsd,
                plddt=80.0,
                passes=passes,
                fail_reason=fail_reason,
            )

        with patch("binder_bench.filter.filter_design", side_effect=mock_filter_design):
            results = run_filter(design_map, config)

        assert len(results) == 5
        passing = [r for r in results if r.passes]
        assert len(passing) == 3


# ---------------------------------------------------------------------------
# Tests for results_to_dataframe
# ---------------------------------------------------------------------------


class TestResultsToDataframe:
    """Tests for results_to_dataframe columns and dtypes."""

    def _make_results(self, tmp_path: Path) -> list[FilterResult]:
        return [
            _make_result("d0", 1.0, 85.0, True, "", tmp_path),
            _make_result("d1", 3.0, 85.0, False, "rmsd", tmp_path),
            _make_result("d2", 1.0, 55.0, False, "plddt", tmp_path),
        ]

    def test_dataframe_has_expected_columns(self, tmp_path: Path):
        results = self._make_results(tmp_path)
        df = results_to_dataframe(results)

        expected_columns = {
            "design_id",
            "backbone_path",
            "folded_path",
            "rmsd",
            "plddt",
            "passes",
            "fail_reason",
        }
        assert set(df.columns) == expected_columns

    def test_dataframe_row_count_matches_input(self, tmp_path: Path):
        results = self._make_results(tmp_path)
        df = results_to_dataframe(results)
        assert len(df) == len(results)

    def test_dataframe_passes_column_is_bool(self, tmp_path: Path):
        results = self._make_results(tmp_path)
        df = results_to_dataframe(results)
        assert df["passes"].dtype == bool

    def test_dataframe_rmsd_column_is_float(self, tmp_path: Path):
        results = self._make_results(tmp_path)
        df = results_to_dataframe(results)
        assert pd.api.types.is_float_dtype(df["rmsd"])

    def test_dataframe_plddt_column_is_float(self, tmp_path: Path):
        results = self._make_results(tmp_path)
        df = results_to_dataframe(results)
        assert pd.api.types.is_float_dtype(df["plddt"])

    def test_dataframe_paths_are_strings(self, tmp_path: Path):
        results = self._make_results(tmp_path)
        df = results_to_dataframe(results)
        assert pd.api.types.is_string_dtype(df["backbone_path"])
        assert pd.api.types.is_string_dtype(df["folded_path"])

    def test_dataframe_values_match_filter_results(self, tmp_path: Path):
        results = self._make_results(tmp_path)
        df = results_to_dataframe(results)

        for i, result in enumerate(results):
            assert df.loc[i, "design_id"] == result.design_id
            assert df.loc[i, "rmsd"] == pytest.approx(result.rmsd)
            assert df.loc[i, "plddt"] == pytest.approx(result.plddt)
            assert df.loc[i, "passes"] == result.passes
            assert df.loc[i, "fail_reason"] == result.fail_reason
