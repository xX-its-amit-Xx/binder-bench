"""Score the binder-target interface and aggregate results into a ranked scoreboard."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from binder_bench.config import ScoreConfig
from binder_bench.metrics import extract_ca_coords, compute_interface_contacts
from binder_bench.filter import FilterResult

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    design_id: str
    backbone_path: Path
    folded_path: Path
    rmsd: float
    plddt: float
    passes_filter: bool
    n_interface_contacts: int
    interface_score: float  # normalized: contacts / max_possible * 100
    rosetta_score: Optional[float] = None
    foldx_score: Optional[float] = None
    rank: int = 0  # filled after sorting


def score_design(
    filter_result: FilterResult,
    target_pdb: Path,
    config: ScoreConfig,
) -> ScoreResult:
    """Score a single design by computing interface contacts.

    Parameters
    ----------
    filter_result : FilterResult
        Result from the filter stage containing paths, rmsd, plddt, and
        filter pass/fail status.
    target_pdb : Path
        Path to the target PDB file.
    config : ScoreConfig
        Scoring configuration (contact distance, optional Rosetta/FoldX flags).

    Returns
    -------
    ScoreResult
        Populated score result; returns a zero-score result on any error.
    """
    try:
        binder_ca = extract_ca_coords(filter_result.folded_path)
        target_ca = extract_ca_coords(target_pdb)

        n_contacts = compute_interface_contacts(
            binder_ca,
            target_ca,
            distance_threshold=config.contact_distance,
        )

        n_binder_residues = max(1, len(binder_ca))
        interface_score = (n_contacts / n_binder_residues) * 100.0

        rosetta_score: Optional[float] = None
        foldx_score: Optional[float] = None

        if config.rosetta_enabled or config.foldx_enabled:
            logger.warning("Rosetta/FoldX scoring not available in stub mode")

        return ScoreResult(
            design_id=filter_result.design_id,
            backbone_path=filter_result.backbone_path,
            folded_path=filter_result.folded_path,
            rmsd=filter_result.rmsd,
            plddt=filter_result.plddt,
            passes_filter=filter_result.passes,
            n_interface_contacts=n_contacts,
            interface_score=interface_score,
            rosetta_score=rosetta_score,
            foldx_score=foldx_score,
        )

    except Exception as exc:
        logger.error(
            "Failed to score design %s: %s", filter_result.design_id, exc
        )
        return ScoreResult(
            design_id=filter_result.design_id,
            backbone_path=filter_result.backbone_path,
            folded_path=filter_result.folded_path,
            rmsd=filter_result.rmsd,
            plddt=filter_result.plddt,
            passes_filter=False,
            n_interface_contacts=0,
            interface_score=0.0,
            rosetta_score=None,
            foldx_score=None,
        )


def run_score(
    filter_results: list[FilterResult],
    target_pdb: Path,
    config: ScoreConfig,
) -> list[ScoreResult]:
    """Score all designs and return them in ranked order.

    All designs are scored regardless of whether they pass the filter.
    Results are sorted by:
      1. passes_filter descending (passing designs first)
      2. plddt descending
      3. rmsd ascending
      4. interface_score descending

    Rank is 1-indexed and filled in after sorting.

    Parameters
    ----------
    filter_results : list[FilterResult]
        All filter results from the filter stage.
    target_pdb : Path
        Path to the target PDB file.
    config : ScoreConfig
        Scoring configuration.

    Returns
    -------
    list[ScoreResult]
        Ranked list of score results.
    """
    results: list[ScoreResult] = []
    for fr in filter_results:
        result = score_design(fr, target_pdb, config)
        results.append(result)

    results.sort(
        key=lambda r: (
            not r.passes_filter,   # False sorts before True, so negate to get desc
            -r.plddt,              # descending plddt
            r.rmsd,                # ascending rmsd
            -r.interface_score,    # descending interface_score
        )
    )

    for idx, result in enumerate(results, start=1):
        result.rank = idx

    top_n = min(3, len(results))
    for result in results[:top_n]:
        logger.info(
            "Rank %d | design_id=%s | passes_filter=%s | plddt=%.2f | "
            "rmsd=%.3f | interface_score=%.2f | n_contacts=%d",
            result.rank,
            result.design_id,
            result.passes_filter,
            result.plddt,
            result.rmsd,
            result.interface_score,
            result.n_interface_contacts,
        )

    return results


def results_to_scoreboard(results: list[ScoreResult]) -> pd.DataFrame:
    """Convert a list of ScoreResult objects into a pandas DataFrame.

    Parameters
    ----------
    results : list[ScoreResult]
        Scored and ranked design results.

    Returns
    -------
    pd.DataFrame
        One row per design; columns match ScoreResult fields with correct dtypes.
    """
    rows = [
        {
            "design_id": r.design_id,
            "rank": r.rank,
            "backbone_path": str(r.backbone_path),
            "folded_path": str(r.folded_path),
            "rmsd": r.rmsd,
            "plddt": r.plddt,
            "passes_filter": r.passes_filter,
            "n_interface_contacts": r.n_interface_contacts,
            "interface_score": r.interface_score,
            "rosetta_score": r.rosetta_score,
            "foldx_score": r.foldx_score,
        }
        for r in results
    ]

    if not rows:
        df = pd.DataFrame(
            columns=[
                "design_id",
                "rank",
                "backbone_path",
                "folded_path",
                "rmsd",
                "plddt",
                "passes_filter",
                "n_interface_contacts",
                "interface_score",
                "rosetta_score",
                "foldx_score",
            ]
        )
        df = df.astype(
            {
                "design_id": str,
                "rank": int,
                "rmsd": float,
                "plddt": float,
                "passes_filter": bool,
                "n_interface_contacts": int,
                "interface_score": float,
            }
        )
        return df

    df = pd.DataFrame(rows)
    df = df.astype(
        {
            "design_id": str,
            "rank": int,
            "rmsd": float,
            "plddt": float,
            "passes_filter": bool,
            "n_interface_contacts": int,
            "interface_score": float,
        }
    )
    return df


def compute_aggregate_stats(df: pd.DataFrame) -> dict:
    """Compute aggregate statistics over all scored designs.

    Parameters
    ----------
    df : pd.DataFrame
        Scoreboard DataFrame as returned by results_to_scoreboard.

    Returns
    -------
    dict
        Dictionary with keys:
        - n_total: total number of designs
        - n_pass: number of designs that pass the filter
        - success_rate: fraction of designs passing the filter (0.0 if n_total == 0)
        - mean_plddt_pass: mean pLDDT of passing designs (NaN if none pass)
        - mean_rmsd_pass: mean RMSD of passing designs (NaN if none pass)
        - mean_contacts_pass: mean interface contacts of passing designs (NaN if none pass)
        - best_design_id: design_id with rank == 1; None if no designs
    """
    n_total = len(df)

    if n_total == 0:
        return {
            "n_total": 0,
            "n_pass": 0,
            "success_rate": 0.0,
            "mean_plddt_pass": float("nan"),
            "mean_rmsd_pass": float("nan"),
            "mean_contacts_pass": float("nan"),
            "best_design_id": None,
        }

    passing = df[df["passes_filter"]]
    n_pass = len(passing)
    success_rate = n_pass / n_total

    mean_plddt_pass = float(passing["plddt"].mean()) if n_pass > 0 else float("nan")
    mean_rmsd_pass = float(passing["rmsd"].mean()) if n_pass > 0 else float("nan")
    mean_contacts_pass = (
        float(passing["n_interface_contacts"].mean()) if n_pass > 0 else float("nan")
    )

    best_row = df[df["rank"] == 1]
    best_design_id: Optional[str] = (
        str(best_row["design_id"].iloc[0]) if len(best_row) > 0 else None
    )

    return {
        "n_total": n_total,
        "n_pass": n_pass,
        "success_rate": success_rate,
        "mean_plddt_pass": mean_plddt_pass,
        "mean_rmsd_pass": mean_rmsd_pass,
        "mean_contacts_pass": mean_contacts_pass,
        "best_design_id": best_design_id,
    }
