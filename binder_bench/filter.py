"""Filter designs by self-consistency (backbone RMSD + pLDDT)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from binder_bench.config import FilterConfig
from binder_bench.metrics import self_consistency_rmsd, compute_plddt, extract_ca_coords

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    design_id: str
    backbone_path: Path
    folded_path: Path
    rmsd: float
    plddt: float
    passes: bool
    fail_reason: str  # "" if passes, else "rmsd" | "plddt" | "length" | "both"


def filter_design(
    design_id: str,
    backbone_path: Path,
    folded_path: Path,
    config: FilterConfig,
) -> FilterResult:
    """Compute RMSD + pLDDT for one design and apply FilterConfig thresholds.

    Parameters
    ----------
    design_id : str
        Unique identifier for this design.
    backbone_path : Path
        Path to the designed backbone PDB file.
    folded_path : Path
        Path to the folded structure PDB file (e.g. ESMFold prediction).
    config : FilterConfig
        Thresholds and length bounds to apply.

    Returns
    -------
    FilterResult
        Populated result.  If structure reading fails a warning is logged and a
        failing result with rmsd=-1, plddt=-1 is returned.
    """
    try:
        rmsd = self_consistency_rmsd(backbone_path, folded_path)
        plddt = compute_plddt(folded_path)
    except Exception as exc:
        logger.warning(
            "Failed to compute metrics for design '%s': %s", design_id, exc
        )
        return FilterResult(
            design_id=design_id,
            backbone_path=backbone_path,
            folded_path=folded_path,
            rmsd=-1.0,
            plddt=-1.0,
            passes=False,
            fail_reason="length",
        )

    # Determine the number of CA atoms in the backbone to check length bounds.
    # Re-use the already-loaded coordinates indirectly via a lightweight read.
    try:
        n_residues = len(extract_ca_coords(backbone_path))
    except Exception as exc:
        logger.warning(
            "Failed to determine binder length for design '%s': %s", design_id, exc
        )
        n_residues = None

    rmsd_fail = rmsd > config.rmsd_threshold
    plddt_fail = plddt < config.plddt_threshold

    if n_residues is not None:
        length_fail = not (config.min_binder_length <= n_residues <= config.max_binder_length)
    else:
        length_fail = False

    if length_fail:
        if rmsd_fail and plddt_fail:
            fail_reason = "both"
        elif rmsd_fail:
            fail_reason = "rmsd"
        elif plddt_fail:
            fail_reason = "plddt"
        else:
            fail_reason = "length"
    elif rmsd_fail and plddt_fail:
        fail_reason = "both"
    elif rmsd_fail:
        fail_reason = "rmsd"
    elif plddt_fail:
        fail_reason = "plddt"
    else:
        fail_reason = ""

    passes = fail_reason == ""

    return FilterResult(
        design_id=design_id,
        backbone_path=backbone_path,
        folded_path=folded_path,
        rmsd=rmsd,
        plddt=plddt,
        passes=passes,
        fail_reason=fail_reason,
    )


def run_filter(
    design_map: Dict[str, Tuple[Path, Path]],
    config: FilterConfig,
) -> List[FilterResult]:
    """Filter a collection of designs by self-consistency metrics.

    Parameters
    ----------
    design_map : dict[str, tuple[Path, Path]]
        Mapping of design_id -> (backbone_path, folded_path).
    config : FilterConfig
        Thresholds to apply.

    Returns
    -------
    list[FilterResult]
        All results (passing and failing).  Callers may filter further via the
        ``passes`` field.
    """
    results: List[FilterResult] = []

    for design_id, (backbone_path, folded_path) in design_map.items():
        result = filter_design(design_id, backbone_path, folded_path, config)
        results.append(result)

    n_total = len(results)
    n_pass = sum(r.passes for r in results)
    logger.info(
        "Filter complete: %d / %d designs pass (%.1f%%)",
        n_pass,
        n_total,
        100.0 * n_pass / n_total if n_total > 0 else 0.0,
    )

    return results


def results_to_dataframe(results: List[FilterResult]) -> pd.DataFrame:
    """Convert a list of FilterResult objects to a pandas DataFrame.

    Parameters
    ----------
    results : list[FilterResult]
        Output from :func:`run_filter`.

    Returns
    -------
    pd.DataFrame
        One row per design with all FilterResult fields as columns.
        The ``passes`` column is boolean; ``backbone_path`` and
        ``folded_path`` are cast to strings for serialisation convenience.
    """
    rows = [
        {
            "design_id": r.design_id,
            "backbone_path": str(r.backbone_path),
            "folded_path": str(r.folded_path),
            "rmsd": r.rmsd,
            "plddt": r.plddt,
            "passes": r.passes,
            "fail_reason": r.fail_reason,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    return df
