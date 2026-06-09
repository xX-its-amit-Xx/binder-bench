"""Structural metrics for binder benchmarking."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import numpy as np


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------

def kabsch_rmsd(P: np.ndarray, Q: np.ndarray) -> float:
    """Compute RMSD between P and Q after optimal Kabsch superposition.

    Parameters
    ----------
    P : np.ndarray, shape (N, 3)
        Mobile point set (float64).
    Q : np.ndarray, shape (N, 3)
        Reference point set (float64).

    Returns
    -------
    float
        RMSD after optimal rotation and translation.

    Raises
    ------
    ValueError
        If shapes differ or N < 3.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    if P.shape != Q.shape:
        raise ValueError(
            f"Shape mismatch: P has shape {P.shape}, Q has shape {Q.shape}."
        )
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("P and Q must be (N, 3) arrays.")
    N = P.shape[0]
    if N < 3:
        raise ValueError(f"At least 3 atoms required for Kabsch alignment; got {N}.")

    # Centre
    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    # Covariance matrix H = P^T Q
    H = P_c.T @ Q_c

    # SVD
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Correct for reflection: if det(V @ U^T) < 0, flip last column of V
    d = np.linalg.det(V @ U.T)
    if d < 0.0:
        V[:, -1] *= -1.0

    # Optimal rotation
    R = V @ U.T

    # Rotate P_c
    P_rot = P_c @ R.T

    # RMSD
    diff = P_rot - Q_c
    rmsd = float(np.sqrt((diff ** 2).sum() / N))
    return rmsd


# ---------------------------------------------------------------------------
# Align and return coordinates
# ---------------------------------------------------------------------------

def align_structures(
    mobile_coords: np.ndarray,
    ref_coords: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Align mobile onto ref using Kabsch and return RMSD + aligned coords.

    Parameters
    ----------
    mobile_coords : np.ndarray, shape (N, 3)
        Coordinates of the mobile structure.
    ref_coords : np.ndarray, shape (N, 3)
        Coordinates of the reference structure.

    Returns
    -------
    tuple[float, np.ndarray]
        (rmsd, aligned_mobile_coords) where aligned_mobile_coords has shape (N, 3).
    """
    P = np.asarray(mobile_coords, dtype=np.float64)
    Q = np.asarray(ref_coords, dtype=np.float64)

    if P.shape != Q.shape:
        raise ValueError(
            f"Shape mismatch: mobile {P.shape} vs ref {Q.shape}."
        )
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("Coordinate arrays must be (N, 3).")
    N = P.shape[0]
    if N < 3:
        raise ValueError(f"At least 3 atoms required; got {N}.")

    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)

    P_c = P - P_centroid
    Q_c = Q - Q_centroid

    H = P_c.T @ Q_c
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    d = np.linalg.det(V @ U.T)
    if d < 0.0:
        V[:, -1] *= -1.0

    R = V @ U.T

    # Apply rotation then translate to ref centroid
    aligned = (P_c @ R.T) + Q_centroid

    diff = (P_c @ R.T) - Q_c
    rmsd = float(np.sqrt((diff ** 2).sum() / N))

    return rmsd, aligned


# ---------------------------------------------------------------------------
# Structure I/O helpers
# ---------------------------------------------------------------------------

def _get_structure(structure_path: Union[Path, str]):
    """Load a Biopython Structure object from a PDB file, suppressing warnings."""
    from Bio.PDB import PDBParser  # type: ignore

    structure_path = Path(structure_path)
    parser = PDBParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        structure = parser.get_structure(structure_path.stem, str(structure_path))
    return structure


def _pick_chain(model):
    """Return chain A if present, else the first chain in the model."""
    if "A" in model:
        return model["A"]
    return next(iter(model.get_chains()))


def extract_ca_coords(structure_path: Union[Path, str]) -> np.ndarray:
    """Extract C-alpha coordinates from a PDB file.

    Uses the first model; prefers chain A, falls back to the first chain.

    Parameters
    ----------
    structure_path : Path or str
        Path to the PDB file.

    Returns
    -------
    np.ndarray, shape (N, 3)
        C-alpha coordinates in Angstroms.
    """
    structure = _get_structure(structure_path)
    model = next(iter(structure))
    chain = _pick_chain(model)

    coords = []
    for residue in chain.get_residues():
        if "CA" in residue:
            coords.append(residue["CA"].get_vector().get_array())

    if not coords:
        raise ValueError(
            f"No CA atoms found in {structure_path} (chain {chain.id})."
        )
    return np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# pLDDT from B-factor column
# ---------------------------------------------------------------------------

def compute_plddt(structure_path: Union[Path, str]) -> float:
    """Compute mean pLDDT score from B-factor column of CA atoms.

    ESMFold and AlphaFold2 store pLDDT (0–100) in the B-factor field.

    Parameters
    ----------
    structure_path : Path or str
        Path to the PDB file produced by ESMFold or AF2.

    Returns
    -------
    float
        Mean pLDDT across all CA atoms in the structure.
    """
    structure = _get_structure(structure_path)

    bfactors = []
    for model in structure:
        for chain in model:
            for residue in chain.get_residues():
                if "CA" in residue:
                    bfactors.append(residue["CA"].get_bfactor())

    if not bfactors:
        raise ValueError(f"No CA atoms found in {structure_path}.")

    return float(np.mean(bfactors))


# ---------------------------------------------------------------------------
# Interface contacts
# ---------------------------------------------------------------------------

def compute_interface_contacts(
    binder_coords: np.ndarray,
    target_coords: np.ndarray,
    distance_threshold: float = 8.0,
) -> int:
    """Count binder residues within distance_threshold of any target residue.

    Parameters
    ----------
    binder_coords : np.ndarray, shape (Nb, 3)
        C-alpha (or any representative) coordinates of binder residues.
    target_coords : np.ndarray, shape (Nt, 3)
        C-alpha (or any representative) coordinates of target residues.
    distance_threshold : float, optional
        Distance cutoff in Angstroms (default 8.0).

    Returns
    -------
    int
        Number of unique binder residues with at least one target neighbour
        within the cutoff.
    """
    binder_coords = np.asarray(binder_coords, dtype=np.float64)
    target_coords = np.asarray(target_coords, dtype=np.float64)

    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(target_coords)
        # For each binder residue, check if any target residue is within threshold
        counts = tree.query_ball_point(binder_coords, r=distance_threshold)
        contact_mask = np.array([len(c) > 0 for c in counts], dtype=bool)
    except ImportError:
        # Pure numpy fallback via broadcasting
        # Shape: (Nb, Nt)
        diff = binder_coords[:, np.newaxis, :] - target_coords[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=-1))
        contact_mask = dists.min(axis=1) < distance_threshold

    return int(contact_mask.sum())


# ---------------------------------------------------------------------------
# Self-consistency RMSD
# ---------------------------------------------------------------------------

def self_consistency_rmsd(
    designed_bb_path: Union[Path, str],
    folded_path: Union[Path, str],
) -> float:
    """Compute CA RMSD between a designed backbone and its folded prediction.

    Trims to the shorter of the two sequences when lengths differ (common
    when the folded structure carries extra terminal residues).

    Parameters
    ----------
    designed_bb_path : Path or str
        Path to the designed backbone PDB (e.g. ProteinMPNN/RFdiffusion output).
    folded_path : Path or str
        Path to the folded structure PDB (e.g. ESMFold prediction).

    Returns
    -------
    float
        CA RMSD after Kabsch superposition on the trimmed coordinates.
    """
    designed_ca = extract_ca_coords(designed_bb_path)
    folded_ca = extract_ca_coords(folded_path)

    min_len = min(len(designed_ca), len(folded_ca))
    if min_len < 3:
        raise ValueError(
            f"Cannot compute RMSD: only {min_len} common CA atoms available."
        )

    rmsd, _ = align_structures(designed_ca[:min_len], folded_ca[:min_len])
    return rmsd
