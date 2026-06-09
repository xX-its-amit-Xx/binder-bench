from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import requests

from binder_bench.config import FoldConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class StructurePredictor(ABC):
    """Abstract base class for sequence-to-structure predictors."""

    @abstractmethod
    def fold(self, sequence: str, output_path: Path, config: FoldConfig) -> Path:
        """Fold *sequence* and write the predicted PDB to *output_path*.

        Returns the path to the written PDB file.
        """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seq_hash(sequence: str) -> str:
    """Return the first 8 hex characters of the MD5 of *sequence*."""
    return hashlib.md5(sequence.encode()).hexdigest()[:8]


def _make_stub_pdb(sequence: str) -> str:
    """Generate a minimal valid PDB with alpha-helix CA-only geometry.

    Residue i is placed at:
        x = 1.5 * cos(i * 100 deg)
        y = 1.5 * sin(i * 100 deg)
        z = 1.5 * i

    All residues are represented as ALA with B-factor 75.0 (stub pLDDT).
    """
    deg_to_rad = math.pi / 180.0
    angle_step = 100.0 * deg_to_rad

    lines: list[str] = []
    for i, _aa in enumerate(sequence):
        atom_serial = i + 1
        res_seq = i + 1
        x = 1.5 * math.cos(i * angle_step)
        y = 1.5 * math.sin(i * angle_step)
        z = 1.5 * i
        # PDB ATOM record format (columns 1-80):
        # 1-6   record name "ATOM  "
        # 7-11  serial (right-justified)
        # 12    blank
        # 13-16 atom name " CA "
        # 17    alt loc ' '
        # 18-20 residue name "ALA"
        # 21    blank
        # 22    chain "A"
        # 23-26 res seq (right-justified)
        # 27    iCode ' '
        # 28-30 blanks
        # 31-38 x (8.3f)
        # 39-46 y (8.3f)
        # 47-54 z (8.3f)
        # 55-60 occupancy (6.2f)
        # 61-66 tempFactor (6.2f)
        # 77-78 element
        line = (
            f"ATOM  {atom_serial:5d}  CA  ALA A{res_seq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 75.00           C  "
        )
        lines.append(line)

    lines.append("TER")
    lines.append("END")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# StubPredictor
# ---------------------------------------------------------------------------

class StubPredictor(StructurePredictor):
    """Return a pre-computed structure from stub_dir, or generate a minimal PDB.

    The look-up key is the first 8 characters of the MD5 hex digest of the
    sequence.  The predictor searches for any file whose stem starts with that
    hash prefix inside *config.stub_dir*.  If nothing is found, a minimal
    alpha-helix CA-only PDB is written to *output_path*.
    """

    def fold(self, sequence: str, output_path: Path, config: FoldConfig) -> Path:
        seq_id = _seq_hash(sequence)
        stub_dir: Path = config.stub_dir

        # Search for a precomputed structure
        if stub_dir.is_dir():
            for candidate in stub_dir.iterdir():
                if candidate.stem.startswith(seq_id) and candidate.suffix in {
                    ".pdb",
                    ".ent",
                }:
                    logger.debug(
                        "StubPredictor: using precomputed structure %s for hash %s",
                        candidate,
                        seq_id,
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(candidate.read_bytes())
                    return output_path

        # No precomputed structure found — generate a minimal placeholder
        logger.debug(
            "StubPredictor: no precomputed structure for hash %s; generating placeholder PDB",
            seq_id,
        )
        pdb_content = _make_stub_pdb(sequence)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(pdb_content)
        return output_path


# ---------------------------------------------------------------------------
# ESMFoldAPIPredictor
# ---------------------------------------------------------------------------

class ESMFoldAPIPredictor(StructurePredictor):
    """Fold a sequence via the ESMFold Atlas REST API.

    POSTs the sequence to *config.esmfold_api_url* and saves the PDB text
    response to *output_path*.  Retries up to 3 times with exponential
    back-off on transient failures.
    """

    _MAX_RETRIES: int = 3
    _BASE_BACKOFF: float = 2.0  # seconds

    def fold(self, sequence: str, output_path: Path, config: FoldConfig) -> Path:
        url: str = config.esmfold_api_url
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        last_exc: Optional[Exception] = None
        for attempt in range(self._MAX_RETRIES):
            try:
                logger.debug(
                    "ESMFoldAPIPredictor: attempt %d/%d for sequence of length %d",
                    attempt + 1,
                    self._MAX_RETRIES,
                    len(sequence),
                )
                response = requests.post(
                    url,
                    data=sequence,
                    headers=headers,
                    timeout=120,
                )
                response.raise_for_status()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(response.text)
                logger.info(
                    "ESMFoldAPIPredictor: wrote PDB to %s (%d bytes)",
                    output_path,
                    len(response.text),
                )
                return output_path
            except requests.RequestException as exc:
                last_exc = exc
                wait = self._BASE_BACKOFF ** attempt
                logger.warning(
                    "ESMFoldAPIPredictor: request failed (attempt %d/%d): %s. "
                    "Retrying in %.1fs …",
                    attempt + 1,
                    self._MAX_RETRIES,
                    exc,
                    wait,
                )
                if attempt < self._MAX_RETRIES - 1:
                    time.sleep(wait)

        raise RuntimeError(
            f"ESMFoldAPIPredictor: all {self._MAX_RETRIES} attempts failed. "
            f"Last error: {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_predictor(config: FoldConfig) -> StructurePredictor:
    """Return the appropriate :class:`StructurePredictor` for *config*."""
    if config.backend == "stub":
        return StubPredictor()
    if config.backend == "esmfold_api":
        return ESMFoldAPIPredictor()
    raise ValueError(f"Unknown fold backend: {config.backend!r}")


# ---------------------------------------------------------------------------
# Module-level runner
# ---------------------------------------------------------------------------

def run_fold(
    sequences: dict[Path, list[str]],
    config: FoldConfig,
    output_dir: Path,
) -> dict[str, Path]:
    """Fold all designed sequences and return a mapping of design_id -> PDB path.

    Parameters
    ----------
    sequences:
        Mapping of backbone Path -> list of designed sequences for that backbone.
    config:
        Folding configuration.
    output_dir:
        Root directory under which per-design PDB files are written.

    Returns
    -------
    dict[str, Path]
        ``design_id -> folded_pdb_path`` where design_id has the form
        ``"{backbone_stem}_seq{idx:03d}"``.
    """
    predictor = get_predictor(config)
    results: dict[str, Path] = {}

    for backbone_path, seqs in sequences.items():
        backbone_stem = backbone_path.stem
        for idx, seq in enumerate(seqs):
            design_id = f"{backbone_stem}_seq{idx:03d}"
            pdb_filename = f"{design_id}.pdb"
            output_path = output_dir / pdb_filename

            logger.info("Folding design %s (seq length %d) …", design_id, len(seq))
            try:
                folded_path = predictor.fold(seq, output_path, config)
                results[design_id] = folded_path
                logger.info("Folded %s -> %s", design_id, folded_path)
            except Exception as exc:
                logger.error("Failed to fold %s: %s", design_id, exc)
                raise

    return results
