"""Sequence design for each backbone using ProteinMPNN or a stub."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from binder_bench.config import DesignConfig

logger = logging.getLogger(__name__)

_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class DesignResult:
    backbone_path: Path
    sequence_index: int
    sequence: str
    length: int


class SequenceDesigner(ABC):
    """Abstract base class for sequence designers."""

    @abstractmethod
    def design(
        self, backbone_paths: list[Path], config: DesignConfig
    ) -> dict[Path, list[str]]:
        """Design sequences for each backbone.

        Parameters
        ----------
        backbone_paths:
            List of backbone PDB file paths.
        config:
            Design configuration.

        Returns
        -------
        dict[Path, list[str]]
            Mapping of backbone_path -> list of designed sequences
            (one-letter FASTA strings).
        """


class StubDesigner(SequenceDesigner):
    """CI-friendly designer that reads sequences from config.stub_dir.

    Looks for files named ``{backbone_stem}_seqs.txt`` (one sequence per line).
    If a file is missing, generates deterministic placeholder sequences seeded
    from the backbone filename so results are reproducible.
    """

    def design(
        self, backbone_paths: list[Path], config: DesignConfig
    ) -> dict[Path, list[str]]:
        rng = np.random.default_rng(config.seed)
        stub_dir = Path(config.stub_dir)
        result: dict[Path, list[str]] = {}

        for backbone_path in backbone_paths:
            seq_file = stub_dir / f"{backbone_path.stem}_seqs.txt"

            if seq_file.exists():
                sequences = [
                    line.strip()
                    for line in seq_file.read_text().splitlines()
                    if line.strip()
                ]
                sequences = sequences[: config.n_seqs_per_backbone]
                logger.debug(
                    "StubDesigner: loaded %d sequences from %s", len(sequences), seq_file
                )
            else:
                # Deterministic fallback: use backbone name as part of the seed
                name_seed = sum(ord(c) for c in backbone_path.stem)
                local_rng = np.random.default_rng(config.seed + name_seed)
                # Derive a length from the backbone if possible, else use 65
                try:
                    from binder_bench.metrics import extract_ca_coords  # noqa: PLC0415

                    n_residues = len(extract_ca_coords(backbone_path))
                    n_residues = max(config.filter_min_length if hasattr(config, "filter_min_length") else 50, n_residues)
                except Exception:
                    n_residues = 65

                sequences = [
                    "".join(
                        local_rng.choice(list(_AA_ALPHABET), size=n_residues)
                    )
                    for _ in range(config.n_seqs_per_backbone)
                ]
                logger.debug(
                    "StubDesigner: generated %d stub sequences for %s (no seqs file found)",
                    len(sequences),
                    backbone_path.stem,
                )

            result[backbone_path] = sequences

        return result


class ProteinMPNNDesigner(SequenceDesigner):
    """Sequence designer backed by ProteinMPNN.

    Requires either the ``protein_mpnn`` package to be importable, or the
    ``PROTEINMPNN_PATH`` environment variable to point to a checkout of the
    ProteinMPNN repository (github.com/dauparas/ProteinMPNN).
    """

    def design(
        self, backbone_paths: list[Path], config: DesignConfig
    ) -> dict[Path, list[str]]:
        # Check availability
        mpnn_path = os.environ.get("PROTEINMPNN_PATH")
        try:
            import protein_mpnn  # noqa: F401  # type: ignore

            _available = True
        except ImportError:
            _available = mpnn_path is not None

        if not _available:
            raise ImportError(
                "ProteinMPNN is not available. Either:\n"
                "  1. Install it: pip install git+https://github.com/dauparas/ProteinMPNN.git\n"
                "  2. Set PROTEINMPNN_PATH to a local checkout of the ProteinMPNN repository.\n"
                "Or use backend='stub' for CPU-only / CI runs."
            )

        # TODO: Implement ProteinMPNN inference.
        # Reference: https://github.com/dauparas/ProteinMPNN
        # Expected steps:
        #   1. For each backbone PDB, parse chain definitions.
        #   2. Call ProteinMPNN protein_mpnn_run.py via subprocess or Python API
        #      with --num_seq_per_target=config.n_seqs_per_backbone,
        #      --sampling_temp=str(config.temperature),
        #      --seed=str(config.seed).
        #   3. Parse output FASTA files and return sequences.
        raise NotImplementedError(
            "ProteinMPNNDesigner.design() is not yet fully implemented. "
            "See TODO comments in binder_bench/design.py for guidance."
        )


def get_designer(config: DesignConfig) -> SequenceDesigner:
    """Factory returning the appropriate SequenceDesigner.

    Parameters
    ----------
    config:
        Design configuration. Uses ``config.backend``.

    Returns
    -------
    SequenceDesigner
        Instantiated designer for the requested backend.
    """
    if config.backend == "stub":
        return StubDesigner()
    if config.backend == "proteinmpnn":
        return ProteinMPNNDesigner()
    raise ValueError(f"Unknown sequence design backend: {config.backend!r}")


def run_design(
    backbone_paths: list[Path], config: DesignConfig
) -> dict[Path, list[str]]:
    """Design sequences for all backbones and log results.

    Parameters
    ----------
    backbone_paths:
        List of backbone PDB file paths.
    config:
        Design configuration.

    Returns
    -------
    dict[Path, list[str]]
        backbone_path -> list of designed sequences.
    """
    designer = get_designer(config)
    sequences = designer.design(backbone_paths, config)
    n_total = sum(len(v) for v in sequences.values())
    logger.info(
        "run_design: produced %d sequences across %d backbones using backend=%r",
        n_total,
        len(sequences),
        config.backend,
    )
    return sequences
