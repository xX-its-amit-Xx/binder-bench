"""Backbone generation toward a target epitope."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from Bio.PDB import PDBParser, Structure

from binder_bench.config import GenerateConfig

logger = logging.getLogger(__name__)


class BackboneGenerator(ABC):
    """Abstract base class for backbone generators."""

    @abstractmethod
    def generate(self, config: GenerateConfig) -> List[Path]:
        """Generate backbone PDB files toward a target epitope.

        Parameters
        ----------
        config:
            Generation configuration.

        Returns
        -------
        List[Path]
            Paths to generated PDB files.
        """


class StubGenerator(BackboneGenerator):
    """Default/CI generator that loads pre-computed PDB files from a directory.

    Useful for testing and continuous integration environments where
    RFdiffusion is not available.
    """

    def generate(self, config: GenerateConfig) -> List[Path]:
        """Load pre-computed PDB files from config.stub_dir.

        Parameters
        ----------
        config:
            Generation configuration. Uses ``stub_dir`` and ``n_designs``.

        Returns
        -------
        List[Path]
            Up to ``config.n_designs`` PDB file paths found in ``config.stub_dir``.

        Raises
        ------
        FileNotFoundError
            If ``config.stub_dir`` does not exist.
        """
        stub_dir = Path(config.stub_dir)
        if not stub_dir.exists():
            raise FileNotFoundError(
                f"Stub directory not found: {stub_dir}. "
                "Please create the directory and populate it with pre-computed "
                "*.pdb files, or set backend='rfdiffusion' to run inference."
            )

        pdb_files = sorted(stub_dir.glob("*.pdb"))
        selected = pdb_files[: config.n_designs]
        logger.debug(
            "StubGenerator: found %d PDB files in %s, returning %d",
            len(pdb_files),
            stub_dir,
            len(selected),
        )
        return selected


class RFDiffusionGenerator(BackboneGenerator):
    """Backbone generator backed by RFdiffusion.

    Requires the ``rfdiffusion`` package to be installed.
    See: https://github.com/RosettaCommons/RFdiffusion
    """

    def generate(self, config: GenerateConfig) -> List[Path]:
        """Run RFdiffusion inference to generate binder backbones.

        Parameters
        ----------
        config:
            Generation configuration.

        Returns
        -------
        List[Path]
            Paths to generated PDB files.

        Raises
        ------
        ImportError
            If the ``rfdiffusion`` package is not installed.
        """
        try:
            import rfdiffusion  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "The 'rfdiffusion' package is required to use RFDiffusionGenerator. "
                "Please install it by following the instructions at: "
                "https://github.com/RosettaCommons/RFdiffusion"
            ) from exc

        # TODO: Implement RFdiffusion inference call.
        # Reference: https://github.com/RosettaCommons/RFdiffusion
        # Expected steps:
        #   1. Build hydra/OmegaConf config from GenerateConfig fields
        #      (target_pdb, hotspot_residues, n_designs, rfdiffusion_weights, seed).
        #   2. Call rfdiffusion.inference.utils.sampler or equivalent entry point.
        #   3. Collect output PDB paths written by RFdiffusion and return them.
        raise NotImplementedError(
            "RFDiffusionGenerator.generate() is not yet fully implemented. "
            "See TODO comments in binder_bench/generate.py for guidance."
        )


def get_generator(config: GenerateConfig) -> BackboneGenerator:
    """Factory function returning the appropriate BackboneGenerator.

    Parameters
    ----------
    config:
        Generation configuration. Uses ``config.backend`` to select the
        generator (``'stub'`` or ``'rfdiffusion'``).

    Returns
    -------
    BackboneGenerator
        Instantiated generator for the requested backend.
    """
    if config.backend == "stub":
        return StubGenerator()
    if config.backend == "rfdiffusion":
        return RFDiffusionGenerator()
    raise ValueError(f"Unknown backbone generation backend: {config.backend!r}")


def run_generate(config: GenerateConfig) -> List[Path]:
    """Generate backbone PDB files and log the number of results.

    Parameters
    ----------
    config:
        Generation configuration.

    Returns
    -------
    List[Path]
        Paths to generated backbone PDB files.
    """
    generator = get_generator(config)
    results = generator.generate(config)
    logger.info(
        "run_generate: produced %d backbone(s) using backend=%r",
        len(results),
        config.backend,
    )
    return results
