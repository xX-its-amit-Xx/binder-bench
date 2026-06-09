"""Shared pytest fixtures for binder-bench tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


def _write_ca_pdb(path: Path, n_atoms: int, bfactor: float = 75.0, chain: str = "A") -> Path:
    """Write a minimal CA-only PDB with atoms on the x-axis."""
    lines = []
    for i in range(n_atoms):
        x = float(i)
        y = 0.0
        z = 0.0
        line = (
            f"ATOM  {i+1:5d}  CA  ALA {chain}{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           C  "
        )
        lines.append(line)
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


def _write_helix_pdb(path: Path, n_atoms: int, bfactor: float = 75.0) -> Path:
    """Write a minimal CA-only PDB with atoms in helix geometry."""
    deg = math.pi / 180.0
    lines = []
    for i in range(n_atoms):
        angle = i * 100.0 * deg
        x = 2.3 * math.cos(angle)
        y = 2.3 * math.sin(angle)
        z = 1.5 * i
        line = (
            f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           C  "
        )
        lines.append(line)
    lines.append("TER")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture(scope="session")
def make_pdb_file(tmp_path_factory):
    """Session-scoped factory: make_pdb_file(subdir, name, n, bfactor) -> Path."""
    base = tmp_path_factory.mktemp("pdbs")

    def _factory(name: str, n: int = 10, bfactor: float = 75.0) -> Path:
        path = base / name
        return _write_ca_pdb(path, n, bfactor)

    return _factory


@pytest.fixture
def sample_backbone_pdb(tmp_path) -> Path:
    """70-residue CA-only backbone PDB (bfactor=75.0)."""
    return _write_helix_pdb(tmp_path / "backbone.pdb", 70, bfactor=75.0)


@pytest.fixture
def sample_folded_pdb(tmp_path) -> Path:
    """70-residue CA-only folded PDB (bfactor=80.0 simulating pLDDT)."""
    return _write_helix_pdb(tmp_path / "folded.pdb", 70, bfactor=80.0)
