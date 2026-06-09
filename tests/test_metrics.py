"""Tests for binder_bench.metrics — Kabsch RMSD and related utilities."""

from __future__ import annotations

import sys
import types
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Inject a stub binder_bench.design module before any package import so that
# ``from binder_bench.design import run_design`` in pipeline.py does not raise
# ImportError when __init__.py is loaded as a side-effect of importing metrics.
# ---------------------------------------------------------------------------
_stub_design = types.ModuleType("binder_bench.design")
_stub_design.run_design = lambda *a, **kw: {}  # type: ignore[attr-defined]
sys.modules.setdefault("binder_bench.design", _stub_design)

from binder_bench.metrics import (
    kabsch_rmsd,
    align_structures,
    extract_ca_coords,
    compute_plddt,
    self_consistency_rmsd,
    compute_interface_contacts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_pdb(path: Path, coords: np.ndarray, bfactor: float = 75.0) -> None:
    """Write a minimal PDB file with CA atoms at the given coordinates."""
    lines = []
    for i, (x, y, z) in enumerate(coords, start=1):
        # ATOM record: cols are fixed-width per PDB spec
        lines.append(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           C  \n"
        )
    lines.append("END\n")
    path.write_text("".join(lines))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def make_pdb(tmp_path):
    """Factory fixture: make_pdb(n_atoms, bfactor=75.0) -> Path.

    Creates a minimal PDB with n_atoms CA atoms placed on a line:
    atom i has coordinates (i, 0, 0) for i = 0 .. n_atoms-1.
    """
    def _factory(n_atoms: int, bfactor: float = 75.0) -> Path:
        coords = np.array([[float(i), 0.0, 0.0] for i in range(n_atoms)])
        pdb_path = tmp_path / f"test_{n_atoms}atoms.pdb"
        _write_pdb(pdb_path, coords, bfactor=bfactor)
        return pdb_path

    return _factory


# ---------------------------------------------------------------------------
# kabsch_rmsd — numerical tests
# ---------------------------------------------------------------------------

class TestKabschRmsd:
    """Tests for the kabsch_rmsd() function."""

    def test_identical_structures(self):
        """Identical point sets must give RMSD = 0."""
        rng = np.random.default_rng(0)
        P = rng.random((10, 3)) * 20.0
        rmsd = kabsch_rmsd(P, P.copy())
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_pure_translation(self):
        """A translated copy must superpose perfectly -> RMSD = 0."""
        rng = np.random.default_rng(1)
        P = rng.random((8, 3)) * 10.0
        translation = np.array([5.0, -3.0, 2.5])
        Q = P + translation
        rmsd = kabsch_rmsd(P, Q)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_pure_rotation_90_deg_z(self):
        """90° rotation around Z of a square in the XY plane -> RMSD = 0."""
        # Simple regular polygon in XY plane
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        P = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(8)])

        # 90° rotation matrix around Z
        theta = np.pi / 2
        Rz = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0],
        ])
        Q = (Rz @ P.T).T  # rotated copy

        rmsd = kabsch_rmsd(P, Q)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_uniform_displacement_one_angstrom(self):
        """Structures that differ by exactly 1 Å per atom -> RMSD = 1.0."""
        # Place atoms such that after centering and optimal rotation the residual
        # is purely a uniform 1 Å shift along X that cannot be explained by rotation.
        # We achieve this by choosing P and Q = P + (1,0,0) and ensuring P is
        # centrosymmetric so the centroid shift is the only difference.
        P = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ], dtype=np.float64)
        # Shift every atom by (1, 0, 0) — centroid moves by (1,0,0) but
        # after re-centering P_c == Q_c, so RMSD = 0.
        # Instead, use an asymmetric displacement.
        #
        # A cleaner construction: P and Q are centred at origin and differ
        # by a pure per-atom offset of constant magnitude 1 Å orthogonal to
        # any rotation axis.  The simplest case: collinear atoms where the
        # *centred* coordinates are identical but each atom in P is shifted
        # by (1,0,0) relative to Q before centring — i.e. use non-centrosymmetric set.
        #
        # Easiest exact construction:
        # Let P = [[0,0,0],[2,0,0],[1,1,0]] and Q = [[1,0,0],[3,0,0],[2,1,0]].
        # They are related by translation (1,0,0), so after Kabsch RMSD = 0.
        # That does NOT give RMSD = 1.
        #
        # Correct approach: pick P; define Q such that after optimal alignment
        # each aligned atom differs from Q by exactly 1 Å.
        # Easiest: P_c and Q_c are already centred and differ by per-atom
        # offsets of magnitude 1 in orthogonal directions -> RMSD = 1.
        # Use P_c = Q_c + delta where ||delta_i|| = 1 for all i and
        # the optimal rotation maps P_c to Q_c (i.e. R = I).
        # That means H = P_c^T Q_c has positive determinant SVD, R = I,
        # which holds when P_c and Q_c are close.
        # Simplest: for each atom add (1,0,0) to Q_c AFTER centering.
        # Then RMSD = 1 exactly, but only if the optimal rotation is I
        # (it will be I since H = P_c^T (P_c + ones*(1,0,0)) is dominated by P_c^T P_c).
        # But this is only guaranteed for specific shapes.
        #
        # Clean exact solution:
        # Use atoms on a 3-D grid so that H remains positive definite.
        N = 4
        # Corners of a tetrahedron (centred)
        tet = np.array([
            [ 1.0,  1.0,  1.0],
            [ 1.0, -1.0, -1.0],
            [-1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0],
        ], dtype=np.float64)
        # tet is already centred; construct Q = tet + displacement d per atom
        # where d is the same for all atoms (constant shift after centering).
        # After centering P_c = tet, Q_c = tet (same centroid), so RMSD is
        # determined by rotation.  If R = I then RMSD = ||d|| = 1 only if d != 0.
        # But centering removes the mean shift.
        #
        # Real solution: use different per-atom displacements summing to zero,
        # each of magnitude 1.
        # d_i in orthogonal directions summing to 0:
        # d = [e1, -e1, e2, -e2] with e1=(1,0,0), e2=(0,1,0)
        d = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float64)
        Q2 = tet + d  # Q2 centroid = tet centroid (d sums to zero)
        # For kabsch to give R=I the cross-covariance H = tet^T (tet+d) = tet^T tet + tet^T d
        # tet^T d mixes orthogonal directions; R may not be exactly I.
        # Verify analytically: RMSD after R = optimal.
        # Instead, directly assert with the computed value.
        # The test intent is RMSD = 1.0 exactly; let's construct it that way.
        #
        # FINAL clean approach: two centred identical structures rotated by an
        # angle such that the Kabsch-optimal alignment cannot reduce RMSD below 1.
        # Actually simplest: define P and Q to already be in their optimal alignment
        # (identical centroid, R=I is optimal) with per-atom diff = 1 Å.
        # This is guaranteed when H = P_c^T Q_c has all positive singular values
        # and the perturbation d is orthogonal to the null space of H.
        # Robust construction: use a large, well-conditioned P and add a *small*
        # perturbation that survives after optimal rotation.
        # But we need RMSD = exactly 1.0.
        #
        # Clearest construction (avoids all the above subtlety):
        # Use a set of N atoms where P and Q differ by a *rotation* of exactly
        # arcsin(1 / radius) so that the chord length = 1 Å, giving RMSD = 1.
        # ... still not exactly 1.
        #
        # SIMPLEST provably-exact construction:
        # P and Q are collinear along X, so optimal rotation = I (or reflections
        # don't help for collinear sets in 3D). After centring:
        # P_c_i = i - mean, Q_c_i = P_c_i + (alternating +-1 in Y) summing to 0.
        # RMSD = sqrt(mean(1^2)) = 1.0.
        N2 = 6  # even number
        xs = np.arange(N2, dtype=np.float64)  # 0,1,2,3,4,5
        P2 = np.column_stack([xs, np.zeros(N2), np.zeros(N2)])
        dy = np.array([1.0, -1.0] * (N2 // 2))  # alternating, sum=0
        Q3 = np.column_stack([xs, dy, np.zeros(N2)])
        # Both centred at same x-centroid; after centering P_c_i=(xi-mean,0,0),
        # Q_c_i=(xi-mean, dy_i, 0).  Optimal rotation:
        # H = P_c^T Q_c; P_c has only x-components, Q_c has x and y.
        # H[0,0] = sum(P_c_x * Q_c_x) = sum(P_c_x^2) > 0
        # H[0,1] = sum(P_c_x * dy_i) -- could be nonzero.
        # sum(P_c_x * dy_i) = sum((i - mean)*(-1)^i)
        # For i=0..5, mean=2.5: offsets=(-2.5,-1.5,-0.5,0.5,1.5,2.5), dy=(1,-1,1,-1,1,-1)
        # product: -2.5,-1.5*-1=-1.5... wait: (-2.5)(1)+(-1.5)(-1)+(-0.5)(1)+(0.5)(-1)+(1.5)(1)+(2.5)(-1)
        #         = -2.5 + 1.5 - 0.5 - 0.5 + 1.5 - 2.5 = -3.0 != 0
        # So R != I and RMSD != 1.
        #
        # TRULY SIMPLE approach: place atoms on Y-axis so that X and Y are decoupled.
        # P_c: atoms at (0, -2, 0),(0,-1,0),(0,0,0),(0,1,0),(0,2,0)  [5 atoms, centred]
        # Q_c: same Y coords but +1 in X for each atom.  Sum of X offsets = 5 != 0. Not centred.
        # Add equal but opposite to keep centred: can't with same offset.
        #
        # OK, FINAL ANSWER using the well-known result:
        # RMSD² = (1/N) * sum ||P_rot_i - Q_i||²
        # If we want RMSD = 1 and choose P = Q + noise where noise is centred and
        # orthogonal to any rotation (so R = I is optimal), then:
        # sum ||noise_i||² = N  =>  RMSD = 1.
        # Noise orthogonal to any rotation = noise in Z when structures lie in XY.
        # P in XY plane; noise in Z direction, summing to zero, each |noise_i| = 1.
        # For an even number of atoms: half +1, half -1 in Z (sum=0).
        N3 = 6
        angles2 = np.linspace(0, 2 * np.pi, N3, endpoint=False)
        radius = 5.0
        P3 = np.column_stack([radius * np.cos(angles2), radius * np.sin(angles2), np.zeros(N3)])
        dz = np.array([1.0, -1.0] * (N3 // 2))  # sums to 0, each magnitude 1
        Q4 = P3.copy()
        Q4[:, 2] = dz
        # Now P3_c = P3 (already centred, z=0), Q4_c = Q4 (centred, z=dz).
        # H = P3^T Q4; Q4 has z=dz so H[0,2] = sum(P3_x * dz_i),
        # H[1,2] = sum(P3_y * dz_i), H[2,:] = 0 (P3_z=0).
        # These off-diagonal terms may cause R != I.
        # sum(cos(k*2pi/6)*(-1)^k) for k=0..5:
        # = cos0*1 + cos(pi/3)*(-1) + cos(2pi/3)*1 + cos(pi)*(-1) + cos(4pi/3)*1 + cos(5pi/3)*(-1)
        # = 1 - 0.5 + (-0.5) - (-1) + (-0.5) - 0.5 = 1 - 0.5 - 0.5 + 1 - 0.5 - 0.5 = 0  ✓
        # sum(sin(k*2pi/6)*(-1)^k) = 0 - sin(pi/3) + sin(2pi/3) - 0 + sin(4pi/3)*... let's skip.
        # Just test the actual output; if the implementation is correct, RMSD should equal 1.
        # By symmetry of the hexagon and alternating Z, H[0,2]=H[1,2]=0, H[2,2]=0.
        # Therefore H is block-diagonal and R acts only in XY, leaving Z unchanged.
        # RMSD_z² = mean(dz²) = 1, RMSD_xy² = 0 => RMSD = 1. ✓
        rmsd = kabsch_rmsd(P3, Q4)
        assert rmsd == pytest.approx(1.0, abs=1e-9)

    def test_shape_mismatch_raises(self):
        """Different shapes must raise ValueError."""
        P = np.ones((5, 3))
        Q = np.ones((6, 3))
        with pytest.raises(ValueError, match="[Ss]hape"):
            kabsch_rmsd(P, Q)

    def test_n_less_than_3_raises(self):
        """N < 3 atoms must raise ValueError."""
        P = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        Q = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="3"):
            kabsch_rmsd(P, Q)

    def test_n_equals_1_raises(self):
        """N = 1 must also raise ValueError."""
        P = np.array([[0.0, 0.0, 0.0]])
        Q = np.array([[0.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="3"):
            kabsch_rmsd(P, Q)


# ---------------------------------------------------------------------------
# align_structures
# ---------------------------------------------------------------------------

class TestAlignStructures:
    def test_returns_tuple_of_float_and_array(self):
        rng = np.random.default_rng(42)
        P = rng.random((6, 3))
        Q = rng.random((6, 3))
        rmsd, aligned = align_structures(P, Q)
        assert isinstance(rmsd, float)
        assert aligned.shape == P.shape

    def test_identical_returns_zero_rmsd(self):
        rng = np.random.default_rng(7)
        P = rng.random((7, 3)) * 15.0
        rmsd, aligned = align_structures(P, P.copy())
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_aligned_coords_close_to_ref(self):
        """After alignment the aligned mobile should be close to reference."""
        rng = np.random.default_rng(3)
        Q = rng.random((8, 3)) * 10.0
        # Rotate Q by 45° around Z and translate, then align back
        theta = np.pi / 4
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0,            0.0,           1.0],
        ])
        P = (R @ Q.T).T + np.array([3.0, -2.0, 1.0])
        rmsd, aligned = align_structures(P, Q)
        assert rmsd == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(aligned, Q, atol=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            align_structures(np.ones((4, 3)), np.ones((5, 3)))

    def test_n_less_than_3_raises(self):
        with pytest.raises(ValueError):
            align_structures(np.ones((2, 3)), np.ones((2, 3)))


# ---------------------------------------------------------------------------
# extract_ca_coords
# ---------------------------------------------------------------------------

class TestExtractCaCoords:
    def test_returns_correct_number_of_atoms(self, make_pdb):
        pdb = make_pdb(5)
        coords = extract_ca_coords(pdb)
        assert coords.shape == (5, 3)

    def test_returns_float64(self, make_pdb):
        pdb = make_pdb(5)
        coords = extract_ca_coords(pdb)
        assert coords.dtype == np.float64

    def test_coordinates_match_expected(self, make_pdb):
        """Atoms placed at (i, 0, 0) for i in 0..4 must be recovered exactly."""
        pdb = make_pdb(5)
        coords = extract_ca_coords(pdb)
        expected = np.array([[float(i), 0.0, 0.0] for i in range(5)])
        np.testing.assert_allclose(coords, expected, atol=1e-3)

    def test_accepts_str_path(self, make_pdb):
        pdb = make_pdb(4)
        coords = extract_ca_coords(str(pdb))
        assert coords.shape == (4, 3)


# ---------------------------------------------------------------------------
# compute_plddt
# ---------------------------------------------------------------------------

class TestComputePlddt:
    def test_uniform_bfactor_returns_that_value(self, make_pdb):
        """All B-factors set to 80.0 -> mean pLDDT = 80.0."""
        pdb = make_pdb(5, bfactor=80.0)
        plddt = compute_plddt(pdb)
        assert plddt == pytest.approx(80.0, abs=1e-6)

    def test_returns_float(self, make_pdb):
        pdb = make_pdb(3, bfactor=65.0)
        assert isinstance(compute_plddt(pdb), float)

    def test_different_bfactor(self, make_pdb):
        pdb = make_pdb(4, bfactor=42.5)
        assert compute_plddt(pdb) == pytest.approx(42.5, abs=1e-6)


# ---------------------------------------------------------------------------
# self_consistency_rmsd
# ---------------------------------------------------------------------------

class TestSelfConsistencyRmsd:
    def test_identical_files_give_zero(self, make_pdb):
        """Two identical PDB files must give scRMSD = 0."""
        pdb = make_pdb(5)
        rmsd = self_consistency_rmsd(pdb, pdb)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_identical_copies_give_zero(self, make_pdb, tmp_path):
        """Two separate identical PDB files must give scRMSD = 0."""
        pdb1 = make_pdb(6)
        # Write a second copy with the same coordinates
        coords = np.array([[float(i), 0.0, 0.0] for i in range(6)])
        pdb2 = tmp_path / "copy.pdb"
        _write_pdb(pdb2, coords)
        rmsd = self_consistency_rmsd(pdb1, pdb2)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_returns_float(self, make_pdb):
        pdb = make_pdb(5)
        assert isinstance(self_consistency_rmsd(pdb, pdb), float)

    def test_different_lengths_trimmed(self, make_pdb, tmp_path):
        """If one structure is longer, trim to the shorter and still get 0."""
        pdb_short = make_pdb(4)
        coords_long = np.array([[float(i), 0.0, 0.0] for i in range(6)])
        pdb_long = tmp_path / "long.pdb"
        _write_pdb(pdb_long, coords_long)
        rmsd = self_consistency_rmsd(pdb_short, pdb_long)
        assert rmsd == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_interface_contacts
# ---------------------------------------------------------------------------

class TestComputeInterfaceContacts:
    def _make_grid(self, n: int, offset: float = 0.0) -> np.ndarray:
        """n atoms on a line along X starting at offset."""
        return np.array([[offset + float(i), 0.0, 0.0] for i in range(n)])

    def test_two_arrays_5ang_apart_default_threshold(self):
        """Binder atoms 5 Å from target atoms are within default 8 Å threshold."""
        # 4 binder atoms at y=0; target atom at (0, 5, 0) — distance = 5 < 8
        binder = np.array([[0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0],
                            [3.0, 0.0, 0.0]], dtype=np.float64)
        target = np.array([[0.0, 5.0, 0.0]], dtype=np.float64)
        # All binder atoms within 8 Å of the single target atom:
        # distances: 5, sqrt(26)~5.1, sqrt(29)~5.4, sqrt(34)~5.8 — all < 8
        count = compute_interface_contacts(binder, target)
        assert count == 4

    def test_arrays_far_apart_no_contacts(self):
        """Binder and target 100 Å apart -> 0 contacts with default threshold."""
        binder = self._make_grid(5, offset=0.0)
        target = self._make_grid(5, offset=100.0)
        count = compute_interface_contacts(binder, target)
        assert count == 0

    def test_custom_threshold(self):
        """With threshold = 4 Å, atoms 5 Å apart should not contact."""
        binder = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        target = np.array([[5.0, 0.0, 0.0]], dtype=np.float64)
        count = compute_interface_contacts(binder, target, distance_threshold=4.0)
        assert count == 0

    def test_returns_int(self):
        binder = np.array([[0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [2.0, 0.0, 0.0]], dtype=np.float64)
        target = np.array([[0.5, 0.0, 0.0]], dtype=np.float64)
        result = compute_interface_contacts(binder, target)
        assert isinstance(result, int)

    def test_5_ang_separation_exact_count(self):
        """Explicit 5 Å separation with known geometry -> expected contact count."""
        # Place binder atoms at z=0 and a single target atom at z=5.
        # All binder atoms are within 8 Å (default) of the target.
        binder = np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [8.0, 0.0, 0.0],
        ], dtype=np.float64)
        target = np.array([[0.0, 0.0, 5.0]], dtype=np.float64)
        # Distances from target: 5, sqrt(29)≈5.39, sqrt(41)≈6.40, sqrt(61)≈7.81, sqrt(89)≈9.43
        # Within 8 Å: atoms at x=0,2,4,6 (distances 5, 5.39, 6.40, 7.81) -> 4 contacts
        count = compute_interface_contacts(binder, target, distance_threshold=8.0)
        assert count == 4
