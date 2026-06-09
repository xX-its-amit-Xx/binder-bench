# Cookbook 4: Loading Scoreboard into pandas & Appending a Docking Stage

After `binder-bench` finishes a run the primary artifact is a ranked
`scoreboard.parquet` (or `scoreboard.csv`) file written under
`output/<run_id>/`.  This cookbook shows how to load that file into pandas,
understand every column, export a shortlist for your wet-lab collaborators,
extend the pipeline with a docking stage, and feed the results into a
downstream foldchain workflow.

---

## 1. Loading binder-bench Output into pandas

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RUN_DIR = Path("output/run_001")

# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------
df = pd.read_parquet(RUN_DIR / "scoreboard.parquet")

print(f"Total designs scored : {len(df)}")
print(f"Designs passing filter: {df['passes_filter'].sum()}")
print(df.dtypes)
print(df.head())

# ------------------------------------------------------------------
# Basic filtering and sorting
# ------------------------------------------------------------------
passing = (
    df[df["passes_filter"]]
    .sort_values("rank")          # rank is 1-indexed; rank 1 is the best design
    .reset_index(drop=True)
)

print(f"\nTop 5 passing designs:")
print(passing[["rank", "design_id", "plddt", "rmsd", "interface_score"]].head(5))

# ------------------------------------------------------------------
# Scatter: pLDDT vs RMSD coloured by interface_score
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
sc = ax.scatter(
    df["rmsd"],
    df["plddt"],
    c=df["interface_score"],
    cmap="viridis",
    alpha=0.7,
    edgecolors="none",
    s=40,
)
plt.colorbar(sc, ax=ax, label="Interface score (%)")
ax.set_xlabel("Self-consistency RMSD (Å)")
ax.set_ylabel("pLDDT")
ax.set_title("All designs – pLDDT vs RMSD")

# Mark passing designs
ax.scatter(
    passing["rmsd"],
    passing["plddt"],
    facecolors="none",
    edgecolors="red",
    linewidths=1.2,
    s=60,
    label="Passes filter",
    zorder=3,
)
ax.legend(fontsize=8)

# ------------------------------------------------------------------
# Bar: interface contacts distribution for passing designs
# ------------------------------------------------------------------
ax2 = axes[1]
ax2.bar(
    passing["rank"],
    passing["n_interface_contacts"],
    color="steelblue",
    width=0.8,
)
ax2.set_xlabel("Rank")
ax2.set_ylabel("Interface contacts")
ax2.set_title("Passing designs – interface contacts by rank")
ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(RUN_DIR / "scoreboard_summary.png", dpi=150)
plt.show()
```

---

## 2. Scoreboard Schema

The Parquet (and CSV) scoreboard contains exactly one row per design.
Columns are written in the order shown below.

| Column | dtype | Description |
|---|---|---|
| `design_id` | `str` | Unique identifier for the design, e.g. `bb_000_seq_003`. Derived from backbone index and sequence index. |
| `rank` | `int` | 1-indexed rank across **all** designs (passing and failing). Rank 1 is the best. Sorted by: `passes_filter` desc → `plddt` desc → `rmsd` asc → `interface_score` desc. |
| `backbone_path` | `str` | Absolute path to the designed backbone PDB produced by the generate stage. |
| `folded_path` | `str` | Absolute path to the ESMFold (or stub) predicted structure PDB. |
| `rmsd` | `float64` | Cα RMSD (Å) between the designed backbone and the folded structure. Lower is better. A value of `-1.0` indicates a metric-computation failure. |
| `plddt` | `float64` | Mean per-residue pLDDT confidence score of the folded structure (0–100). Higher is better. A value of `-1.0` indicates a metric-computation failure. |
| `passes_filter` | `bool` | `True` if the design satisfies all three filter thresholds: `rmsd <= rmsd_threshold`, `plddt >= plddt_threshold`, and `min_binder_length <= n_residues <= max_binder_length`. |
| `n_interface_contacts` | `int64` | Number of binder Cα atoms within `contact_distance` Å of any target Cα atom (default 8.0 Å). |
| `interface_score` | `float64` | Normalised interface score: `(n_interface_contacts / n_binder_residues) * 100`. Range 0–100; higher indicates a denser interface. |
| `rosetta_score` | `float64` or `None` | Rosetta REU total score for the binder–target complex. `None` unless `score.rosetta_enabled = true` in the config. |
| `foldx_score` | `float64` or `None` | FoldX ΔΔG estimate (kcal/mol). `None` unless `score.foldx_enabled = true` in the config. |

> **Tip:** `passes_filter` is the primary column for wet-lab shortlisting.
> `rank` can be used directly as a priority order within passing designs.

---

## 3. Exporting to CSV for Wet-Lab Collaborators

```python
import pandas as pd

df = pd.read_parquet("output/run_001/scoreboard.parquet")
top20 = df[df.passes_filter].sort_values("rank").head(20)
top20[["design_id", "plddt", "rmsd", "interface_score", "rank"]].to_csv("top20_for_lab.csv", index=False)
```

The resulting `top20_for_lab.csv` contains only the five columns most relevant
to a wet-lab handoff: the design identifier, confidence (pLDDT), structural
self-consistency (RMSD), interface quality score, and overall rank.  If you
also want the PDB file paths for retrieval, append `"folded_path"` to the
column list.

---

## 4. Appending a Docking Stage

The scoreboard gives you a first-pass quality signal based on
self-consistency and interface geometry.  A natural next step is to
evaluate binding affinity more rigorously with a docking tool such as
**AF2-multimer** (AlphaFold2 multimer prediction) or **RosettaDock**
(rigid-body + side-chain optimisation).

### Conceptual workflow

```
scoreboard.parquet
      |
      v
[select passing designs]  -->  dock_binder_target(binder_pdb, target_pdb) -> docking_score
      |
      v
scoreboard with new "docking_score" column  -->  re-rank & export
```

### Pseudocode stub

```python
from pathlib import Path
import pandas as pd


def dock_binder_target(binder_pdb: Path, target_pdb: Path) -> float:
    """Run a docking calculation and return a scalar docking score.

    Replace the body with your actual docking backend call, e.g.:
      - subprocess call to RosettaDock (returns interface dG in REU)
      - AF2-multimer prediction followed by ipTM extraction
      - Any other docking engine that writes a score to stdout or a file

    Returns
    -------
    float
        Docking score.  Convention: lower (more negative) is better when
        using energy-based tools; higher is better when using ipTM.
        Normalise appropriately before combining with other metrics.
    """
    raise NotImplementedError("Replace with your docking backend call")


def append_docking_scores(
    scoreboard_path: Path,
    target_pdb: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Load a scoreboard, run docking for all passing designs, and save results."""
    df = pd.read_parquet(scoreboard_path)

    docking_scores = []
    for _, row in df.iterrows():
        if row["passes_filter"]:
            score = dock_binder_target(Path(row["folded_path"]), target_pdb)
        else:
            score = float("nan")   # skip failed designs
        docking_scores.append(score)

    df["docking_score"] = docking_scores

    # Re-rank passing designs by docking score (lower is better for energy-based)
    df_passing = df[df["passes_filter"]].sort_values("docking_score")
    df_failing = df[~df["passes_filter"]]
    df_out = pd.concat([df_passing, df_failing], ignore_index=True)
    df_out["rank"] = range(1, len(df_out) + 1)

    df_out.to_parquet(output_path, index=False)
    print(f"Saved docking-enriched scoreboard to {output_path}")
    return df_out
```

Call it as:

```python
enriched = append_docking_scores(
    scoreboard_path=Path("output/run_001/scoreboard.parquet"),
    target_pdb=Path("targets/IL6R.pdb"),
    output_path=Path("output/run_001/scoreboard_docked.parquet"),
)
```

---

## 5. Cross-Reference to foldchain

If you have a **foldchain** pipeline that accepts a list of
`{"binder": path, "target": path, "id": design_id}` dicts, binder-bench's
scoreboard maps directly to that format.  The adapter below selects all
passing designs and constructs the foldchain input manifest:

```python
from pathlib import Path
import pandas as pd


def to_foldchain_input(scoreboard_df: pd.DataFrame, target_pdb: Path) -> list[dict]:
    # foldchain expects: [{"binder": path, "target": path, "id": design_id}]
    passing = scoreboard_df[scoreboard_df.passes_filter]
    return [{"binder": str(r.folded_path), "target": str(target_pdb), "id": r.design_id}
            for r in passing.itertuples()]
```

Usage example:

```python
import json

df = pd.read_parquet("output/run_001/scoreboard.parquet")
manifest = to_foldchain_input(df, target_pdb=Path("targets/IL6R.pdb"))

# Inspect
print(f"{len(manifest)} designs ready for foldchain")
print(manifest[0])
# {'binder': 'output/run_001/passing/bb_000_seq_003.pdb',
#  'target': 'targets/IL6R.pdb',
#  'id': 'bb_000_seq_003'}

# Write to disk so foldchain can consume it
with open("output/run_001/foldchain_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
```

The `folded_path` column already contains absolute paths to the binder PDB
files written during the fold stage (or copied to `passing/` by the pipeline).
If you want binder-bench to always emit absolute paths, make sure `output_dir`
in `OutputConfig` is itself an absolute path in your config YAML.

---

## 6. Reusable Notebook Template

Below is the cell-by-cell outline of a Jupyter notebook that loads a
binder-bench run and produces a publication-quality figure.  Copy the
skeleton into a new `.ipynb` and fill in your run directory.

```
Cell 1 – Imports and paths
──────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RUN_DIR   = Path("output/run_001")
TARGET_ID = "IL6R"          # used in figure title / save name
PASSING_COLOUR = "#2196F3"
FAILING_COLOUR = "#BDBDBD"

Cell 2 – Load scoreboard
─────────────────────────
df      = pd.read_parquet(RUN_DIR / "scoreboard.parquet")
passing = df[df.passes_filter].sort_values("rank")
failing = df[~df.passes_filter]

print(f"n_total={len(df)}  n_pass={len(passing)}"
      f"  success_rate={len(passing)/len(df)*100:.1f}%")

Cell 3 – Compute summary stats
────────────────────────────────
stats = {
    "mean_plddt_pass"     : passing.plddt.mean(),
    "median_rmsd_pass"    : passing.rmsd.median(),
    "mean_contacts_pass"  : passing.n_interface_contacts.mean(),
    "best_design_id"      : passing.iloc[0].design_id if len(passing) else None,
}
print(stats)

Cell 4 – Publication figure (2 × 2 panel)
───────────────────────────────────────────
fig = plt.figure(figsize=(12, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

# Panel A: pLDDT vs RMSD scatter
ax_a = fig.add_subplot(gs[0, 0])
ax_a.scatter(failing.rmsd, failing.plddt,
             s=20, alpha=0.4, color=FAILING_COLOUR, label="Fail", zorder=1)
sc = ax_a.scatter(passing.rmsd, passing.plddt,
                  c=passing.interface_score, cmap="plasma",
                  s=40, alpha=0.85, label="Pass", zorder=2)
plt.colorbar(sc, ax=ax_a, label="Interface score (%)")
ax_a.set_xlabel("RMSD (Å)")
ax_a.set_ylabel("pLDDT")
ax_a.set_title("A  pLDDT vs RMSD", loc="left", fontweight="bold")
ax_a.legend(fontsize=8)

# Panel B: interface score distribution
ax_b = fig.add_subplot(gs[0, 1])
ax_b.hist(passing.interface_score, bins=20, color=PASSING_COLOUR, edgecolor="white")
ax_b.set_xlabel("Interface score (%)")
ax_b.set_ylabel("Count")
ax_b.set_title("B  Interface score distribution (passing)", loc="left", fontweight="bold")

# Panel C: ranked pLDDT bar chart (top 20 passing)
ax_c = fig.add_subplot(gs[1, 0])
top20 = passing.head(20)
bars  = ax_c.bar(top20.rank, top20.plddt, color=PASSING_COLOUR, width=0.7)
ax_c.axhline(70, color="red", linestyle="--", linewidth=0.8, label="Filter threshold")
ax_c.set_xlabel("Rank")
ax_c.set_ylabel("pLDDT")
ax_c.set_title("C  Top-20 passing designs – pLDDT by rank", loc="left", fontweight="bold")
ax_c.legend(fontsize=8)

# Panel D: cumulative success rate as function of threshold
ax_d = fig.add_subplot(gs[1, 1])
rmsd_thresholds = np.linspace(0.5, 4.0, 50)
success_rates   = [
    (df.rmsd <= t).sum() / len(df) * 100
    for t in rmsd_thresholds
]
ax_d.plot(rmsd_thresholds, success_rates, color="darkorange", linewidth=1.8)
ax_d.set_xlabel("RMSD threshold (Å)")
ax_d.set_ylabel("Designs passing (%)")
ax_d.set_title("D  Cumulative success rate vs RMSD threshold", loc="left", fontweight="bold")

fig.suptitle(f"binder-bench results – {TARGET_ID}", fontsize=14, fontweight="bold", y=1.01)

plt.savefig(RUN_DIR / f"{TARGET_ID}_scoreboard_figure.pdf", bbox_inches="tight")
plt.savefig(RUN_DIR / f"{TARGET_ID}_scoreboard_figure.png", dpi=200, bbox_inches="tight")
plt.show()

Cell 5 – Export CSV for wet lab
─────────────────────────────────
top20 = passing.head(20)
top20[["design_id", "plddt", "rmsd", "interface_score", "rank"]].to_csv(
    RUN_DIR / "top20_for_lab.csv", index=False
)
print("Saved top20_for_lab.csv")
```

> **Saving to PDF:** `matplotlib` writes vector PDFs natively – no additional
> dependency is required.  Use `bbox_inches="tight"` to prevent clipping of
> axis labels.  For journal submission, set `dpi=300` on the PNG fallback.
