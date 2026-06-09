# Cookbook 1: Full Generate→Design→Fold→Filter→Score Run Against IL-6 Receptor

## 1. Background

Interleukin-6 receptor (IL-6R) is a type I cytokine receptor whose signalling axis drives chronic inflammation and is implicated in rheumatoid arthritis, Castleman disease, and cytokine release syndrome. Blocking the IL-6/IL-6R interaction is a clinically validated therapeutic strategy — tocilizumab, a monoclonal antibody targeting IL-6R, has FDA approvals across multiple indications. The canonical structure of IL-6 in complex with its receptor is deposited as PDB [1IL6](https://www.rcsb.org/structure/1IL6), which provides the structural reference used in this cookbook for hotspot identification and RMSD evaluation.

## 2. The Target Epitope

The quickstart config targets chain A of IL-6R at the IL-6 binding interface. The key contact region spans approximately residues 170–230 of the D2 domain (the membrane-proximal immunoglobulin-like domain), where IL-6 site II residues make the dominant energetic contribution to the complex. Hotspot residues F229, Y205, and R179 (IL-6R chain A numbering from 1IL6) account for the majority of buried surface area and are the recommended anchor points for diffusion-based binder hallucination.

The quickstart config expresses this as:

```yaml
target_name: "il6_receptor"
target_pdb: "examples/target_il6.pdb"
generate:
  hotspot_residues: []   # set to e.g. ["A179", "A205", "A229"] for GPU runs
```

In stub mode the hotspot list is accepted but ignored; on a real GPU run with RFdiffusion you would populate `hotspot_residues` to constrain diffusion toward the epitope.

## 3. Running the Pipeline

### 3a. CLI

The fastest way to launch the full five-step pipeline (Generate → Design → Fold → Filter → Score) is:

```bash
binder-bench run examples/config_quickstart.yaml -v
```

The `-v` flag enables debug logging so you can observe per-step timing. You should see output similar to:

```
Starting pipeline for target: il6_receptor
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Step 1/5: Generating backbones...
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Generate complete: 3 backbones in 0.0s
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Step 2/5: Designing sequences...
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Design complete: 6 sequences across 3 backbones in 0.0s
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Step 3/5: Folding sequences...
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Fold complete: 6 structures in 0.0s
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Step 4/5: Filtering structures...
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Filter complete: 6/6 passing in 0.0s
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Step 5/5: Scoring structures...
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Score complete: 6 entries scored in 0.0s
2026-06-09 12:00:01 [INFO]  binder_bench.pipeline: [20260609_120001] Pipeline complete in 0.1s
```

Outputs are written to `output/<run_id>/` where `<run_id>` is a timestamp string (e.g. `20260609_120001`).

To validate the config without running anything:

```bash
binder-bench validate examples/config_quickstart.yaml
```

### 3b. Python API

You can drive the same pipeline programmatically:

```python
from pathlib import Path
from binder_bench.config import BinderBenchConfig
from binder_bench.pipeline import run_pipeline

# Load config from YAML
config = BinderBenchConfig.from_yaml("examples/config_quickstart.yaml")

# Optional: override output directory at runtime
config.output.output_dir = Path("output/my_il6_run")

# Run the full pipeline
result = run_pipeline(config)

# Inspect top-level summary
print(f"Run ID       : {result.run_id}")
print(f"Backbones    : {len(result.backbones)}")
print(f"Designs      : {sum(len(v) for v in result.sequences.values())}")
print(f"Folded       : {len(result.folded)}")
print(f"Elapsed (s)  : {result.elapsed_seconds:.2f}")

# Aggregate stats
print(result.stats)
# {
#   'n_total': 6, 'n_pass': 6, 'success_rate': 1.0,
#   'mean_plddt_pass': 75.0, 'mean_rmsd_pass': 0.0,
#   'mean_contacts_pass': ..., 'best_design_id': 'backbone_0_seq_0'
# }

# Full scoreboard as a DataFrame
print(result.scoreboard.head())
```

## 4. Results (Stub Mode)

The quickstart config uses `backend: stub` for all three compute-heavy steps. The stub backends return pre-built fixture files from `examples/backbones/`, `examples/sequences/`, and `examples/structures/` rather than calling RFdiffusion, ProteinMPNN, or ESMFold.

| Quantity | Stub value | What to expect on real GPU hardware |
|---|---|---|
| Backbones generated | 3 | Set `n_designs: 50–200` for a production screen |
| Sequences per backbone | 2 | Typical production runs use 8–32 |
| Total designs | 6 | 400–6 400 with the above settings |
| Typical pLDDT | ~75 (fixture) | 60–90; filter threshold is 70 |
| Typical self-consistency RMSD | ~0 A (identical fixture files) | 0.5–3 A; filter threshold is 2 A |
| Success rate (RMSD<2 A + pLDDT>70) | 100% in stub (trivial) | 20–40% reported in RFdiffusion+ProteinMPNN literature |

The RMSD of ~0 A in stub mode is expected: the stub fold backend returns the same PDB file that was used as the backbone input, so the self-consistency RMSD between backbone and fold is zero by construction. On a real run, ESMFold or AlphaFold2 would predict an independent fold from the ProteinMPNN sequence, and you would observe nonzero RMSD.

Published benchmarks using RFdiffusion for backbone generation and ProteinMPNN for sequence design report 20–40% of designs passing RMSD < 2 A and pLDDT > 70 at the 50-backbone/8-sequence scale ([Bennett et al., 2023, Science](https://doi.org/10.1126/science.adf3753)).

## 5. Reading the Scoreboard

The pipeline saves a Parquet scoreboard to `output/<run_id>/scoreboard.parquet`. Read it with pandas:

```python
import pandas as pd

# Replace the run_id timestamp with your actual directory name
df = pd.read_parquet("output/20260609_120001/scoreboard.parquet")

print(df.sort_values("rank").head())
```

Expected columns and their meaning:

| Column | Type | Description |
|---|---|---|
| `design_id` | str | Unique identifier: `<backbone_stem>_seq_<n>` |
| `rank` | int | 1-indexed rank (1 = best) |
| `rmsd` | float | Self-consistency RMSD in Angstroms vs. designed backbone |
| `plddt` | float | Mean pLDDT of the folded structure (0–100) |
| `passes_filter` | bool | True if RMSD <= threshold AND pLDDT >= threshold |
| `n_interface_contacts` | int | CA-CA contacts within 8 A at the binder-target interface |
| `interface_score` | float | `n_interface_contacts / n_binder_residues * 100` |
| `rosetta_score` | float or NaN | Rosetta dG (only if `score.rosetta_enabled: true`) |
| `foldx_score` | float or NaN | FoldX ddG (only if `score.foldx_enabled: true`) |

Designs are ranked by: passing filter first, then descending pLDDT, then ascending RMSD, then descending interface score. Use `passes_filter == True` as the primary selection gate before any downstream wet-lab work.

You can also re-score an existing output directory with different thresholds without re-running the pipeline:

```bash
binder-bench score-only output/20260609_120001 --rmsd-threshold 1.5 --plddt-threshold 75
```

## 6. Runtime Comparison: Stub CPU vs. GPU

| Step | Stub (CPU, no model weights) | Typical GPU (A100, production scale) |
|---|---|---|
| Generate (3 backbones) | < 1 s | 5–15 min (RFdiffusion, 200 steps) |
| Design (6 sequences) | < 1 s | 30–90 s (ProteinMPNN, 8 seqs/backbone) |
| Fold (6 structures) | < 1 s | 2–5 min (ESMFold API or local) |
| Filter + Score (6 designs) | < 1 s | < 1 s (CPU; RMSD + pLDDT are lightweight) |
| **Total (6 designs)** | **< 1 s** | **~10–20 min** |
| Total (scaled to 400 designs) | < 5 s | ~2–4 hours |

The stub backend is intended for CI, unit tests, and pipeline development. Switch to real backends by setting `--backend rfdiffusion` (generate), `--backend proteinmpnn` (design), and `--backend esmfold_api` (fold) on the CLI, or by editing the corresponding `backend:` fields in the YAML config.

## 7. Next Steps

- **Cookbook 2 — Threshold Sweep**: Learn how to vary `rmsd_threshold` and `plddt_threshold` in a parameter sweep and plot the resulting success-rate curves. See `docs/cookbook/02_threshold_sweep.md`.
- To switch from stub to real backends, edit `examples/config_quickstart.yaml`, set the three `backend:` fields to `rfdiffusion`, `proteinmpnn`, and `esmfold_api`, and ensure the corresponding model weights and environment variables are configured.
- For larger screens, increase `n_designs` (up to 1000) and `n_seqs_per_backbone` (up to 64) in the config.
- To restrict diffusion to the IL-6R epitope, set `generate.hotspot_residues: ["A179", "A205", "A229"]` in the YAML.
