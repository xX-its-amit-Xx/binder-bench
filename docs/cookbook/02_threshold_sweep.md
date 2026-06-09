# Cookbook 2: Threshold Sweep — Yield vs. Stringency Trade-off

This cookbook shows how varying RMSD and pLDDT thresholds changes the number of
"passing" designs and why the choice of operating point matters for your campaign.

---

## 1. Why Thresholds Matter

Every binder design campaign ends with a filter step: designs whose folded
structure does not recapitulate the intended backbone (high RMSD) or whose
predicted structure is low-confidence (low pLDDT) are discarded before
experimental testing.  Two numbers control this gate:

| Parameter | What it measures | Direction |
|-----------|-----------------|-----------|
| RMSD (Å) | Backbone deviation between designed and folded structure | Lower is better |
| pLDDT (0–100) | Per-residue confidence of the fold prediction | Higher is better |

Setting these thresholds involves a fundamental **yield-stringency trade-off**:

- **Tight thresholds** (RMSD < 1.5 Å, pLDDT > 80): only the most self-consistent,
  high-confidence designs survive.  You are unlikely to waste experimental slots on
  noise, but many potentially valid binders are discarded, and your passing set can
  be very small.
- **Loose thresholds** (RMSD < 4.0 Å, pLDDT > 50): many designs pass, giving you a
  larger set to test.  However, a larger fraction will fail experimentally because the
  computational filter was not selective enough.

The RFdiffusion paper (Watson et al., 2023, *Nature*; Supplementary Fig. S5) shows
this trade-off empirically: success rates drop sharply as thresholds tighten, while
the absolute number of passing designs increases monotonically as thresholds relax.
The key insight is that neither extreme is optimal — campaigns must balance downstream
experimental throughput against the cost of testing designs that will not bind.

---

## 2. Running a Threshold Sweep with Python

The sweep iterates over a grid of `(rmsd_threshold, plddt_threshold)` pairs and
records, for each combination, how many designs from a pre-scored run would pass.

```python
import itertools, pandas as pd
from binder_bench.config import BinderBenchConfig, FilterConfig
from binder_bench.filter import run_filter, results_to_dataframe
from binder_bench.plots import plot_threshold_sweep

# Load a pre-run scoreboard
df = pd.read_parquet("output/.../scoreboard.parquet")

rmsd_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
plddt_thresholds = [50, 60, 70, 80, 90]

rows = []
for rmsd_t, plddt_t in itertools.product(rmsd_thresholds, plddt_thresholds):
    passing = df[(df.rmsd < rmsd_t) & (df.plddt > plddt_t)]
    rows.append(dict(rmsd_threshold=rmsd_t, plddt_threshold=plddt_t,
                     n_pass=len(passing), success_rate=len(passing)/len(df)))
sweep = pd.DataFrame(rows)
```

The resulting `sweep` DataFrame has one row per `(rmsd_threshold, plddt_threshold)`
combination and two outcome columns:

- `n_pass` — absolute count of designs that would pass at that operating point.
- `success_rate` — fraction of all designs that pass (0.0–1.0).

> **Note on the scoreboard format**: `binder_bench` writes `scoreboard.parquet`
> under `output/<run_id>/` by default (controlled by `OutputConfig.output_dir`
> and `OutputConfig.scoreboard_format`).  The file always contains `rmsd` and
> `plddt` columns produced by `results_to_dataframe()`, so the filtering
> expressions above work without modification.

---

## 3. Generating the Heatmap

Pass the sweep DataFrame to `plot_threshold_sweep()` to produce a publication-ready
heatmap:

```python
fig = plot_threshold_sweep(
    sweep,
    value_col="success_rate",   # colour by fraction passing
    title="Threshold sweep — IL-6R binder campaign (n=500 designs)",
    annotate=True,              # print numeric values inside each cell
    fmt=".0%",                  # format values as percentages
    output_path="output/threshold_sweep.png",
)
fig.show()
```

**What the heatmap shows:**

The x-axis spans RMSD thresholds from tight (left) to loose (right); the y-axis
spans pLDDT thresholds from low (bottom) to high (top).  Each cell is coloured by
`success_rate`, with darker (or more saturated) colours indicating more designs
passing.  A strong gradient from the upper-left corner (few designs pass) toward the
lower-right corner (most designs pass) is typical and confirms that the metrics are
acting as effective filters.  Cells where success rate changes steeply between
neighbouring thresholds indicate a region of high sensitivity — small changes to
cutoffs meaningfully alter yield there.  Flat plateaus in the heatmap indicate that
the filter is already saturated and relaxing thresholds further adds mostly noise.

---

## 4. Interpreting Results: Operating Points

The table below summarises three canonical operating points used in the literature
and their typical computational success rates on de novo binder campaigns:

| Operating point | RMSD threshold | pLDDT threshold | Typical success rate | Use case |
|-----------------|---------------|-----------------|----------------------|----------|
| **Lenient** | < 4.0 Å | > 50 | 60–80 % | Early-stage exploration; maximise diversity |
| **Moderate** | < 2.0 Å | > 70 | 30–45 % | Standard screening; good balance of yield and quality |
| **Stringent** | < 1.5 Å | > 80 | 10–20 % | Pre-synthesis triage; minimise wasted experimental slots |

"Success rate" here means the computational pass rate — the fraction of generated
designs that clear both thresholds.  Experimental binding rates are always lower and
are not reflected in these numbers.

When your heatmap shows a success rate at the moderate operating point that is far
outside the 30–45 % range, it is worth investigating:

- **Much lower (< 10 %)**: the backbone generator may be producing geometrically
  inconsistent structures, or the fold predictor is systematically uncertain on your
  target class.  Try increasing `n_designs` before tightening thresholds.
- **Much higher (> 70 %)**: thresholds may be too lenient for your target, or the
  stub/mock backends are being used instead of real structure predictors.

---

## 5. Published Benchmarks

The RFdiffusion paper (Watson et al., 2023; arXiv:2210.04673) reports the following
computational self-consistency rates for de novo binder designs targeting diverse
protein targets:

- **RMSD < 2.0 Å and pLDDT > 70**: approximately **40 %** of designs pass on short
  binders (50–80 residues).  This is the most-cited reference operating point from the
  Baker lab campaigns and forms the baseline expectation for the moderate threshold.
- **RMSD < 1.5 Å and pLDDT > 80**: approximately **15 %** of designs pass.  This
  stringent cutoff was used to select candidates for experimental characterisation in
  the IL-7Rα and TrkA campaigns described in that work.

These numbers were obtained with ESMFold as the fold predictor and Cα RMSD computed
over the binder chain after superposition.  Results may differ with AlphaFold2 or
other predictors because pLDDT scales can vary between models.

> **Citation**: Watson, J. L., et al. "De novo design of protein structure and
> function with RFdiffusion." *Nature* 620, 1089–1100 (2023).
> Preprint: https://arxiv.org/abs/2210.04673

---

## 6. Recommended Starting Point

For an initial screening campaign, use:

```
RMSD < 2.0 Å
pLDDT > 70
```

These values correspond to `FilterConfig` defaults in `binder_bench`:

```python
from binder_bench.config import FilterConfig

cfg = FilterConfig(
    rmsd_threshold=2.0,   # default
    plddt_threshold=70.0, # default
)
```

This moderate operating point is calibrated to match the Baker lab benchmark
numbers (~40 % computational pass rate) and provides enough passing designs to
proceed to scoring and experimental prioritisation without overwhelming downstream
steps.  Once you have observed the success rate on your specific target, use the
threshold sweep to tighten or relax the cutoffs before committing to a larger
generation run.
