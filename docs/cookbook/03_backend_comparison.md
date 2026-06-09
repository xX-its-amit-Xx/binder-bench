# Cookbook 3: Comparing ProteinMPNN vs. Stub Baseline on the Same Backbones

This cookbook walks through running two sequence-design backends on an identical
set of backbones and comparing the results. By the end you will have two
scoreboards side-by-side and a clear picture of how much sequence optimization
matters relative to a random-sequence baseline.

---

## 1. Why Compare Backends?

ProteinMPNN (Dauparas et al., *Science* 2022) is the current gold standard for
fixed-backbone sequence design. It uses a graph-neural-network trained on
thousands of protein structures to propose sequences that fold back onto the
input backbone with high confidence.

The stub baseline assigns sequences without any structural reasoning — it either
reads pre-computed placeholder sequences from disk or generates random amino-acid
strings. Running both backends on the exact same backbones isolates the
contribution of sequence optimization:

- **High gap between stub and ProteinMPNN** → sequence choice is the bottleneck;
  investing in better design methods will pay off.
- **Small gap** → the backbone geometry is the bottleneck; focus on the backbone
  generation step first (see Cookbook 1).

This experiment also makes a convenient sanity check: if ProteinMPNN does not
beat random sequences on your target, something is likely wrong with your filter
thresholds or folding setup.

---

## 2. Setup

This cookbook reuses the three IL-6R backbones introduced in Cookbook 1. Make
sure you have them available:

```
examples/backbones/il6r_bb_0001.pdb
examples/backbones/il6r_bb_0002.pdb
examples/backbones/il6r_bb_0003.pdb
```

You will run the pipeline twice from two separate config files that differ only
in the `design.backend` field and the `output.output_dir` path. Create a shared
base config first:

```yaml
# configs/il6r_base.yaml
target_name: "il6_receptor"
target_pdb: "examples/target_il6.pdb"

generate:
  backend: "stub"
  n_designs: 3
  seed: 42
  stub_dir: "examples/backbones"

fold:
  backend: "esmfold_api"   # or "stub" if you prefer an offline run
  stub_dir: "examples/structures"

filter:
  rmsd_threshold: 2.0
  plddt_threshold: 70.0
  min_binder_length: 50
  max_binder_length: 150

score:
  contact_distance: 8.0
  min_interface_contacts: 10
```

---

## 3. Running the ProteinMPNN Backend

ProteinMPNN requires a CUDA-capable GPU and a separate installation step.

```bash
# Install ProteinMPNN (requires CUDA)
pip install git+https://github.com/dauparas/ProteinMPNN.git
export PROTEINMPNN_PATH=/path/to/ProteinMPNN
```

Create the ProteinMPNN config by extending the base:

```yaml
# configs/il6r_mpnn.yaml  — inherits everything from il6r_base, overrides below
target_name: "il6_receptor"
target_pdb: "examples/target_il6.pdb"

generate:
  backend: "stub"
  n_designs: 3
  seed: 42
  stub_dir: "examples/backbones"

design:
  backend: "proteinmpnn"   # <-- key change
  n_seqs_per_backbone: 8
  temperature: 0.1
  seed: 42

fold:
  backend: "esmfold_api"
  stub_dir: "examples/structures"

filter:
  rmsd_threshold: 2.0
  plddt_threshold: 70.0
  min_binder_length: 50
  max_binder_length: 150

score:
  contact_distance: 8.0
  min_interface_contacts: 10

output:
  output_dir: "output/mpnn_run"
  scoreboard_format: "parquet"
  save_passing_structures: true
  plot_success_rate: true
```

Run it:

```bash
binder-bench run configs/il6r_mpnn.yaml --verbose
```

---

## 4. Running the Stub Baseline

The stub backend loads pre-computed placeholder sequences from disk and requires
no GPU. Create the stub config:

```yaml
# configs/il6r_stub.yaml
target_name: "il6_receptor"
target_pdb: "examples/target_il6.pdb"

generate:
  backend: "stub"
  n_designs: 3
  seed: 42
  stub_dir: "examples/backbones"

design:
  backend: "stub"           # <-- stub sequences, no optimization
  n_seqs_per_backbone: 8
  temperature: 0.1
  seed: 42
  stub_dir: "examples/sequences"

fold:
  backend: "esmfold_api"
  stub_dir: "examples/structures"

filter:
  rmsd_threshold: 2.0
  plddt_threshold: 70.0
  min_binder_length: 50
  max_binder_length: 150

score:
  contact_distance: 8.0
  min_interface_contacts: 10

output:
  output_dir: "output/stub_run"
  scoreboard_format: "parquet"
  save_passing_structures: true
  plot_success_rate: true
```

Run it:

```bash
binder-bench run configs/il6r_stub.yaml --verbose
```

Both runs use identical backbones (`stub_dir: "examples/backbones"` in the
`generate` block), so any difference in the downstream metrics comes purely from
the design step.

---

## 5. Comparing Results

After both runs complete, load their scoreboards and build a side-by-side summary:

```python
import pandas as pd

stub_df = pd.read_parquet("output/stub_run/scoreboard.parquet")
mpnn_df = pd.read_parquet("output/mpnn_run/scoreboard.parquet")

comparison = pd.DataFrame({
    "backend": ["stub", "proteinmpnn"],
    "success_rate": [stub_df.passes_filter.mean(), mpnn_df.passes_filter.mean()],
    "mean_plddt": [stub_df.plddt.mean(), mpnn_df.plddt.mean()],
    "mean_rmsd": [stub_df.rmsd.mean(), mpnn_df.rmsd.mean()],
})
print(comparison)
```

For a richer view, plot the pLDDT distributions:

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (label, df) in zip(axes, [("stub", stub_df), ("proteinmpnn", mpnn_df)]):
    ax.hist(df["plddt"], bins=20, edgecolor="black")
    ax.axvline(70.0, color="red", linestyle="--", label="pLDDT threshold")
    ax.set_title(label)
    ax.set_xlabel("pLDDT")
    ax.set_ylabel("count")
    ax.legend()

plt.tight_layout()
plt.savefig("output/plddt_comparison.png", dpi=150)
```

---

## 6. Expected Results

The table below shows representative numbers. Your exact figures will vary with
backbone quality, the number of sequences per backbone, and the folding backend.

| Metric | Stub baseline | ProteinMPNN |
|---|---|---|
| Success rate (passes filter) | ~5–15 % | ~40–65 % |
| Mean pLDDT (all designs) | ~55–62 | ~72–80 |
| Mean RMSD to backbone (A) | ~2.5–4.0 | ~1.2–2.0 |
| Mean interface contacts | ~4–8 | ~12–20 |

**Notes on the stub numbers.** Stub sequences are placeholder strings that were
not optimized to fold onto the input backbone, so pLDDT scores tend to fall
below the 70.0 filter threshold and RMSD to the designed backbone tends to be
high. A small fraction of stub designs may pass the filter by chance, which is
why the success rate is non-zero rather than zero.

**Notes on the ProteinMPNN numbers.** The success rates and pLDDT values above
are consistent with the published benchmark results in Dauparas et al. (2022),
which reported per-residue recovery of ~52.4 % on single chains and significantly
higher folding confidence versus baseline methods. Higher `n_seqs_per_backbone`
values (up to the config maximum of 64) and lower sampling temperatures (0.05–0.1)
generally improve success rates at the cost of sequence diversity.

> **Reference:** Dauparas, J. et al. Robust deep learning-based protein sequence
> design using ProteinMPNN. *Science* **378**, 49–56 (2022).
> https://doi.org/10.1126/science.add2187

---

## 7. Adding a New Backend via the Plugin API

binder-bench uses an abstract base class (`SequenceDesigner`) for the design
step — the same pattern used by `BackboneGenerator` in `binder_bench/generate.py`.
To add a custom backend, subclass it and register it in the factory function.

### The abstract base class

```python
# binder_bench/design.py  (abridged)
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List
from binder_bench.config import DesignConfig

class SequenceDesigner(ABC):
    """Abstract base class for sequence design backends."""

    @abstractmethod
    def design(
        self,
        backbones: List[Path],
        config: DesignConfig,
    ) -> Dict[Path, List[str]]:
        """Return a mapping of backbone path -> list of designed sequences.

        Parameters
        ----------
        backbones:
            PDB files produced by the generate step.
        config:
            Design configuration (n_seqs_per_backbone, temperature, seed, …).

        Returns
        -------
        Dict[Path, List[str]]
            One-letter amino-acid sequences keyed by backbone path.
        """
```

### A minimal custom backend (20 lines)

The example below implements a backend that calls a hypothetical REST API:

```python
# my_project/api_designer.py
import requests
from pathlib import Path
from typing import Dict, List

from binder_bench.design import SequenceDesigner
from binder_bench.config import DesignConfig


class APISequenceDesigner(SequenceDesigner):
    """Send backbone PDB content to an external design API."""

    API_URL = "https://my-design-service.example.com/design"

    def design(
        self,
        backbones: List[Path],
        config: DesignConfig,
    ) -> Dict[Path, List[str]]:
        results: Dict[Path, List[str]] = {}
        for pdb_path in backbones:
            payload = {
                "pdb": pdb_path.read_text(),
                "n_seqs": config.n_seqs_per_backbone,
                "temperature": config.temperature,
                "seed": config.seed,
            }
            response = requests.post(self.API_URL, json=payload, timeout=120)
            response.raise_for_status()
            results[pdb_path] = response.json()["sequences"]
        return results
```

### Registering the backend

In `binder_bench/design.py`, extend the factory function:

```python
from my_project.api_designer import APISequenceDesigner

def get_designer(config: DesignConfig) -> SequenceDesigner:
    if config.backend == "stub":
        return StubDesigner()
    if config.backend == "proteinmpnn":
        return ProteinMPNNDesigner()
    if config.backend == "api":          # <-- new entry
        return APISequenceDesigner()
    raise ValueError(f"Unknown design backend: {config.backend!r}")
```

Then set `design.backend: "api"` in your YAML config and run as normal. The
rest of the pipeline (fold, filter, score) is unchanged.

---

**Next steps:** Cookbook 4 shows how to sweep `n_seqs_per_backbone` and
`temperature` with ProteinMPNN to build a success-rate vs. compute curve.
