# examples/

Pre-computed stub data for running binder-bench without a GPU.

## Contents

| Path | Description |
|------|-------------|
| `backbones/stub_backbone_001-003.pdb` | Three 65-residue CA-only alpha-helix backbones targeting the IL-6 receptor binding interface (PDB 1IL6). Stub pLDDT = 75.0. |
| `sequences/stub_backbone_001-003_seqs.txt` | Two designed sequences per backbone (one sequence per line, one-letter amino acid code). |
| `structures/stub_backbone_{NNN}_seq{MMM}.pdb` | Six stub folded structures (2 per backbone). These are helix-geometry CA-only PDBs with B-factor = 82.0 (stub pLDDT). |
| `target_il6.pdb` | 50-residue CA-only stub representing the IL-6R epitope region (chain B). |
| `config_quickstart.yaml` | Ready-to-run YAML config using all stub backends. |

## Naming Convention

- **Backbones**: `stub_backbone_{NNN}.pdb` where NNN is zero-padded index (001, 002, 003)
- **Sequences**: `stub_backbone_{NNN}_seqs.txt` — one sequence per line, matching its backbone
- **Structures**: `stub_backbone_{NNN}_seq{MMM}.pdb` where MMM is zero-padded sequence index (000, 001)

## Running the Quickstart

```bash
binder-bench run examples/config_quickstart.yaml -v
```

This runs the full generate→design→fold→filter→score pipeline using stub backends
and writes results to `output/`.
