"""Typer CLI for binder-bench."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from binder_bench.config import (
    BinderBenchConfig,
    DesignConfig,
    FilterConfig,
    FoldConfig,
    GenerateConfig,
    OutputConfig,
    ScoreConfig,
)
from binder_bench.pipeline import run_pipeline

try:
    from rich.console import Console
    from rich.table import Table

    _rich_available = True
    _console = Console()
except ImportError:
    _rich_available = False
    _console = None  # type: ignore[assignment]

app = typer.Typer(help="binder-bench: reproducible protein-binder design benchmark")

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _print_summary(results: dict) -> None:  # type: ignore[type-arg]
    """Print a summary table of pipeline results."""
    n_pass = results.get("n_pass", 0)
    n_total = results.get("n_total", 0)
    success_rate = results.get("success_rate", 0.0)
    best_design = results.get("best_design", "N/A")
    mean_plddt = results.get("mean_pLDDT", float("nan"))

    if _rich_available and _console is not None:
        table = Table(title="binder-bench Summary", show_header=True, header_style="bold cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Designs passing filters", f"{n_pass} / {n_total}")
        table.add_row("Success rate", f"{success_rate:.1%}")
        table.add_row("Best design", str(best_design))
        table.add_row("Mean pLDDT", f"{mean_plddt:.2f}")
        _console.print(table)
    else:
        typer.echo("")
        typer.echo(typer.style("=== binder-bench Summary ===", fg=typer.colors.CYAN, bold=True))
        typer.echo(f"  Designs passing filters : {n_pass} / {n_total}")
        typer.echo(f"  Success rate            : {success_rate:.1%}")
        typer.echo(f"  Best design             : {best_design}")
        typer.echo(f"  Mean pLDDT              : {mean_plddt:.2f}")
        typer.echo("")


@app.command()
def run(
    config_path: Path = typer.Argument(..., help="Path to YAML config file"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Override output directory"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override random seed for all stages"),
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help="Override all backends: stub|proteinmpnn|esmfold_api",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """Run the full binder design and evaluation pipeline."""
    _setup_logging(verbose)

    if not config_path.exists():
        typer.echo(
            typer.style(f"Error: config file not found: {config_path}", fg=typer.colors.RED, bold=True),
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        config = BinderBenchConfig.from_yaml(config_path)
        logger.debug("Loaded config from %s", config_path)
    except Exception as exc:
        typer.echo(
            typer.style(f"Error loading config: {exc}", fg=typer.colors.RED, bold=True),
            err=True,
        )
        raise typer.Exit(code=1)

    # Apply CLI overrides
    if output_dir is not None:
        config.output.output_dir = output_dir
        logger.debug("Overriding output_dir -> %s", output_dir)

    if seed is not None:
        config.generate.seed = seed
        config.design.seed = seed
        config.fold.seed = seed
        logger.debug("Overriding seed -> %d", seed)

    if backend is not None:
        _valid_backends = {"stub", "proteinmpnn", "esmfold_api", "rfdiffusion"}
        if backend not in _valid_backends:
            typer.echo(
                typer.style(
                    f"Error: unknown backend '{backend}'. Choose from: {', '.join(sorted(_valid_backends))}",
                    fg=typer.colors.RED,
                    bold=True,
                ),
                err=True,
            )
            raise typer.Exit(code=1)

        # Apply backend overrides where applicable
        if backend in {"stub", "rfdiffusion"}:
            config.generate.backend = backend  # type: ignore[assignment]
        if backend in {"stub", "proteinmpnn"}:
            config.design.backend = backend  # type: ignore[assignment]
        if backend in {"stub", "esmfold_api"}:
            config.fold.backend = backend  # type: ignore[assignment]
        logger.debug("Overriding backends -> %s", backend)

    typer.echo(
        typer.style(
            f"Starting pipeline for target: {config.target_name}",
            fg=typer.colors.GREEN,
            bold=True,
        )
    )
    logger.info("Pipeline config: %s", config.model_dump())

    try:
        pipeline_run = run_pipeline(config)
    except Exception as exc:
        logger.exception("Pipeline failed")
        typer.echo(
            typer.style(f"Pipeline error: {exc}", fg=typer.colors.RED, bold=True),
            err=True,
        )
        raise typer.Exit(code=1)

    # Normalise stats keys for _print_summary
    raw = dict(pipeline_run.stats) if pipeline_run.stats else {}
    summary = {
        "n_pass": raw.get("n_pass", 0),
        "n_total": raw.get("n_total", 0),
        "success_rate": raw.get("success_rate", 0.0),
        "best_design": raw.get("best_design_id", "N/A"),
        "mean_pLDDT": raw.get("mean_plddt_pass", float("nan")),
    }
    _print_summary(summary)
    typer.echo(typer.style("Pipeline completed successfully.", fg=typer.colors.GREEN))
    raise typer.Exit(code=0)


@app.command()
def validate(
    config_path: Path = typer.Argument(..., help="Path to YAML config file"),
) -> None:
    """Validate a config file without running the pipeline."""
    if not config_path.exists():
        typer.echo(
            typer.style(f"Error: file not found: {config_path}", fg=typer.colors.RED, bold=True),
            err=True,
        )
        raise typer.Exit(code=1)

    try:
        config = BinderBenchConfig.from_yaml(config_path)
        typer.echo(typer.style("Config is valid.", fg=typer.colors.GREEN, bold=True))
        typer.echo(f"  target_name : {config.target_name}")
        typer.echo(f"  target_pdb  : {config.target_pdb}")
        typer.echo(f"  output_dir  : {config.output.output_dir}")
        typer.echo(f"  generate    : backend={config.generate.backend}, n_designs={config.generate.n_designs}")
        typer.echo(f"  design      : backend={config.design.backend}, n_seqs={config.design.n_seqs_per_backbone}")
        typer.echo(f"  fold        : backend={config.fold.backend}")
        typer.echo(
            f"  filter      : rmsd<={config.filter.rmsd_threshold}, pLDDT>={config.filter.plddt_threshold}"
        )
    except Exception as exc:
        typer.echo(
            typer.style(f"Config validation failed: {exc}", fg=typer.colors.RED, bold=True),
            err=True,
        )
        raise typer.Exit(code=1)


@app.command()
def init(
    output: Path = typer.Option(Path("config.yaml"), "--output", "-o", help="Output path for the config file"),
) -> None:
    """Write a default config.yaml with stub backends to the current directory."""
    if output.exists():
        overwrite = typer.confirm(
            typer.style(f"File '{output}' already exists. Overwrite?", fg=typer.colors.YELLOW),
            default=False,
        )
        if not overwrite:
            typer.echo("Aborted.")
            raise typer.Exit(code=0)

    default_config = BinderBenchConfig(
        target_name="my_target",
        target_pdb=Path("examples/target.pdb"),
        generate=GenerateConfig(
            backend="stub",
            n_designs=10,
            seed=42,
            stub_dir=Path("examples/backbones"),
        ),
        design=DesignConfig(
            backend="stub",
            n_seqs_per_backbone=8,
            temperature=0.1,
            seed=42,
            stub_dir=Path("examples/sequences"),
        ),
        fold=FoldConfig(
            backend="stub",
            stub_dir=Path("examples/structures"),
        ),
        filter=FilterConfig(
            rmsd_threshold=2.0,
            plddt_threshold=70.0,
            min_binder_length=50,
            max_binder_length=150,
        ),
        score=ScoreConfig(
            contact_distance=8.0,
            min_interface_contacts=10,
        ),
        output=OutputConfig(
            output_dir=Path("output"),
            scoreboard_format="parquet",
            save_passing_structures=True,
            plot_success_rate=True,
        ),
    )

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        default_config.save_yaml(output)
        typer.echo(typer.style(f"Default config written to: {output}", fg=typer.colors.GREEN, bold=True))
        typer.echo("Edit the file to set your target_pdb and desired backends, then run:")
        typer.echo(f"  binder-bench run {output}")
    except Exception as exc:
        typer.echo(
            typer.style(f"Error writing config: {exc}", fg=typer.colors.RED, bold=True),
            err=True,
        )
        raise typer.Exit(code=1)


@app.command(name="score-only")
def score_only(
    output_dir: Path = typer.Argument(..., help="Existing output directory containing results parquet/csv"),
    rmsd_threshold: float = typer.Option(2.0, "--rmsd-threshold", help="RMSD threshold for pass/fail (Angstroms)"),
    plddt_threshold: float = typer.Option(70.0, "--plddt-threshold", help="pLDDT threshold for pass/fail (0-100)"),
) -> None:
    """Re-score an existing output directory with new filter thresholds."""
    if not output_dir.exists() or not output_dir.is_dir():
        typer.echo(
            typer.style(f"Error: output directory not found: {output_dir}", fg=typer.colors.RED, bold=True),
            err=True,
        )
        raise typer.Exit(code=1)

    # Locate the scoreboard file
    parquet_files = list(output_dir.glob("*.parquet"))
    csv_files = list(output_dir.glob("*.csv"))

    scoreboard_path: Optional[Path] = None
    use_parquet = False

    if parquet_files:
        scoreboard_path = parquet_files[0]
        use_parquet = True
        logger.debug("Found parquet scoreboard: %s", scoreboard_path)
    elif csv_files:
        scoreboard_path = csv_files[0]
        logger.debug("Found CSV scoreboard: %s", scoreboard_path)
    else:
        typer.echo(
            typer.style(
                f"Error: no .parquet or .csv scoreboard found in {output_dir}",
                fg=typer.colors.RED,
                bold=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo(f"Loading scoreboard from: {scoreboard_path}")

    try:
        if use_parquet:
            try:
                import pandas as pd  # type: ignore

                df = pd.read_parquet(scoreboard_path)
            except ImportError:
                typer.echo(
                    typer.style(
                        "Error: pandas/pyarrow is required to read parquet files. "
                        "Install with: pip install pandas pyarrow",
                        fg=typer.colors.RED,
                        bold=True,
                    ),
                    err=True,
                )
                raise typer.Exit(code=1)
        else:
            import pandas as pd  # type: ignore

            df = pd.read_csv(scoreboard_path)
    except Exception as exc:
        typer.echo(
            typer.style(f"Error reading scoreboard: {exc}", fg=typer.colors.RED, bold=True),
            err=True,
        )
        raise typer.Exit(code=1)

    required_cols = {"rmsd", "plddt"}
    missing = required_cols - set(df.columns)
    if missing:
        typer.echo(
            typer.style(
                f"Error: scoreboard is missing required columns: {missing}",
                fg=typer.colors.RED,
                bold=True,
            ),
            err=True,
        )
        raise typer.Exit(code=1)

    mask = (df["rmsd"] <= rmsd_threshold) & (df["plddt"] >= plddt_threshold)
    n_total = len(df)
    n_pass = int(mask.sum())
    success_rate = n_pass / n_total if n_total > 0 else 0.0
    mean_plddt = float(df.loc[mask, "plddt"].mean()) if n_pass > 0 else float("nan")

    best_design = "N/A"
    if n_pass > 0:
        best_idx = df.loc[mask, "plddt"].idxmax()
        best_design = str(df.loc[best_idx, "design_id"] if "design_id" in df.columns else best_idx)

    results = {
        "n_pass": n_pass,
        "n_total": n_total,
        "success_rate": success_rate,
        "best_design": best_design,
        "mean_pLDDT": mean_plddt,
    }

    typer.echo(
        typer.style(
            f"Re-scoring with RMSD <= {rmsd_threshold} A, pLDDT >= {plddt_threshold}",
            fg=typer.colors.CYAN,
        )
    )
    _print_summary(results)

    # Write updated summary JSON alongside scoreboard
    summary_path = output_dir / "rescored_summary.json"
    try:
        with open(summary_path, "w") as fh:
            json.dump(
                {
                    "rmsd_threshold": rmsd_threshold,
                    "plddt_threshold": plddt_threshold,
                    **results,
                },
                fh,
                indent=2,
            )
        typer.echo(f"Summary written to: {summary_path}")
    except Exception as exc:
        logger.warning("Could not write summary JSON: %s", exc)


if __name__ == "__main__":
    app()
