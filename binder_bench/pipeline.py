"""Orchestrate the full generate -> design -> fold -> filter -> score pipeline."""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from binder_bench.config import BinderBenchConfig, OutputConfig
from binder_bench.generate import run_generate
from binder_bench.design import run_design
from binder_bench.fold import run_fold
from binder_bench.filter import run_filter, results_to_dataframe as filter_to_df
from binder_bench.score import run_score, results_to_scoreboard, compute_aggregate_stats

logger = logging.getLogger(__name__)


@dataclass
class PipelineRun:
    config: BinderBenchConfig
    run_id: str  # datetime-based: YYYYMMDD_HHMMSS
    output_dir: Path
    backbones: list[Path] = field(default_factory=list)
    sequences: dict = field(default_factory=dict)  # backbone -> [seqs]
    folded: dict = field(default_factory=dict)  # design_id -> path
    filter_results: list = field(default_factory=list)
    score_results: list = field(default_factory=list)
    scoreboard: Optional[pd.DataFrame] = None
    stats: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0


class Pipeline:
    def __init__(self, config: BinderBenchConfig) -> None:
        self.config = config

    def run(self) -> PipelineRun:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.config.output.output_dir
        run = PipelineRun(
            config=self.config,
            run_id=run_id,
            output_dir=output_dir,
        )

        pipeline_start = time.monotonic()

        # Create a run-specific working directory for intermediate files
        run_dir = self.config.output.output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        fold_dir = run_dir / "folded"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate backbones
        logger.info("[%s] Step 1/5: Generating backbones...", run_id)
        step_start = time.monotonic()
        try:
            run.backbones = run_generate(self.config.generate)
            logger.info(
                "[%s] Generate complete: %d backbones in %.1fs",
                run_id,
                len(run.backbones),
                time.monotonic() - step_start,
            )
        except Exception:
            logger.exception("[%s] Generate step failed", run_id)
            run.elapsed_seconds = time.monotonic() - pipeline_start
            return run

        # Step 2: Design sequences
        logger.info("[%s] Step 2/5: Designing sequences...", run_id)
        step_start = time.monotonic()
        try:
            run.sequences = run_design(run.backbones, self.config.design)
            n_seqs = sum(len(v) for v in run.sequences.values())
            logger.info(
                "[%s] Design complete: %d sequences across %d backbones in %.1fs",
                run_id,
                n_seqs,
                len(run.sequences),
                time.monotonic() - step_start,
            )
        except Exception:
            logger.exception("[%s] Design step failed", run_id)
            run.elapsed_seconds = time.monotonic() - pipeline_start
            return run

        # Step 3: Fold sequences
        logger.info("[%s] Step 3/5: Folding sequences...", run_id)
        step_start = time.monotonic()
        try:
            run.folded = run_fold(run.sequences, self.config.fold, fold_dir)
            logger.info(
                "[%s] Fold complete: %d structures in %.1fs",
                run_id,
                len(run.folded),
                time.monotonic() - step_start,
            )
        except Exception:
            logger.exception("[%s] Fold step failed", run_id)
            run.elapsed_seconds = time.monotonic() - pipeline_start
            return run

        # Step 4: Filter structures — build design_map from sequences + folded dicts
        logger.info("[%s] Step 4/5: Filtering structures...", run_id)
        step_start = time.monotonic()
        try:
            design_map: dict[str, tuple[Path, Path]] = {}
            for backbone_path, seqs in run.sequences.items():
                for idx in range(len(seqs)):
                    design_id = f"{backbone_path.stem}_seq{idx:03d}"
                    if design_id in run.folded:
                        design_map[design_id] = (backbone_path, run.folded[design_id])
            run.filter_results = run_filter(design_map, self.config.filter)
            n_passing = sum(1 for r in run.filter_results if r.passes)
            logger.info(
                "[%s] Filter complete: %d/%d passing in %.1fs",
                run_id,
                n_passing,
                len(run.filter_results),
                time.monotonic() - step_start,
            )
        except Exception:
            logger.exception("[%s] Filter step failed", run_id)
            run.elapsed_seconds = time.monotonic() - pipeline_start
            return run

        # Step 5: Score structures
        logger.info("[%s] Step 5/5: Scoring structures...", run_id)
        step_start = time.monotonic()
        try:
            run.score_results = run_score(
                run.filter_results, self.config.target_pdb, self.config.score
            )
            run.scoreboard = results_to_scoreboard(run.score_results)
            run.stats = compute_aggregate_stats(run.scoreboard)
            logger.info(
                "[%s] Score complete: %d entries scored in %.1fs",
                run_id,
                len(run.score_results),
                time.monotonic() - step_start,
            )
        except Exception:
            logger.exception("[%s] Score step failed", run_id)
            run.elapsed_seconds = time.monotonic() - pipeline_start
            return run

        run.elapsed_seconds = time.monotonic() - pipeline_start
        logger.info(
            "[%s] Pipeline complete in %.1fs", run_id, run.elapsed_seconds
        )

        self._save_outputs(run)
        return run

    def _save_outputs(self, run: PipelineRun) -> None:
        run_dir = run.output_dir / run.run_id
        run_dir.mkdir(parents=True, exist_ok=True)  # idempotent — already created in run()
        logger.info("Saving outputs to %s", run_dir)

        # Save scoreboard
        if run.scoreboard is not None and not run.scoreboard.empty:
            fmt = self.config.output.scoreboard_format
            if fmt == "parquet":
                scoreboard_path = run_dir / "scoreboard.parquet"
                run.scoreboard.to_parquet(scoreboard_path, index=False)
            else:
                scoreboard_path = run_dir / "scoreboard.csv"
                run.scoreboard.to_csv(scoreboard_path, index=False)
            logger.info("Scoreboard saved to %s", scoreboard_path)
        else:
            logger.warning("No scoreboard to save (empty or None).")

        # Save stats JSON
        stats_path = run_dir / "stats.json"
        try:
            with open(stats_path, "w") as fh:
                json.dump(run.stats, fh, indent=2, default=str)
            logger.info("Stats saved to %s", stats_path)
        except Exception:
            logger.exception("Failed to save stats.json")

        # Copy passing structures
        if self.config.output.save_passing_structures:
            passing_dir = run_dir / "passing"
            passing_dir.mkdir(parents=True, exist_ok=True)
            passing_ids = {
                r.design_id
                for r in run.filter_results
                if r.passes
            }
            copied = 0
            for design_id, pdb_path in run.folded.items():
                if design_id in passing_ids:
                    src = Path(pdb_path)
                    if src.exists():
                        dest = passing_dir / src.name
                        shutil.copy2(src, dest)
                        copied += 1
            logger.info("Copied %d passing structures to %s", copied, passing_dir)

        # Plot success rate
        if self.config.output.plot_success_rate and run.scoreboard is not None:
            try:
                from binder_bench.plots import generate_all_plots  # type: ignore

                generate_all_plots(
                    run.scoreboard,
                    run_dir,
                    config_filter=self.config.filter,
                    config_score=self.config.score,
                )
                logger.info("Plots saved to %s/plots/", run_dir)
            except ImportError:
                logger.debug(
                    "binder_bench.plots not available; skipping success rate plot."
                )
            except Exception:
                logger.exception("Failed to generate success rate plot")


def run_pipeline(config: BinderBenchConfig) -> PipelineRun:
    """Convenience wrapper: construct a Pipeline and run it end to end."""
    return Pipeline(config).run()
