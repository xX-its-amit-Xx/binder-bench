"""Generate summary plots for binder-bench results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    _MATPLOTLIB_AVAILABLE = False

from binder_bench.score import ScoreResult

logger = logging.getLogger(__name__)

_PASS_COLOR = "#2ca02c"   # green
_FAIL_COLOR = "#d62728"   # red


def _require_matplotlib(fn_name: str) -> bool:
    """Return True if matplotlib is available, else log a warning and return False."""
    if not _MATPLOTLIB_AVAILABLE:
        logger.warning(
            "matplotlib is not installed; skipping %s. "
            "Install it with: pip install matplotlib",
            fn_name,
        )
        return False
    return True


def plot_success_rate(
    df: pd.DataFrame,
    output_path: Path,
    title: str = "Design Success Rate",
) -> None:
    """Bar chart of passing vs failing designs, annotated with success rate.

    Parameters
    ----------
    df : pd.DataFrame
        Scoreboard DataFrame containing at least a ``passes_filter`` boolean column.
    output_path : Path
        Destination file for the saved figure (e.g. ``output/plots/success_rate.png``).
    title : str
        Figure title.
    """
    if not _require_matplotlib("plot_success_rate"):
        return

    n_total = len(df)
    n_pass = int(df["passes_filter"].sum()) if n_total > 0 else 0
    n_fail = n_total - n_pass
    success_rate = (n_pass / n_total * 100.0) if n_total > 0 else 0.0

    fig, ax = plt.subplots(figsize=(6, 5))

    bars = ax.bar(
        ["Passing", "Failing"],
        [n_pass, n_fail],
        color=[_PASS_COLOR, _FAIL_COLOR],
        width=0.5,
        edgecolor="white",
    )

    # Annotate each bar with its count
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(n_total * 0.01, 0.3),
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    # Annotate with success rate
    ax.text(
        0.98,
        0.97,
        f"Success rate: {success_rate:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="#333333",
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Number of designs", fontsize=12)
    ax.set_ylim(0, max(n_pass, n_fail, 1) * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved success-rate plot to %s", output_path)


def plot_rmsd_plddt_scatter(
    df: pd.DataFrame,
    output_path: Path,
    rmsd_threshold: float = 2.0,
    plddt_threshold: float = 70.0,
) -> None:
    """Scatter plot of RMSD vs pLDDT, coloured by filter pass/fail status.

    Parameters
    ----------
    df : pd.DataFrame
        Scoreboard DataFrame containing ``rmsd``, ``plddt``, and
        ``passes_filter`` columns.
    output_path : Path
        Destination file for the saved figure.
    rmsd_threshold : float
        RMSD threshold line drawn on the x-axis.
    plddt_threshold : float
        pLDDT threshold line drawn on the y-axis.
    """
    if not _require_matplotlib("plot_rmsd_plddt_scatter"):
        return

    passing = df[df["passes_filter"]]
    failing = df[~df["passes_filter"]]

    fig, ax = plt.subplots(figsize=(7, 6))

    if len(failing) > 0:
        ax.scatter(
            failing["rmsd"],
            failing["plddt"],
            c=_FAIL_COLOR,
            alpha=0.7,
            s=50,
            label="Fail",
            edgecolors="none",
            zorder=2,
        )
    if len(passing) > 0:
        ax.scatter(
            passing["rmsd"],
            passing["plddt"],
            c=_PASS_COLOR,
            alpha=0.7,
            s=50,
            label="Pass",
            edgecolors="none",
            zorder=3,
        )

    ax.axvline(
        x=rmsd_threshold,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"RMSD threshold ({rmsd_threshold} Å)",
        zorder=1,
    )
    ax.axhline(
        y=plddt_threshold,
        color="#888888",
        linestyle=":",
        linewidth=1.2,
        label=f"pLDDT threshold ({plddt_threshold})",
        zorder=1,
    )

    ax.set_xlabel("RMSD (Å)", fontsize=12)
    ax.set_ylabel("pLDDT", fontsize=12)
    ax.set_title("RMSD vs pLDDT", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved RMSD/pLDDT scatter plot to %s", output_path)


def plot_threshold_sweep(
    df: pd.DataFrame,
    output_path: Path,
    rmsd_range: Tuple[float, float, float] = (0.5, 5.0, 0.5),
    plddt_range: Tuple[float, float, float] = (50.0, 95.0, 5.0),
) -> None:
    """2-D heatmap of success rate over a grid of RMSD and pLDDT thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        Scoreboard DataFrame containing ``rmsd`` and ``plddt`` columns.
    output_path : Path
        Destination file for the saved figure.
    rmsd_range : tuple(start, stop, step)
        ``np.arange`` arguments for the RMSD threshold axis.
    plddt_range : tuple(start, stop, step)
        ``np.arange`` arguments for the pLDDT threshold axis.
    """
    if not _require_matplotlib("plot_threshold_sweep"):
        return

    rmsd_thresholds = np.arange(*rmsd_range)
    plddt_thresholds = np.arange(*plddt_range)

    n_total = len(df)

    grid = np.zeros((len(plddt_thresholds), len(rmsd_thresholds)), dtype=float)

    if n_total > 0:
        rmsd_vals = df["rmsd"].to_numpy()
        plddt_vals = df["plddt"].to_numpy()
        for i, p_thresh in enumerate(plddt_thresholds):
            for j, r_thresh in enumerate(rmsd_thresholds):
                n_pass = int(
                    np.sum((rmsd_vals <= r_thresh) & (plddt_vals >= p_thresh))
                )
                grid[i, j] = n_pass / n_total * 100.0

    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap="YlGn",
        vmin=0.0,
        vmax=100.0,
        extent=[
            rmsd_thresholds[0] - (rmsd_range[2] / 2),
            rmsd_thresholds[-1] + (rmsd_range[2] / 2),
            plddt_thresholds[0] - (plddt_range[2] / 2),
            plddt_thresholds[-1] + (plddt_range[2] / 2),
        ],
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate (%)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xlabel("RMSD threshold (Å)", fontsize=12)
    ax.set_ylabel("pLDDT threshold", fontsize=12)
    ax.set_title("Threshold Sweep: Success Rate (%)", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(axis="both", labelsize=10)

    # Annotate cells with the success-rate value
    for i in range(len(plddt_thresholds)):
        for j in range(len(rmsd_thresholds)):
            ax.text(
                rmsd_thresholds[j],
                plddt_thresholds[i],
                f"{grid[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if grid[i, j] < 70 else "white",
            )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved threshold-sweep heatmap to %s", output_path)


def plot_score_distribution(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Overlaid histogram of pLDDT for passing and failing designs.

    Parameters
    ----------
    df : pd.DataFrame
        Scoreboard DataFrame containing ``plddt`` and ``passes_filter`` columns.
    output_path : Path
        Destination file for the saved figure.
    """
    if not _require_matplotlib("plot_score_distribution"):
        return

    passing = df[df["passes_filter"]]["plddt"].to_numpy()
    failing = df[~df["passes_filter"]]["plddt"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))

    bins = np.linspace(0, 100, 26)

    if len(failing) > 0:
        ax.hist(
            failing,
            bins=bins,
            alpha=0.7,
            color=_FAIL_COLOR,
            label=f"Fail (n={len(failing)})",
            edgecolor="white",
            linewidth=0.5,
        )
    if len(passing) > 0:
        ax.hist(
            passing,
            bins=bins,
            alpha=0.7,
            color=_PASS_COLOR,
            label=f"Pass (n={len(passing)})",
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xlabel("pLDDT", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("pLDDT Score Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved score-distribution histogram to %s", output_path)


def generate_all_plots(
    df: pd.DataFrame,
    output_dir: Path,
    config_filter=None,
    config_score=None,
) -> None:
    """Generate all summary plots and save them under ``output_dir/plots/``.

    Parameters
    ----------
    df : pd.DataFrame
        Scoreboard DataFrame as produced by
        :func:`binder_bench.score.results_to_scoreboard`.
    output_dir : Path
        Root output directory; a ``plots/`` sub-directory will be created.
    config_filter : FilterConfig, optional
        When provided, RMSD and pLDDT thresholds are read from this config
        for the scatter plot.
    config_score : ScoreConfig, optional
        Reserved for future use; currently unused.
    """
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Resolve thresholds from config when available
    rmsd_threshold = 2.0
    plddt_threshold = 70.0
    if config_filter is not None:
        rmsd_threshold = float(getattr(config_filter, "rmsd_threshold", rmsd_threshold))
        plddt_threshold = float(getattr(config_filter, "plddt_threshold", plddt_threshold))

    plot_success_rate(
        df=df,
        output_path=plots_dir / "success_rate.png",
    )

    plot_rmsd_plddt_scatter(
        df=df,
        output_path=plots_dir / "rmsd_plddt_scatter.png",
        rmsd_threshold=rmsd_threshold,
        plddt_threshold=plddt_threshold,
    )

    plot_threshold_sweep(
        df=df,
        output_path=plots_dir / "threshold_sweep.png",
    )

    plot_score_distribution(
        df=df,
        output_path=plots_dir / "score_distribution.png",
    )

    logger.info("All plots saved to %s", plots_dir)
