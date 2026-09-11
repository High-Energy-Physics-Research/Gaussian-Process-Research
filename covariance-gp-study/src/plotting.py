"""Plotting helpers for article-style covariance inference figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Array = np.ndarray


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save a matplotlib figure as both PDF and PNG."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")


def summarize_by_method(df: pd.DataFrame, value: str) -> pd.DataFrame:
    """Summarize Monte Carlo distributions by M and method."""
    q = df.groupby(["M", "method"])[value].quantile([0.025, 0.16, 0.5, 0.84, 0.975]).unstack()
    q.columns = ["p02_5", "p16", "median", "p84", "p97_5"]
    stats = df.groupby(["M", "method"])[value].agg(["mean", "std"]).join(q).reset_index()
    return stats


def plot_frobenius_vs_M(
    df: pd.DataFrame,
    output_stem: Path,
    n: int | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot median relative Frobenius error with 16-84% bands."""
    stats = summarize_by_method(df, "frobenius_error")
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    for method, sub in stats.groupby("method"):
        sub = sub.sort_values("M")
        x = sub["M"].to_numpy(dtype=float)
        ax.plot(x, sub["median"], marker="o", label=method)
        ax.fill_between(x, sub["p16"], sub["p84"], alpha=0.2)
    if n is not None:
        ax.axvline(n, color="0.4", linestyle="--", linewidth=1.0, label="M = n")
    ax.set_xscale("log")
    ax.set_xlabel("M")
    ax.set_ylabel("relative Frobenius error")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False)
    save_figure(fig, output_stem)
    return fig


def plot_parameter_vs_M(
    df: pd.DataFrame,
    parameter: str,
    output_stem: Path,
    true_value: float | None = None,
    method: str | None = None,
) -> plt.Figure:
    """Plot fitted parameter recovery versus M."""
    col = f"fitted_{parameter}"
    data = df.copy()
    if method is not None:
        data = data[data["method"] == method]
    stats = data.groupby("M")[col].quantile([0.16, 0.5, 0.84]).unstack().reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    x = stats["M"].to_numpy(dtype=float)
    ax.plot(x, stats[0.5], marker="o")
    ax.fill_between(x, stats[0.16], stats[0.84], alpha=0.2)
    if true_value is not None:
        ax.axhline(true_value, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("M")
    ax.set_ylabel(f"{parameter} estimate")
    save_figure(fig, output_stem)
    return fig


def plot_gain_vs_M(df: pd.DataFrame, output_stem: Path) -> pd.DataFrame:
    """Plot statistical efficiency gain E_sample / E_RBF."""
    med = df.groupby(["M", "method"])["frobenius_error"].median().unstack()
    if "sample" not in med or "rbf" not in med:
        raise ValueError("Data must contain sample and rbf methods.")
    gain = (med["sample"] / med["rbf"]).rename("gain").reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    ax.plot(gain["M"], gain["gain"], marker="o")
    ax.axhline(1.0, color="0.4", linestyle="--", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("M")
    ax.set_ylabel("statistical efficiency gain")
    save_figure(fig, output_stem)
    return gain


def plot_covariance_heatmaps(
    matrices: dict[str, Array],
    output_stem: Path,
    title: str | None = None,
) -> plt.Figure:
    """Plot a row of covariance matrices or residuals."""
    ncols = len(matrices)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.0), constrained_layout=True)
    if ncols == 1:
        axes = [axes]
    for ax, (name, matrix) in zip(axes, matrices.items()):
        im = ax.imshow(matrix, aspect="equal")
        ax.set_title(name)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title:
        fig.suptitle(title)
    save_figure(fig, output_stem)
    return fig


def plot_cross_kernel_heatmap(df: pd.DataFrame, M: int, output_stem: Path) -> plt.Figure:
    """Plot the 3x3 median cross-kernel error matrix for a fixed M."""
    sub = df[(df["M"] == M) & (df["method"] != "sample")]
    pivot = sub.pivot_table(
        values="frobenius_error",
        index="true_family",
        columns="method",
        aggfunc="median",
    ).reindex(index=["diagonal", "cs", "rbf"], columns=["diagonal", "cs", "rbf"])
    fig, ax = plt.subplots(figsize=(4.8, 4.0), constrained_layout=True)
    im = ax.imshow(pivot.to_numpy(dtype=float))
    ax.set_xticks(range(3), pivot.columns)
    ax.set_yticks(range(3), pivot.index)
    ax.set_xlabel("fit")
    ax.set_ylabel("truth")
    ax.set_title(f"median relative Frobenius error, M={M}")
    for i in range(3):
        for j in range(3):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.3g}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax)
    save_figure(fig, output_stem)
    return fig
