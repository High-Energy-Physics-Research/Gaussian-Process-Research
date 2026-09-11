"""Experiment drivers used by the notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_: Any):
        return iterable

from .covariance_models import (
    covariance_from_family,
    positions,
    validate_covariance,
)
from .fitting import fit_covariance_model, negative_log_likelihood
from .metrics import (
    max_absolute_matrix_error,
    mean_absolute_matrix_error,
    relative_frobenius_error,
)
from .plotting import (
    plot_covariance_heatmaps,
    plot_cross_kernel_heatmap,
    plot_frobenius_vs_M,
    plot_gain_vs_M,
    plot_parameter_vs_M,
)
from .simulation import generate_realizations, repetition_seed, sample_covariance


Array = np.ndarray


def project_root() -> Path:
    """Return the project root from inside src/."""
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_output_dirs(root: Path) -> None:
    """Create the expected results and plots directories."""
    for rel in [
        "results/phase1/tables",
        "results/phase1/matrices",
        "results/phase1/raw",
        "results/phase2/tables",
        "results/phase2/matrices",
        "results/phase2/raw",
        "plots/phase1",
        "plots/phase2",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def flat_fit_record(prefix: str, params: dict[str, float]) -> dict[str, float]:
    """Flatten fit parameters into fitted_sigma, fitted_ell, ... columns."""
    out = {f"{prefix}_sigma": np.nan, f"{prefix}_ell": np.nan, f"{prefix}_noise": np.nan, f"{prefix}_rho": np.nan}
    for key, value in params.items():
        name = "noise" if key == "sigma_n" else key
        out[f"{prefix}_{name}"] = float(value)
    return out


def sample_diagnostics(Sigma: Array) -> dict[str, float | int | bool]:
    """Return rank, condition number, and eigenvalue diagnostics."""
    info = validate_covariance(Sigma)
    return {
        "sample_rank": info.rank,
        "sample_condition_number": info.condition_number,
        "sample_min_eigenvalue": info.min_eigenvalue,
        "sample_positive_semidefinite": info.positive_semidefinite,
    }


def append_csv(records: list[dict[str, Any]], path: Path) -> None:
    """Append records to CSV, writing a header only for a new file."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def run_phase1(config: dict[str, Any], root: Path | None = None, overwrite: bool = True) -> pd.DataFrame:
    """Run Phase 1 statistical-efficiency experiments."""
    root = project_root() if root is None else Path(root)
    ensure_output_dirs(root)
    x = positions(config["n_bins"], config["x_min"], config["x_max"])
    n = int(config["n_bins"])
    Sigma_true = covariance_from_family("rbf", n, config["truth"], x=x)
    table_path = root / "results/phase1/tables/frobenius_vs_M.csv"
    if overwrite and table_path.exists():
        table_path.unlink()

    all_records: list[dict[str, Any]] = []
    iterator = tqdm(config["M_values"], desc="Phase 1 M")
    for m_index, M in enumerate(iterator):
        batch: list[dict[str, Any]] = []
        for repetition in range(int(config["n_repetitions"])):
            seed = repetition_seed(int(config["seed_base"]), m_index, repetition)
            rng = np.random.default_rng(seed)
            Y = generate_realizations(Sigma_true, int(M), rng)

            Sigma_sample = sample_covariance(Y)
            sample_record = {
                "M": int(M),
                "method": "sample",
                "repetition": repetition,
                "seed": seed,
                "frobenius_error": relative_frobenius_error(Sigma_sample, Sigma_true),
                "log_likelihood": -negative_log_likelihood(Y, Sigma_sample, include_constant=True)
                if int(M) > n
                else np.nan,
                "success": True,
                **sample_diagnostics(Sigma_sample),
                **flat_fit_record("fitted", {}),
            }
            batch.append(sample_record)

            for fit_index, fit_family in enumerate(["diagonal", "cs", "rbf"]):
                fit = fit_covariance_model(Y, fit_family, x=x, seed=seed + 1 + fit_index)
                batch.append(
                    {
                        "M": int(M),
                        "method": fit_family,
                        "repetition": repetition,
                        "seed": seed,
                        "frobenius_error": relative_frobenius_error(fit.Sigma_hat, Sigma_true),
                        "log_likelihood": fit.log_likelihood,
                        "success": fit.optimizer_success,
                        "optimizer_message": fit.message,
                        "number_iterations": fit.number_iterations,
                        **flat_fit_record("fitted", fit.parameters),
                    }
                )
        append_csv(batch, table_path)
        all_records.extend(batch)

    df = pd.DataFrame(all_records)
    summary = _write_summary(df, root / "results/phase1/tables/frobenius_vs_M_summary.csv")
    plot_frobenius_vs_M(df, root / "plots/phase1/frobenius_vs_M", n=n, title="Phase 1")
    plot_parameter_vs_M(df[df["method"] == "rbf"], "ell", root / "plots/phase1/ell_recovery_vs_M", true_value=config["truth"]["ell"])
    gain = plot_gain_vs_M(df, root / "plots/phase1/statistical_gain_vs_M")
    gain.to_csv(root / "results/phase1/tables/statistical_gain_vs_M.csv", index=False)
    _save_phase1_example_matrices(config, root, Sigma_true, x)
    return df


def _write_summary(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    summary = (
        df.groupby(["M", "method"])["frobenius_error"]
        .agg(
            mean="mean",
            std="std",
            median="median",
            p16=lambda s: s.quantile(0.16),
            p84=lambda s: s.quantile(0.84),
            p02_5=lambda s: s.quantile(0.025),
            p97_5=lambda s: s.quantile(0.975),
        )
        .reset_index()
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return summary


def _save_phase1_example_matrices(config: dict[str, Any], root: Path, Sigma_true: Array, x: Array) -> None:
    for M in [10, 50, 500]:
        if M not in config["M_values"]:
            continue
        seed = repetition_seed(int(config["seed_base"]), int(M), 999)
        rng = np.random.default_rng(seed)
        Y = generate_realizations(Sigma_true, M, rng)
        Sigma_sample = sample_covariance(Y)
        fit = fit_covariance_model(Y, "rbf", x=x, seed=seed + 1)
        matrices = {
            "Sigma_true": Sigma_true,
            "Sigma_sample": Sigma_sample,
            "Sigma_RBF_fit": fit.Sigma_hat,
            "Sigma_RBF_fit_minus_true": fit.Sigma_hat - Sigma_true,
        }
        for name, matrix in matrices.items():
            np.save(root / f"results/phase1/matrices/M{M:03d}_{name}.npy", matrix)
        plot_covariance_heatmaps(matrices, root / f"plots/phase1/heatmaps_M{M:03d}", title=f"M={M}")


def run_phase2_recovery(config: dict[str, Any], root: Path | None = None, overwrite: bool = True) -> pd.DataFrame:
    """Run Phase 2A correct-model recovery experiments."""
    root = project_root() if root is None else Path(root)
    ensure_output_dirs(root)
    n = int(config["n_bins"])
    x = positions(n, config["x_min"], config["x_max"])
    table_path = root / "results/phase2/tables/parameter_recovery.csv"
    if overwrite and table_path.exists():
        table_path.unlink()

    scenarios = recovery_scenarios()
    records: list[dict[str, Any]] = []
    for case_index, case in enumerate(tqdm(scenarios, desc="Phase 2A cases")):
        family, params = case["family"], case["params"]
        Sigma_true = covariance_from_family(family, n, params, x=x)
        for m_index, M in enumerate(config["M_values"]):
            batch: list[dict[str, Any]] = []
            for repetition in range(int(config["n_repetitions"])):
                seed = repetition_seed(int(config["seed_base"]), case_index, m_index, repetition)
                rng = np.random.default_rng(seed)
                Y = generate_realizations(Sigma_true, int(M), rng)
                fit = fit_covariance_model(Y, family, x=x, seed=seed + 1)
                record = {
                    "true_family": family,
                    "fitted_family": family,
                    "method": family,
                    "true_parameters": repr(params),
                    "M": int(M),
                    "repetition": repetition,
                    "seed": seed,
                    "frobenius_error": relative_frobenius_error(fit.Sigma_hat, Sigma_true),
                    "log_likelihood": fit.log_likelihood,
                    "success": fit.optimizer_success,
                    "optimizer_message": fit.message,
                    **flat_fit_record("fitted", fit.parameters),
                    **flat_fit_record("true", params),
                }
                batch.append(record)
            append_csv(batch, table_path)
            records.extend(batch)
    df = pd.DataFrame(records)
    for family in ["rbf", "cs", "diagonal"]:
        sub = df[df["true_family"] == family]
        if not sub.empty:
            param = "ell" if family == "rbf" else "rho" if family == "cs" else "sigma"
            plot_parameter_vs_M(
                sub.rename(columns={"fitted_family": "method"}),
                param,
                root / f"plots/phase2/recovery_{family.upper() if family != 'diagonal' else 'diagonal'}",
            )
    return df


def recovery_scenarios() -> list[dict[str, Any]]:
    """Return Phase 2A parameter-recovery scenarios."""
    scenarios: list[dict[str, Any]] = []
    for sigma in [0.5, 1.0, 2.0]:
        scenarios.append({"family": "diagonal", "params": {"sigma": sigma}})
    for rho in [0.1, 0.3, 0.5, 0.7, 0.9]:
        scenarios.append({"family": "cs", "params": {"sigma": 1.0, "rho": rho}})
    for ell in [0.2, 0.5, 1.0, 2.0, 5.0]:
        scenarios.append({"family": "rbf", "params": {"sigma": 1.0, "ell": ell, "noise": 0.05}})
    return scenarios


def run_phase2_cross_kernel(config: dict[str, Any], root: Path | None = None, overwrite: bool = True) -> pd.DataFrame:
    """Run Phase 2B cross-kernel misspecification experiments."""
    root = project_root() if root is None else Path(root)
    ensure_output_dirs(root)
    n = int(config["n_bins"])
    x = positions(n, config["x_min"], config["x_max"])
    table_path = root / "results/phase2/tables/cross_kernel.csv"
    if overwrite and table_path.exists():
        table_path.unlink()

    records: list[dict[str, Any]] = []
    truth_items = list(config["truth_models"].items())
    fit_models = list(config["fit_models"])
    for t_index, (true_family, true_params) in enumerate(tqdm(truth_items, desc="Phase 2B truth")):
        Sigma_true = covariance_from_family(true_family, n, true_params, x=x)
        for m_index, M in enumerate(config["M_values"]):
            batch: list[dict[str, Any]] = []
            for repetition in range(int(config["n_repetitions"])):
                seed = repetition_seed(int(config["seed_base"]), t_index, m_index, repetition)
                rng = np.random.default_rng(seed)
                Y = generate_realizations(Sigma_true, int(M), rng)
                Sigma_sample = sample_covariance(Y)
                batch.append(_method_record(true_family, "sample", true_params, {}, Sigma_sample, Sigma_true, Y, M, repetition, seed, True, ""))
                for fit_family in fit_models:
                    fit = fit_covariance_model(Y, fit_family, x=x, seed=seed + 1)
                    batch.append(
                        _method_record(
                            true_family,
                            fit_family,
                            true_params,
                            fit.parameters,
                            fit.Sigma_hat,
                            Sigma_true,
                            Y,
                            M,
                            repetition,
                            seed,
                            fit.optimizer_success,
                            fit.message,
                            fit.log_likelihood,
                        )
                    )
            append_csv(batch, table_path)
            records.extend(batch)

    df = pd.DataFrame(records)
    _write_summary(df, root / "results/phase2/tables/cross_kernel_summary.csv")
    for M in config["M_values"]:
        plot_cross_kernel_heatmap(df, int(M), root / f"plots/phase2/cross_kernel_M{int(M):03d}")
    for true_family in ["diagonal", "cs", "rbf"]:
        sub = df[df["true_family"] == true_family].copy()
        if not sub.empty:
            plot_frobenius_vs_M(
                sub,
                root / f"plots/phase2/true_{true_family}_all_methods",
                n=n,
                title=f"Truth: {true_family}",
            )
    _plot_misspecification_floor(df, root / "plots/phase2/misspecification_error_floor")
    return df


def _method_record(
    true_family: str,
    method: str,
    true_params: dict[str, float],
    fitted_params: dict[str, float],
    Sigma_hat: Array,
    Sigma_true: Array,
    Y: Array,
    M: int,
    repetition: int,
    seed: int,
    success: bool,
    message: str,
    log_likelihood: float | None = None,
) -> dict[str, Any]:
    return {
        "true_family": true_family,
        "method": method,
        "fitted_family": method,
        "true_parameters": repr(true_params),
        "fitted_parameters": repr(fitted_params),
        "M": int(M),
        "repetition": repetition,
        "seed": seed,
        "frobenius_error": relative_frobenius_error(Sigma_hat, Sigma_true),
        "mean_absolute_error": mean_absolute_matrix_error(Sigma_hat, Sigma_true),
        "max_absolute_error": max_absolute_matrix_error(Sigma_hat, Sigma_true),
        "log_likelihood": log_likelihood,
        "success": success,
        "optimizer_message": message,
        **flat_fit_record("fitted", fitted_params),
        **flat_fit_record("true", true_params),
    }


def _plot_misspecification_floor(df: pd.DataFrame, output_stem: Path) -> None:
    fig_df = df[df["method"] != "sample"].copy()
    fig_df["comparison"] = fig_df["true_family"] + " -> " + fig_df["method"]
    stats = fig_df.groupby(["M", "comparison"])["frobenius_error"].median().reset_index()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    for comparison, sub in stats.groupby("comparison"):
        ax.plot(sub["M"], sub["frobenius_error"], marker="o", label=comparison)
    ax.set_xscale("log")
    ax.set_xlabel("M")
    ax.set_ylabel("median relative Frobenius error")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    from .plotting import save_figure

    save_figure(fig, output_stem)
