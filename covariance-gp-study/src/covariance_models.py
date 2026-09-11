"""Parametric covariance models and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import cholesky


Array = np.ndarray


@dataclass(frozen=True)
class MatrixValidation:
    """Numerical diagnostics for a covariance matrix."""

    symmetric: bool
    positive_definite: bool
    positive_semidefinite: bool
    min_eigenvalue: float
    rank: int
    condition_number: float


def positions(n: int, x_min: float = 0.0, x_max: float = 10.0) -> Array:
    """Return the fixed one-dimensional bin positions used by RBF kernels."""
    return np.linspace(x_min, x_max, n)


def diagonal_covariance(n: int, sigma: float = 1.0) -> Array:
    """Return Sigma_ij = sigma^2 delta_ij."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return sigma**2 * np.eye(n)


def compound_symmetry_covariance(n: int, sigma: float = 1.0, rho: float = 0.0) -> Array:
    """Return a compound-symmetry covariance matrix."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    lower = -1.0 / (n - 1)
    if not lower < rho < 1.0:
        raise ValueError(f"rho must satisfy {lower} < rho < 1.")
    corr = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return sigma**2 * corr


def rbf_covariance(
    x: Array,
    sigma: float = 1.0,
    ell: float = 1.0,
    noise: float = 0.0,
    include_noise: bool = True,
) -> Array:
    """Return an RBF covariance matrix with optional diagonal noise."""
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if ell <= 0:
        raise ValueError("ell must be positive.")
    if noise < 0:
        raise ValueError("noise must be non-negative.")
    x = np.asarray(x, dtype=float)
    dx = x[:, None] - x[None, :]
    cov = sigma**2 * np.exp(-(dx**2) / (2.0 * ell**2))
    if include_noise:
        cov = cov + noise**2 * np.eye(x.size)
    return cov


def covariance_from_family(
    family: str,
    n: int,
    params: dict[str, float],
    x: Array | None = None,
    include_noise: bool = True,
) -> Array:
    """Build a covariance matrix from a named parametric family."""
    family = family.lower()
    if family in {"diagonal", "diag"}:
        return diagonal_covariance(n, sigma=float(params.get("sigma", 1.0)))
    if family == "cs":
        return compound_symmetry_covariance(
            n,
            sigma=float(params.get("sigma", 1.0)),
            rho=float(params.get("rho", 0.0)),
        )
    if family == "rbf":
        if x is None:
            x = positions(n)
        return rbf_covariance(
            x,
            sigma=float(params.get("sigma", 1.0)),
            ell=float(params.get("ell", 1.0)),
            noise=float(params.get("noise", params.get("sigma_n", 0.0))),
            include_noise=include_noise,
        )
    raise ValueError(f"Unknown covariance family: {family}")


def validate_covariance(matrix: Array, atol: float = 1e-10) -> MatrixValidation:
    """Check symmetry, eigenvalues, rank, and condition number."""
    matrix = np.asarray(matrix, dtype=float)
    symmetric = bool(np.allclose(matrix, matrix.T, atol=atol, rtol=0.0))
    eigvals = np.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    min_eig = float(eigvals.min())
    pd = bool(min_eig > atol)
    psd = bool(min_eig >= -atol)
    rank = int(np.linalg.matrix_rank(matrix, tol=atol))
    if rank == matrix.shape[0] and np.all(eigvals > 0):
        cond = float(eigvals.max() / eigvals.min())
    else:
        cond = float("inf")
    return MatrixValidation(
        symmetric=symmetric,
        positive_definite=pd,
        positive_semidefinite=psd,
        min_eigenvalue=min_eig,
        rank=rank,
        condition_number=cond,
    )


def add_jitter_for_cholesky(
    matrix: Array,
    initial_jitter: float = 1e-12,
    max_jitter: float = 1e-7,
) -> tuple[Array, float]:
    """Add the smallest numerical jitter needed for a Cholesky factorization."""
    matrix = np.asarray(matrix, dtype=float)
    jitter = 0.0
    while True:
        try:
            test = matrix if jitter == 0.0 else matrix + jitter * np.eye(matrix.shape[0])
            cholesky(test, lower=True, check_finite=False)
            return test, jitter
        except np.linalg.LinAlgError:
            jitter = initial_jitter if jitter == 0.0 else 10.0 * jitter
            if jitter > max_jitter:
                raise


def validation_record(family: str, params: dict[str, Any], matrix: Array) -> dict[str, Any]:
    """Return flat validation diagnostics suitable for CSV output."""
    info = validate_covariance(matrix)
    return {
        "family": family,
        "parameters": repr(params),
        "symmetric": info.symmetric,
        "positive_definite": info.positive_definite,
        "positive_semidefinite": info.positive_semidefinite,
        "min_eigenvalue": info.min_eigenvalue,
        "rank": info.rank,
        "condition_number": info.condition_number,
    }

