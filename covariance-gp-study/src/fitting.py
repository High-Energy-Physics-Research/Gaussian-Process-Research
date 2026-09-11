"""Likelihood-based fitting of structured covariance models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from .covariance_models import covariance_from_family


Array = np.ndarray


@dataclass
class FitResult:
    """Result of a covariance-model maximum-likelihood fit."""

    family: str
    parameters: dict[str, float]
    Sigma_hat: Array
    log_likelihood: float
    optimizer_success: bool
    number_iterations: int
    message: str
    objective_value: float


def _softplus(z: float) -> float:
    return float(np.logaddexp(0.0, z))


def _inv_softplus(y: float) -> float:
    return float(np.log(np.expm1(max(y, 1e-12))))


def _rho_from_unconstrained(z: float, n: int, eps: float = 1e-6) -> float:
    lower = -1.0 / (n - 1) + eps
    upper = 1.0 - eps
    return float(lower + (upper - lower) / (1.0 + np.exp(-z)))


def _params_from_vector(family: str, z: Array, n: int, fit_noise: bool = True) -> dict[str, float]:
    family = family.lower()
    if family in {"diagonal", "diag"}:
        return {"sigma": _softplus(float(z[0])) + 1e-10}
    if family == "cs":
        return {
            "sigma": _softplus(float(z[0])) + 1e-10,
            "rho": _rho_from_unconstrained(float(z[1]), n),
        }
    if family == "rbf":
        params = {
            "sigma": _softplus(float(z[0])) + 1e-10,
            "ell": _softplus(float(z[1])) + 1e-10,
        }
        params["noise"] = _softplus(float(z[2])) + 1e-10 if fit_noise else 0.0
        return params
    raise ValueError(f"Unknown covariance family: {family}")


def _initial_vectors(family: str, n: int, rng: np.random.Generator, restarts: int, fit_noise: bool) -> list[Array]:
    base: list[Array]
    if family in {"diagonal", "diag"}:
        base = [np.array([_inv_softplus(1.0)])]
    elif family == "cs":
        base = [np.array([_inv_softplus(1.0), 0.0])]
    elif family == "rbf":
        z_noise = _inv_softplus(0.05)
        base = [np.array([_inv_softplus(1.0), _inv_softplus(1.0), z_noise])]
        if not fit_noise:
            base = [np.array([_inv_softplus(1.0), _inv_softplus(1.0)])]
    else:
        raise ValueError(f"Unknown covariance family: {family}")
    while len(base) < restarts:
        trial = base[0] + rng.normal(0.0, 1.0, size=base[0].shape)
        if family == "cs":
            trial[1] = rng.normal(0.0, 1.5)
        base.append(trial)
    return base[:restarts]


def negative_log_likelihood(
    Y: Array,
    Sigma: Array,
    include_constant: bool = False,
) -> float:
    """Return -log L for zero-mean multivariate-normal realizations."""
    Y = np.asarray(Y, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    M, n = Y.shape
    try:
        c_factor, lower = cho_factor(Sigma, lower=True, check_finite=False)
        alpha = cho_solve((c_factor, lower), Y.T, check_finite=False)
        quadratic = float(np.sum(Y.T * alpha))
        logdet = 2.0 * float(np.sum(np.log(np.diag(c_factor))))
    except Exception:
        return float("inf")
    nll = 0.5 * (M * logdet + quadratic)
    if include_constant:
        nll += 0.5 * M * n * np.log(2.0 * np.pi)
    return float(nll)


def fit_covariance_model(
    Y: Array,
    family: str,
    x: Array | None = None,
    n_restarts: int = 8,
    seed: int = 0,
    fit_noise: bool = True,
    maxiter: int = 500,
) -> FitResult:
    """Fit a parametric covariance model by maximizing the joint likelihood."""
    Y = np.asarray(Y, dtype=float)
    n = Y.shape[1]
    family = family.lower()
    rng = np.random.default_rng(seed)

    def objective(z: Array) -> float:
        params = _params_from_vector(family, z, n, fit_noise=fit_noise)
        Sigma = covariance_from_family(family, n, params, x=x, include_noise=fit_noise)
        return negative_log_likelihood(Y, Sigma)

    best = None
    for z0 in _initial_vectors(family, n, rng, max(1, n_restarts), fit_noise):
        result = minimize(objective, z0, method="Nelder-Mead", options={"maxiter": maxiter})
        if best is None or result.fun < best.fun:
            best = result

    assert best is not None
    params = _params_from_vector(family, best.x, n, fit_noise=fit_noise)
    Sigma_hat = covariance_from_family(family, n, params, x=x, include_noise=fit_noise)
    log_likelihood = -negative_log_likelihood(Y, Sigma_hat, include_constant=True)
    return FitResult(
        family=family,
        parameters=params,
        Sigma_hat=Sigma_hat,
        log_likelihood=float(log_likelihood),
        optimizer_success=bool(best.success),
        number_iterations=int(getattr(best, "nit", -1)),
        message=str(best.message),
        objective_value=float(best.fun),
    )

