"""Metrics for covariance matrix inference."""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


Array = np.ndarray


def relative_frobenius_error(A: Array, B: Array) -> float:
    """Return ||A - B||_F / ||B||_F, where B is usually Sigma_true."""
    return float(np.linalg.norm(A - B, ord="fro") / np.linalg.norm(B, ord="fro"))


def absolute_frobenius_error(A: Array, B: Array) -> float:
    """Return ||A - B||_F."""
    return float(np.linalg.norm(A - B, ord="fro"))


def mean_absolute_matrix_error(A: Array, B: Array) -> float:
    """Return the mean absolute element-wise matrix error."""
    return float(np.mean(np.abs(A - B)))


def max_absolute_matrix_error(A: Array, B: Array) -> float:
    """Return the max absolute element-wise matrix error."""
    return float(np.max(np.abs(A - B)))


def parameter_relative_error(theta_hat: float, theta_true: float) -> float:
    """Return a scalar relative parameter error."""
    if theta_true == 0:
        return float(abs(theta_hat - theta_true))
    return float(abs(theta_hat - theta_true) / abs(theta_true))


def heldout_negative_log_likelihood(Y_test: Array, Sigma_hat: Array) -> float:
    """Return mean held-out negative log likelihood without explicit inverses."""
    Y_test = np.asarray(Y_test, dtype=float)
    Sigma_hat = np.asarray(Sigma_hat, dtype=float)
    M, n = Y_test.shape
    c_factor, lower = cho_factor(Sigma_hat, lower=True, check_finite=False)
    alpha = cho_solve((c_factor, lower), Y_test.T, check_finite=False)
    quadratic = float(np.sum(Y_test.T * alpha))
    logdet = 2.0 * float(np.sum(np.log(np.diag(c_factor))))
    return 0.5 * (n * np.log(2.0 * np.pi) + logdet + quadratic / M)

