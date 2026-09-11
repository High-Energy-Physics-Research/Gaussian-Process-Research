"""Synthetic data generation for covariance inference studies."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def generate_realizations(
    Sigma_true: Array,
    M: int,
    rng: np.random.Generator,
    mean: Array | None = None,
) -> Array:
    """Generate M independent rows y^(m) ~ N(mean, Sigma_true)."""
    Sigma_true = np.asarray(Sigma_true, dtype=float)
    n = Sigma_true.shape[0]
    if mean is None:
        mean = np.zeros(n)
    return rng.multivariate_normal(np.asarray(mean, dtype=float), Sigma_true, size=M)


def repetition_seed(seed_base: int, *indices: int) -> int:
    """Derive reproducible deterministic seeds from experiment indices."""
    seed = int(seed_base)
    for value in indices:
        seed = (seed * 1_000_003 + int(value) * 9_176 + 12_345) % (2**32 - 1)
    return seed


def generate_pseudo_experiments(
    Sigma_true: Array,
    M: int,
    R: int,
    seed_base: int,
    mean: Array | None = None,
) -> list[tuple[int, Array]]:
    """Generate R full pseudo-experiments and return their seeds and samples."""
    experiments = []
    for repetition in range(R):
        seed = repetition_seed(seed_base, M, repetition)
        rng = np.random.default_rng(seed)
        experiments.append((seed, generate_realizations(Sigma_true, M, rng, mean=mean)))
    return experiments


def sample_covariance(Y: Array) -> Array:
    """Unstructured sample covariance baseline with rows as realizations."""
    return np.cov(Y, rowvar=False, ddof=1)

