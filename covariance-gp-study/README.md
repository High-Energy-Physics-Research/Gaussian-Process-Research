# Covariance GP Study

Small, reproducible Python/Jupyter project for testing structured covariance estimators against the sample covariance matrix.

The object of inference is the covariance matrix, not Gaussian-process regression of a mean function. Synthetic experiments use

```text
y^(m) ~ N(0, Sigma_true)
```

with zero true mean so that mean inference does not confound covariance recovery.

## Scientific Question

When the true covariance matrix has structure, can a kernel-based covariance model estimate it with lower sampling error than the empirical covariance, especially when `M << n`?

The main metric is relative Frobenius error:

```text
||Sigma_hat - Sigma_true||_F / ||Sigma_true||_F
```

The code is intended to allow falsification. It does not assume that a structured model will win.

## Structure

```text
covariance-gp-study/
  notebooks/
  src/
  configs/
  results/
  plots/
```

Core implementation lives in `src/`; notebooks are thin experiment drivers.

## Dependencies

Required packages:

```text
numpy
scipy
pandas
matplotlib
pyyaml
tqdm
jupyter
```

## Running

From this directory:

```bash
jupyter notebook
```

Open the notebooks in order:

1. `notebooks/00_problem_definition.ipynb`
2. `notebooks/01_covariance_generators.ipynb`
3. `notebooks/02_phase1_statistical_efficiency.ipynb`
4. `notebooks/03_phase2_kernel_recovery.ipynb`
5. `notebooks/04_phase2_cross_kernel.ipynb`
6. `notebooks/05_summary.ipynb`

For a quick smoke test, temporarily lower `n_repetitions` in the YAML configs. The committed defaults follow the study specification (`R = 100`).

## Outputs

Phase 1 writes:

```text
results/phase1/tables/frobenius_vs_M.csv
results/phase1/tables/frobenius_vs_M_summary.csv
results/phase1/tables/statistical_gain_vs_M.csv
plots/phase1/
```

Phase 2 writes:

```text
results/phase2/tables/parameter_recovery.csv
results/phase2/tables/cross_kernel.csv
results/phase2/tables/cross_kernel_summary.csv
plots/phase2/
```

Seeds are recorded in raw result tables, and failures of optimizer convergence are kept rather than silently discarded.
