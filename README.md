# CPT–Vs: baseline modelling code

How do empirical correlations and statistical learning predict shear-wave velocity
from cone penetration test measurements when different inputs are available?

The repository provides the analysis and modelling code associated with the study.
The original research datasets are not redistributed. This compact release is
limited to the supported **local baseline workflow**; it is not a complete
reproduction of the published article.

**Associated publication:** Hu, Y., Ibrahim, A., and Mousa, A. (2026).
*Correlations versus machine learning models for prediction of Vs from CPT in
fine-grained soil: a practice-oriented approach*. Canadian Geotechnical Journal,
63, 1–22. [DOI: 10.1139/cgj-2025-1057](https://doi.org/10.1139/cgj-2025-1057).
Yizhe Hu is the first author of this collaborative study; this portfolio release
does not assert sole ownership of the research or every implementation component.

## Methods and scope

- Decision tree, random forest, gradient boosting, SVR, and shallow MLP.
- Five input regimes: Depth/qt/fs/u2; Depth/qt/fs; Depth/qt/u2; Depth/qt; qt.
- Optional enumeration of all 15 nonempty feature combinations.
- Feature scaling inside fitted pipelines; MLP target scaling through
  `TransformedTargetRegressor`.
- Inner Bayesian search minimises mean squared error; outer predictions provide
  pooled OOF R², RMSE, MAE and MAPE. Parameters are recorded per outer fold.
- Optional model-agnostic SHAP and selected notebook correlation equations.

**Validation distinction:** this extracted code uses shuffled, row-wise `KFold`
(5 outer / 3 inner, seed 42), not grouped validation. The published abstract
describes grouped, nested cross-validation. Do not interpret this local baseline
as that grouped protocol, a new-site evaluation, or verification of paper values.

## Installation and use

Python 3.12 is the tested interpreter. From the repository root:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_baseline.py --synthetic --smoke --jobs 1
```

Synthetic example data are provided only to demonstrate the software workflow
and do not reproduce the reported study results. They are generated in memory
from independently invented equations and random draws.

For a longer synthetic run with the retained baseline defaults:

```bash
python scripts/run_baseline.py --synthetic --jobs 1
```

For an authorised private table with columns `Depth`, `qt`, `fs`, `u2`, `Vs`:

```bash
python scripts/run_baseline.py --input data/private/input.csv --jobs 1
```

Use the units of your scientific protocol consistently. The synthetic magnitudes
are illustrative and do not establish a physical-unit contract for empirical
equations. Outputs go to ignored `results/`; no data or fitted model is deposited.
The optional `--all-subsets` and `--models SVR RandomForest` flags restrict analyses.

For optional SHAP: `python -m pip install -r requirements-explain.txt`; call
`cpt_vs_ml.explain.kernel_shap` on a fitted end-to-end estimator and a small,
authorised background. This interface does not save or distribute training models.

## Layout and verification

`src/cpt_vs_ml/` holds preprocessing, feature definitions, original model/search
spaces, nested CV, correlations, explanation and plotting. `scripts/` contains the
CLI; `examples/` the independent generator; `tests/` checks OOF coverage, scoring,
missing rows and safe parameter parsing. `docs/` records data policy, protocol and
limitations.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Limitations, data and citation

No real CPT records, transformed subsets, borehole/site metadata, augmented clay
records, original plots, restricted-data metrics, correspondence, or saved research
models are included. Augmentation, duplicated-dataset learning curves and
new-domain claims are outside this release. Missing rows are dropped separately
per feature regime, so sample populations can differ. Row-wise validation may be
optimistic under within-site dependence; synthetic tests assess software only.

See [data access](docs/data_access.md), [reproducibility](docs/reproducibility.md),
[limitations](docs/limitations.md), and [rights](docs/rights.md). Cite the associated
article and specify the repository commit and local baseline protocol when using
this code. `CITATION.cff` supplies article metadata. No collaborator-wide permissive
licence is inferred from public visibility.

See [observed verification results](docs/verification.md) for executed checks and
their limits.
