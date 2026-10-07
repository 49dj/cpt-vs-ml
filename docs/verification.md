# Verification, 7 October 2026

Tested with Python 3.12.10 on Windows; direct dependency versions are pinned in
`requirements.txt`. Editable package installation and `pip check` passed.

- `python -m pytest -q`: 3 tests passed.
- `python scripts/run_baseline.py --synthetic --smoke --jobs 1`: all five models
  completed with finite OOF metrics (one regime, 3x2 folds, two searches/model).
- Optional SHAP: a fitted pipeline produced finite 2x2 Kernel SHAP values on
  independent synthetic inputs; the plotting helper wrote a synthetic PNG.

MLP emitted batch-size clipping warnings on small training folds; these were
not suppressed. Synthetic RMSE values are workflow diagnostics, not study results.
No real-data or grouped-CV paper reproduction was attempted.

`CITATION.cff` passed validation against the official CFF 1.2 schema.
