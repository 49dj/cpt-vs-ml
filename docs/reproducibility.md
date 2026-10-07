# Extracted protocol and changes

The principal source is the combined baseline cell in `All files/Code.ipynb`.
The five default regimes are also present in the earlier baseline cell. The
combined cell supplies all five models, search spaces and a consistent input
table; an earlier separate MLP cell pointed at an augmented table and is excluded.
Selected correlation equations come from the existing modelling notebook.

Preserved: estimator definitions, search bounds, seeds, MLP stopping settings,
scaling order, negative-MSE selection, shuffle KFold, fold-wise refitting, pooled
metrics, zero-denominator MAPE convention, 40 non-NN / 30 NN search iterations.
Training R² pools repeated training predictions and is not an independent estimate.

Refactoring changes: relative CLI inputs, five-regime default with 15-regime option,
no warning suppression, explicit input errors, `error_score='raise'`, configurable
job count, JSON output and OOF indices in original row order. String hidden-layer
tuples use `ast.literal_eval` rather than executable `eval`. These changes improve
interfaces and bookkeeping without claiming a new validation design.

Smoke mode uses one full-input regime, 3 outer / 2 inner folds and two Bayesian
search evaluations per model. This configuration is visibly labelled in output.
MLP warnings about batch clipping or convergence are retained. Finite output does
not prove optimality or convergence. Paper-level results cannot be checked without
the excluded data and a confirmed final-paper analysis protocol.
