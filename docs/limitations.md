# Scientific limitations

1. Local baseline `KFold` is row-wise. No group labels or grouped split are supplied;
   this code cannot support claims of generalisation to independent sites.
2. The published abstract describes a grouped protocol that differs from this
   local implementation. This discrepancy is disclosed rather than repaired by
   inventing unavailable group metadata.
3. Real-data sample counts, tuning outcomes and reported values are unverified here.
4. Per-regime missing-row removal can change the evaluated population.
5. MAPE is unstable near zero; zero targets are replaced by 1e-8 as in the source.
6. Empirical equations are transcribed literally; source-unit contracts and
   calibration domains have not been independently confirmed. They are excluded
   from the arbitrary-unit synthetic comparison.
7. Kernel SHAP uses the complete estimator prediction function. Explanations are
   background-dependent, computationally expensive and are not causal findings.
8. No data augmentation, external-domain experiment, or grouped-CV implementation
   is included. No XGBoost model was introduced.
