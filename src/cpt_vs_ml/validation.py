import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV

OUTER_FOLDS = 5
INNER_FOLDS = 3
RANDOM_STATE = 42

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-8, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0

def evaluate_with_nested_cv(X, y, base_estimator, search_space, n_iter,
                            outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS, n_jobs=-1):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y):
        raise ValueError("X must be 2D and y a matching 1D vector.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Inputs must be finite.")
    if outer_folds < 3 or inner_folds < 2 or n_iter < 1:
        raise ValueError("Use outer_folds >= 3, inner_folds >= 2, n_iter >= 1.")
    n_samples = len(X)

    # Check outer fold count
    outer_k = min(outer_folds, n_samples)
    if outer_k < 3:
        return {
            'error': f"Too few samples (n={n_samples}) for outer {outer_folds}-fold CV.",
            'n_samples': n_samples
        }

    outer_cv = KFold(n_splits=outer_k, shuffle=True, random_state=RANDOM_STATE)

    y_true_train, y_pred_train = [], []
    y_true_test,  y_pred_test  = [], []
    best_params_per_fold = []
    oof = np.full(n_samples, np.nan)
    folds = np.full(n_samples, -1, dtype=int)
    split_indices = []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X)):
        split_indices.append({"train": tr_idx.tolist(), "test": te_idx.tolist()})
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Inner fold count must be <= training set size; at least 2
        inner_k_max = max(2, min(inner_folds, len(X_tr)))
        if inner_k_max > len(X_tr):
            inner_k_max = max(2, len(X_tr))  # Fallback
        if inner_k_max > 10:
            inner_k_max = 10  # Safety upper limit to avoid too many folds for small data

        inner_cv = KFold(n_splits=inner_k_max, shuffle=True, random_state=RANDOM_STATE)

        search = BayesSearchCV(
            estimator=base_estimator,
            search_spaces=search_space,
            cv=inner_cv,
            n_iter=n_iter,
            scoring='neg_mean_squared_error',
            random_state=RANDOM_STATE,
            n_jobs=n_jobs, error_score="raise"
        )
        search.fit(X_tr, y_tr)

        best_params_per_fold.append(search.best_params_)

        y_hat_te = search.predict(X_te)
        oof[te_idx] = y_hat_te
        folds[te_idx] = fold
        y_hat_tr = search.predict(X_tr)

        y_true_test.append(y_te)
        y_pred_test.append(y_hat_te)
        y_true_train.append(y_tr)
        y_pred_train.append(y_hat_tr)

    # Aggregate results
    y_true_train_all = np.concatenate(y_true_train)
    y_pred_train_all = np.concatenate(y_pred_train)
    y_true_test_all  = np.concatenate(y_true_test)
    y_pred_test_all  = np.concatenate(y_pred_test)

    metrics = {
        'Train_R2': float(r2_score(y_true_train_all, y_pred_train_all)),
        'Test_R2':  float(r2_score(y_true_test_all,  y_pred_test_all)),
        'RMSE':     float(np.sqrt(mean_squared_error(y_true_test_all, y_pred_test_all))),
        'MAE':      float(mean_absolute_error(y_true_test_all, y_pred_test_all)),
        'MAPE':     float(mean_absolute_percentage_error(y_true_test_all, y_pred_test_all)),
        'BestParamsPerFold': best_params_per_fold,
        'OOF_predictions': oof.tolist(),
        'Outer_fold': folds.tolist(),
        'Split_indices': split_indices
    }
    return metrics
