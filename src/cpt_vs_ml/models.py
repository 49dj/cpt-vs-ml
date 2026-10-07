import ast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from skopt.space import Integer, Real, Categorical

RANDOM_STATE = 42
N_ITER_NONNN = 40
N_ITER_NN = 30

class MLPRegressorWrapper(MLPRegressor):
    """Convert string hidden_layer_sizes to tuple to avoid skopt dimension check issues"""
    def set_params(self, **params):
        hls_key = 'hidden_layer_sizes'
        if hls_key in params and isinstance(params[hls_key], str):
            params[hls_key] = ast.literal_eval(params[hls_key])
        return super().set_params(**params)

def make_non_nn_models():
    """Return estimators & search spaces for non-NN models"""
    models_param_grid = {}

    # Decision Tree
    dt_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', DecisionTreeRegressor(random_state=RANDOM_STATE))
    ])
    models_param_grid['DecisionTree'] = {
        'estimator': dt_pipe,
        'search_space': {
            'model__max_depth': Integer(3, 20),
            'model__min_samples_split': Integer(2, 20),
            'model__min_samples_leaf': Integer(1, 10),
        },
        'n_iter': N_ITER_NONNN
    }

    # Random Forest
    rf_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(random_state=RANDOM_STATE))
    ])
    models_param_grid['RandomForest'] = {
        'estimator': rf_pipe,
        'search_space': {
            'model__n_estimators': Integer(50, 200),
            'model__max_depth': Integer(5, 20),
            'model__max_features': Categorical(['sqrt', 'log2', None]),
        },
        'n_iter': N_ITER_NONNN
    }

    # Gradient Boosting
    gbr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GradientBoostingRegressor(random_state=RANDOM_STATE))
    ])
    models_param_grid['GradientBoosting'] = {
        'estimator': gbr_pipe,
        'search_space': {
            'model__n_estimators': Integer(50, 200),
            'model__learning_rate': Real(1e-2, 5e-1, prior='log-uniform'),  # Equivalent to 0.01~0.5
            'model__max_depth': Integer(3, 10),
            'model__min_samples_split': Integer(2, 20),
        },
        'n_iter': N_ITER_NONNN
    }

    # SVR
    svr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVR())
    ])
    models_param_grid['SVR'] = {
        'estimator': svr_pipe,
        'search_space': {
            'model__C': Real(1e-1, 1e2, prior='log-uniform'),
            'model__gamma': Categorical(['scale', 'auto']),
        },
        'n_iter': N_ITER_NONNN
    }

    return models_param_grid

def make_mlp_model():
    """
    Use TTR to standardize y; X is standardized in the Pipeline.
    Key: hidden_layer_sizes is represented as a string + wrapper to avoid skopt 'Not all points are within the bounds' error.
    """
    nn_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPRegressorWrapper(
            random_state=RANDOM_STATE,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=10,
            solver='adam',
            validation_fraction=0.1  # Explicitly set, must be (0,1)
        ))
    ])
    estimator = TransformedTargetRegressor(
        regressor=nn_pipe,
        transformer=StandardScaler()  # Standardize y to improve MLP convergence and stability
    )

    # Use string to represent layer structure, wrapper converts to tuple
    nn_param_space = {
        'regressor__model__hidden_layer_sizes': Categorical([
            "(50,)", "(100,)", "(50,50)", "(100,50)"
            # Add "(100,100)" for deeper structures if needed
        ]),
        'regressor__model__activation': Categorical(['relu', 'tanh']),
        'regressor__model__learning_rate_init': Real(1e-4, 1e-2, prior='log-uniform'),
        'regressor__model__alpha': Real(1e-5, 1e-3, prior='log-uniform'),
        'regressor__model__batch_size': Integer(16, 128),
    }

    return {
        'MLP': {
            'estimator': estimator,
            'search_space': nn_param_space,
            'n_iter': N_ITER_NN
        }
    }
