import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_squared_error
from cpt_vs_ml.preprocessing import prepare_features
from cpt_vs_ml.features import FEATURES_ALL, build_all_nonempty_subsets
from cpt_vs_ml.models import make_non_nn_models, MLPRegressorWrapper
from cpt_vs_ml.validation import evaluate_with_nested_cv

def test_oof_coverage_and_score():
    rng=np.random.default_rng(9); X=rng.normal(size=(60,2)); y=2*X[:,0]-X[:,1]
    entry=make_non_nn_models()['DecisionTree']
    result=evaluate_with_nested_cv(X,y,entry['estimator'],entry['search_space'],1,3,2,1)
    indices=[i for split in result['Split_indices'] for i in split['test']]
    assert sorted(indices)==list(range(len(y)))
    for split in result['Split_indices']: assert set(split['train']).isdisjoint(split['test'])
    assert np.isfinite(result['OOF_predictions']).all()
    assert result['RMSE']==pytest.approx(np.sqrt(mean_squared_error(y,result['OOF_predictions'])))

def test_missing_rows_are_feature_specific():
    frame=pd.DataFrame({'qt':[1.,2.,3.],'fs':[np.nan,0.1,0.2],'Vs':[10.,20.,30.]})
    assert len(prepare_features(frame,['qt'])[1])==3
    assert len(prepare_features(frame,['qt','fs'])[1])==2
    with pytest.raises(ValueError): prepare_features(frame,['Depth'])

def test_mlp_parser_rejects_code_and_subsets():
    assert len(build_all_nonempty_subsets(FEATURES_ALL))==15
    model=MLPRegressorWrapper().set_params(hidden_layer_sizes='(50,50)')
    assert model.hidden_layer_sizes==(50,50)
    with pytest.raises((ValueError,SyntaxError)):
        model.set_params(hidden_layer_sizes="__import__('os').getcwd()")
