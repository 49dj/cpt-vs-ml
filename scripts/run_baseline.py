"""Run baseline code on authorised private input or an independent synthetic demo."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples'))
from cpt_vs_ml.features import BASELINE_SUBSETS, FEATURES_ALL, build_all_nonempty_subsets
from cpt_vs_ml.preprocessing import load_table, prepare_features
from cpt_vs_ml.models import make_non_nn_models, make_mlp_model
from cpt_vs_ml.validation import evaluate_with_nested_cv
from synthetic import make_synthetic

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--synthetic',action='store_true')
    group.add_argument('--input',type=Path)
    ap.add_argument('--smoke',action='store_true',help='3x2 folds, 2 searches; software check only')
    ap.add_argument('--all-subsets',action='store_true')
    ap.add_argument('--models',nargs='+',choices=['DecisionTree','RandomForest','GradientBoosting','SVR','MLP'])
    ap.add_argument('--jobs',type=int,default=-1)
    ap.add_argument('--output',type=Path,default=Path('results/baseline.json'))
    args=ap.parse_args()
    frame=make_synthetic() if args.synthetic else load_table(args.input)
    model_defs={**make_non_nn_models(),**make_mlp_model()}
    subsets=build_all_nonempty_subsets(FEATURES_ALL) if args.all_subsets else BASELINE_SUBSETS
    if args.smoke:
        subsets={'Depth+qt+fs+u2':BASELINE_SUBSETS['Depth+qt+fs+u2']}
    output={'synthetic':args.synthetic,'smoke':args.smoke,'validation':'row-wise shuffled KFold','results':{}}
    for subset,features in subsets.items():
        X,y,rows=prepare_features(frame,features)
        output['results'][subset]={}
        for name,entry in model_defs.items():
            if args.models and name not in args.models: continue
            metrics=evaluate_with_nested_cv(X,y,entry['estimator'],entry['search_space'],
                2 if args.smoke else entry['n_iter'],3 if args.smoke else 5,2 if args.smoke else 3,args.jobs)
            if 'error' in metrics: raise ValueError(metrics['error'])
            metrics['row_indices']=rows.tolist()
            output['results'][subset][name]=metrics
            print(f'{subset} / {name}: OOF RMSE={metrics["RMSE"]:.4f}',flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(output,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else v),encoding='utf-8')

if __name__=='__main__': main()
