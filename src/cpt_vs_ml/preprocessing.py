"""Drop missing rows per feature regime; do not engineer or augment observations."""
from pathlib import Path
import numpy as np
import pandas as pd

def load_table(path):
    path = Path(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    if path.suffix.lower() == '.xlsx':
        return pd.read_excel(path)
    raise ValueError('Supply a CSV or XLSX file that you are authorised to use.')

def prepare_features(frame, features, target='Vs'):
    missing = set(features + [target]) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    subset = frame.dropna(subset=features+[target])
    X, y = subset[features].to_numpy(float), subset[target].to_numpy(float)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite values remain after missing-row removal.')
    return X, y, subset.index.to_numpy()
