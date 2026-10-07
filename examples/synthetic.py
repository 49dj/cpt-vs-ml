"""Independently invented positive-valued inputs, not transformed research data."""
import numpy as np
import pandas as pd

def make_synthetic(n=100, seed=2026):
    rng = np.random.default_rng(seed)
    depth = rng.uniform(1,30,n)
    qt = rng.uniform(1,15,n)
    fs = rng.uniform(0.05,0.5,n)
    u2 = rng.uniform(0.1,2,n)
    vs = 100 + 12*np.sqrt(qt) + 1.5*depth + 15*fs + 4*u2 + rng.normal(0,3,n)
    return pd.DataFrame({'Depth':depth,'qt':qt,'fs':fs,'u2':u2,'Vs':vs})
