import itertools

FEATURES_ALL = ['Depth', 'qt', 'fs', 'u2']
TARGET_COL = 'Vs'
BASELINE_SUBSETS = {
    'Depth+qt+fs+u2': ['Depth','qt','fs','u2'],
    'Depth+qt+fs': ['Depth','qt','fs'],
    'Depth+qt+u2': ['Depth','qt','u2'],
    'Depth+qt': ['Depth','qt'],
    'qt': ['qt'],
}

def build_all_nonempty_subsets(features_in_order):
    """Generate all non-empty subsets and name them by joining with '+'"""
    subsets = {}
    for r in range(1, len(features_in_order) + 1):
        for comb in itertools.combinations(features_in_order, r):
            name = "+".join(comb)
            subsets[name] = list(comb)
    return subsets
