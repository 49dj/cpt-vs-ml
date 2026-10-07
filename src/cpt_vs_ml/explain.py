"""Optional model-agnostic SHAP on an already fitted end-to-end estimator."""
import numpy as np

def kernel_shap(estimator, background, samples, nsamples=100):
    import shap
    # predict includes feature scaling and, for MLP, inverse target scaling.
    explainer=shap.KernelExplainer(estimator.predict,np.asarray(background,float))
    return explainer.shap_values(np.asarray(samples,float),nsamples=nsamples)
