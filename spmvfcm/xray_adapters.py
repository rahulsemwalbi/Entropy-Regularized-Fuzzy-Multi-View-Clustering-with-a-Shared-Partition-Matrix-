"""The shared-view models of the COVID / non-COVID code (MVKFCM1D, MVJFCM1D) optimised with spmvfcm.methods.

The X-ray run functions call  MVKFCM1D(x_data, C=2, fuzzy=..., sigma=..., param=..., iterations=..., seed=...)  /
MVJFCM1D(...), then .optimize() and .get_labels(). make_adapter keeps this interface: the X-ray class is constructed
for its initialisation (per-image seed, view weights 1/V, centres inside the data range, normalised memberships),
and the iterations are those of SPMVKFCM / SPMVFCM (spmvfcm/methods.py) with epsilon = 1e-10.
Data (V, N) is passed as (V, N, 1).
"""
import numpy as np

from . import methods

XRAY_THRESHOLD = 1e-10    # stop when |Theta(t) - Theta(t-1)| < epsilon


def make_adapter(original_cls, kernel: bool):
    """The X-ray class original_cls, optimised with SPMVKFCM (kernel=True) or SPMVFCM."""
    model_cls = methods.MVKFC if kernel else methods.MVJFC

    class Adapter:
        __name__ = original_cls.__name__
        original = original_cls
        model_class = model_cls

        def __init__(self, x, C, fuzzy=2.0, sigma=None, param=1e11, iterations=40, seed=None):
            if kernel:
                init = original_cls(x, C, fuzzy=fuzzy, sigma=sigma, param=param, iterations=iterations, seed=seed)
            else:
                init = original_cls(x, C, fuzzy=fuzzy, param=param, iterations=iterations, seed=seed)
            self.init_state = (init.w.copy(), init.v.copy(), init.u.copy())
            kw = dict(param=init.param, div_error_corr=XRAY_THRESHOLD, iterations=init.iterations, fuzzy=init.fuzzy)
            if kernel:
                kw["sigma"] = init.sigma
            self.model = model_cls(init.x[:, :, None], init.w.copy(), init.v.T.copy(), init.u[:, :, None].copy(),
                                       C, **kw)
            self.loss_margin = self.model.loss_margin

        def optimize(self):
            self.model.optimize()
            self.u = self.model.u[:, :, 0]
            self.w = self.model.w
            self.v = self.model.v.T

        def get_labels(self):
            return np.argmax(self.u, axis=0)

    Adapter.__qualname__ = original_cls.__name__
    return Adapter
