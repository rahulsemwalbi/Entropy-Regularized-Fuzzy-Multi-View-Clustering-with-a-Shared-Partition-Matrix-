"""The four methods: the update rules of spmvfcm/models.py, iterated one block after another, P -> h -> M.

Per iteration: the objective Theta is evaluated and the iterations stop when |Theta(t) - Theta(t-1)| < epsilon; then

    P <- update_v()    (from the current M and h)
    h <- update_w()    (from the new P and the current M)
    M <- update_mu()   (from the new P and the new h)

Every block update minimises the objective given the other blocks (the kernel centre update is the fixed-point step,
which does not increase the objective), so the objective does not increase from one iteration to the next.
"""
import numpy as np

from . import models


def _kernel_dist(self):
    return 1 - np.exp(-np.stack(tuple((self.x - self.v[:, i][:, np.newaxis, np.newaxis]) ** 2
                                      for i in range(self.C))) / (self.sigma ** 2))


def _sq_dist(self):
    return np.stack(tuple((self.x - self.v[:, i][:, np.newaxis, np.newaxis]) ** 2 for i in range(self.C)))


def objective(self):
    """The objective of the model: Eq. (1) SPMVFCM, (5b) SPMVKFCM, (9) SPWPMVFCM, (13b) SPWPMVKFCM."""
    name = type(self).__mro__[1].__name__
    if name == "MVJFC":
        weighted = np.sum(_sq_dist(self) * self.w[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
    elif name == "MVKFC":
        weighted = np.sum(2 * _kernel_dist(self) * self.w[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
    elif name == "WEIGHT_POINT_MVJFC":
        weighted = np.sum(_sq_dist(self) * self.w, axis=1)
    else:                                                   # WEIGHT_POINT_MVKFC
        weighted = np.sum(2 * _kernel_dist(self) * self.w, axis=1)
    entropy = self.param * np.sum(self.w * np.log(self.w))
    return np.sum(weighted * (self.u ** self.fuzzy)) + entropy


def _sequential_optimize(self):
    prev_result = None
    self.objective_trace = []
    for _ in range(self.iterations):
        final_result = objective(self)
        self.objective_trace.append(final_result)
        self.loss_margin.append(abs(final_result))
        if prev_result is not None and self.div_error_corr is not None:
            if abs(final_result - prev_result) < self.div_error_corr:
                break
        prev_result = final_result
        self.v = self.update_v()        # P: from the current M (and h)
        self.w = self.update_w()        # h: from the new P and the current M
        self.u = self.update_mu()       # M: from the new P and the new h


def _make(base):
    cls = type(base.__name__, (base,), {"optimize": _sequential_optimize, "update_order": "P -> h -> M (sequential)"})
    cls.__module__ = __name__
    cls.__doc__ = f"{base.__name__} with one-after-another updates (P -> h -> M); everything else from {base.__module__}."
    return cls


MVJFC = _make(models.MVJFC)                             # SPMVFCM
MVKFC = _make(models.MVKFC)                             # SPMVKFCM
WEIGHT_POINT_MVJFC = _make(models.WEIGHT_POINT_MVJFC)   # SPWPMVFCM
WEIGHT_POINT_MVKFC = _make(models.WEIGHT_POINT_MVKFC)   # SPWPMVKFCM
CLASSES = {"MVJFC": MVJFC, "MVKFC": MVKFC, "WEIGHT_POINT_MVJFC": WEIGHT_POINT_MVJFC,
              "WEIGHT_POINT_MVKFC": WEIGHT_POINT_MVKFC}


# method names
SPMVKFCM = MVKFC
SPMVFCM = MVJFC
SPWPMVFCM = WEIGHT_POINT_MVJFC
SPWPMVKFCM = WEIGHT_POINT_MVKFC
