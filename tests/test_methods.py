"""The four methods (spmvfcm/methods.py) and the dermatology views (spmvfcm/views.py). Synthetic data only."""
import cv2
import numpy as np
import pytest

from spmvfcm import methods
from spmvfcm.views import derma_views

V, H, W, C = 3, 12, 14, 2
SETTINGS = {"MVJFC": dict(param=2e4), "MVKFC": dict(param=5.0, sigma=60.0),
            "WEIGHT_POINT_MVJFC": dict(param=2e4, sigma=60.0), "WEIGHT_POINT_MVKFC": dict(param=0.5, sigma=60.0)}


def state(name, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 255, (V, H, W))
    u = rng.random((C, H, W)); u /= u.sum(0)
    v = rng.uniform(40, 215, (V, C))
    w = np.full((V, H, W), 1 / V) if name.startswith("WEIGHT") else np.full(V, 1 / V)
    return x, w, v, u


def model(name, seed=0, iterations=30):
    x, w, v, u = state(name, seed)
    return methods.CLASSES[name](x, w, v, u, C, iterations=iterations, fuzzy=2.0, div_error_corr=None, **SETTINGS[name])


@pytest.mark.parametrize("name", list(methods.CLASSES))
def test_objective_does_not_increase(name):
    for seed in range(5):
        m = model(name, seed)
        m.optimize()
        t = np.array(m.objective_trace)
        assert len(t) == 30
        assert np.all(np.diff(t) <= 1e-9 * np.abs(t[:-1]).max())


@pytest.mark.parametrize("name", list(methods.CLASSES))
def test_memberships_and_weights_are_distributions(name):
    m = model(name)
    m.optimize()
    assert np.all(m.u >= 0) and np.allclose(m.u.sum(axis=0), 1)
    assert np.all(m.w >= 0) and np.allclose(m.w.sum(axis=0), 1)


@pytest.mark.parametrize("name", list(methods.CLASSES))
def test_update_order_is_P_then_h_then_M_with_the_newest_values(name):
    m = model(name, iterations=2)
    seen = []
    for meth in ("update_v", "update_w", "update_mu"):
        f = getattr(m, meth)
        setattr(m, meth, lambda f=f, meth=meth: (seen.append((meth, id(m.v), id(m.w), id(m.u))), f())[1])
    m.optimize()
    assert [s[0] for s in seen[:3]] == ["update_v", "update_w", "update_mu"]
    assert seen[1][1] != seen[0][1]                               # h uses the new P
    assert seen[2][1] == seen[1][1] and seen[2][2] != seen[1][2]   # M uses the new P and the new h


def test_stopping_test_on_the_objective_change():
    m = model("MVJFC", iterations=40)
    m.div_error_corr = 1e300                                      # any change is below epsilon: stop at the 2nd check
    m.optimize()
    assert len(m.objective_trace) == 2


def test_derma_views():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (200, 200, 3)).astype(np.uint8)
    v = derma_views(img, ("rgb", "lab", "morph"))
    assert v.shape == (9, 200, 200) and v.dtype == np.uint8
    rgb, lab, gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(img, cv2.COLOR_BGR2LAB), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(v[0], rgb[:, :, 0]) and np.array_equal(v[3], lab[:, :, 0])
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    assert np.array_equal(v[7], cv2.subtract(cv2.dilate(gray, k), cv2.erode(gray, k)))
