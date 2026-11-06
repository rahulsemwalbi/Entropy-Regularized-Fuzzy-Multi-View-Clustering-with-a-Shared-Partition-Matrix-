"""Run one method on one dataset, or one parameter-sensitivity experiment.

The experiment code is in experiments/<derma | xray | sensitivity>/: setup.py is executed first, then the file of the
method (or experiment), in one namespace. The runner sets the values of the run (RUN_SEED, RUN_LAMBDA, RUN_SIGMA, the
data paths IMAGE_DIR / XRAY_DIR, the image IDs, LAMBDA_GRID) and the model classes of spmvfcm/methods.py. The
experiment code writes its CSV of per-image measures into the run folder; run_info.json records the run.
"""
import contextlib
import hashlib
import importlib
import importlib.util
import json
import platform
import time
import types
from pathlib import Path

from . import methods
from .views import derma_views
from .xray_adapters import make_adapter

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
PROPOSED_METHODS = ["SPMVFCM", "SPMVKFCM", "SPWPMVFCM", "SPWPMVKFCM"]
COMPETITORS = ["kmeans", "fcm", "dcn", "dec", "dipencoder", "dipdeck"]
DEEP = {"dcn", "dec", "dipencoder", "dipdeck"}
CLASS_OF = {"SPMVKFCM": "MVKFC", "SPMVFCM": "MVJFC", "SPWPMVFCM": "WEIGHT_POINT_MVJFC", "SPWPMVKFCM": "WEIGHT_POINT_MVKFC"}
# names under which the X-ray code uses the weighted-point models
XRAY_CLASS_NAME = {"SPWPMVKFCM": "WEIGHT_POINT_MVJFC", "SPWPMVFCM": "WEIGHT_POINT_MVKFC"}
XRAY_SUBSETS = {"covid": ("Covid", 1, 208), "noncovid": ("Non_Covid", 208, 240)}   # folder, first ID, end ID
XRAY_OUTPUT_DIRS = ["1D_NON_COVID"]
SENS_EXPERIMENTS = {"SPMVKFCM_lambda_sigma": "MVKFC", "SPMVKFCM_d": "MVKFC",
                    "SPMVFCM_lambda": "MVJFC", "SPMVFCM_d": "MVJFC",
                    "SPWPMVKFCM_lambda_sigma": "WEIGHT_POINT_MVKFC", "SPWPMVKFCM_d": "WEIGHT_POINT_MVKFC",
                    "SPWPMVFCM_lambda": "WEIGHT_POINT_MVJFC", "SPWPMVFCM_d": "WEIGHT_POINT_MVJFC"}
SENS_OUTPUT_DIRS = ["MVKFCM/LAMDA_SIGMA", "MVKFCM/FUZZY", "MVJFCM/Lamda", "MVJFCM/FUZZY",
                    "WMVKFCM/FUZZY", "WMVKFCM/LAMDA_SIGMA", "WMVJFCM/FUZZY", "WMVJFCM"]


# ----------------------------------------------------------------------------- helpers
def code(group: str, name: str) -> str:
    return (EXPERIMENTS / group / f"{name}.py").read_text(encoding="utf-8")


def _run(ns, src, label, log):
    with contextlib.redirect_stdout(log):
        exec(compile(src, label, "exec"), ns)


def _prepare_runtime(log):
    """Figures are not displayed: plt.show() closes them (no result is computed from figures)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not getattr(plt.show, "_spmvfcm", False):
        def show(*_a, **_k):
            plt.close("all")
        show._spmvfcm = True
        plt.show = show


def _namespace(log=None):
    """A fresh namespace for the experiment code. If TensorFlow is not installed, `import tensorflow as tf` in the
    experiment code receives a stand-in whose tf.random.set_seed is a no-op: the X-ray deep-learning and FCM code only
    calls tf.random.set_seed; their models are PyTorch (ClustPy) / NumPy and never draw from TensorFlow's generator.
    The stand-in is visible to the experiment code only."""
    import builtins
    ns = {"__name__": "__notebook__"}
    if importlib.util.find_spec("tensorflow") is None:
        stub = types.ModuleType("tensorflow")
        stub.random = types.SimpleNamespace(set_seed=lambda *_a, **_k: None)
        real_import = builtins.__import__

        def cell_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tensorflow":
                if log is not None and not getattr(cell_import, "noted", False):
                    log.write("NOTE: tensorflow not installed; tf.random.set_seed is a no-op stand-in.\n")
                    cell_import.noted = True
                return stub
            return real_import(name, globals, locals, fromlist, level)
        b = dict(builtins.__dict__)
        b["__import__"] = cell_import
        ns["__builtins__"] = b
    return ns


def _versions():
    out = {"python": platform.python_version()}
    for m in ("numpy", "cv2", "pandas", "scipy", "skimage", "sklearn", "torch", "clustpy", "matplotlib"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception:                                 # pragma: no cover
            out[m] = "not installed"
    return out


def read_image_list(path):
    ids = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"{path}: no image IDs (one image ID per line expected)")
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"{path}: duplicated IDs {dup[:5]}")
    return ids


def _check_derma_files(ids, image_dir, mask_dir):
    missing = [i for i in ids if not (Path(image_dir) / f"{i}.jpg").is_file()
               or not (Path(mask_dir) / f"{i}_Segmentation.png").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} image/mask pairs not found, e.g. {missing[:5]}")


def _finish(target, info, t0):
    info["seconds"] = round(time.time() - t0, 1)
    info["csv_files"] = sorted(str(p.relative_to(target)) for p in Path(target).rglob("*.csv"))
    info["versions"] = _versions()
    (Path(target) / "run_info.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


MASK_SHAPE = (200, 200)   # ImageFeature size of the dermatology images


def _install_mask_recorder(ns, record_masks, method):
    """Save the mask passed to compute_metrics (the scored segmentation) and its inverse for chosen images.
    compute_metrics is called exactly once per image by every dermatology method; its result is unchanged."""
    import cv2
    import numpy as np
    want, folder = set(record_masks[0]), Path(record_masks[1]).resolve() / method
    folder.mkdir(parents=True, exist_ok=True)
    original = ns["compute_metrics"]
    state = {"calls": 0, "saved": []}

    def recording(preds, labels):
        name = ns["images"][state["calls"]]
        state["calls"] += 1
        if name in want:
            m = np.asarray(preds) > 0
            if m.shape != MASK_SHAPE:     # the deep-learning methods score flat masks (image.reshape(-1, 3), GT.ravel())
                if m.size != MASK_SHAPE[0] * MASK_SHAPE[1]:
                    raise ValueError(f"mask recorder: {name}: prediction of shape {m.shape}, expected {MASK_SHAPE}")
                m = m.reshape(MASK_SHAPE)  # row-major, the order in which the 200 x 200 image was flattened
            cv2.imwrite(str(folder / f"{name}_segmented.png"), (m * 255).astype(np.uint8))
            cv2.imwrite(str(folder / f"{name}_inverted.png"), ((~m) * 255).astype(np.uint8))
            state["saved"].append(name)
        return original(preds, labels)
    ns["compute_metrics"] = recording
    return state


def _model_parameters(ns, params):
    ns["RUN_LAMBDA"] = float(params["lambda"])
    if params.get("sigma") is not None:
        ns["RUN_SIGMA"] = float(params["sigma"])


def _info(dataset, method, seed, params, ids=None):
    info = {"dataset": dataset, "method": method, "seed": int(seed), "params": params}
    if ids is not None:
        info.update(n_images=len(ids), image_list_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest())
    return info


# ----------------------------------------------------------------------------- dermatology
def run_derma(method, image_list, image_dir, mask_dir, out_dir, seed, params=None, record_masks=None):
    """record_masks: (image ids, folder) -> <folder>/<method>/<id>_segmented.png and <id>_inverted.png."""
    ids = read_image_list(image_list)
    _check_derma_files(ids, image_dir, mask_dir)
    target = Path(out_dir) / "derma" / method
    target.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info = _info("derma", method, seed, params, ids)
    with open(target / "run.log", "w") as log, contextlib.chdir(target):
        _prepare_runtime(log)
        ns = _namespace(log)
        _run(ns, code("derma", "setup"), "experiments/derma/setup.py", log)
        ns["images"] = list(ids)
        ns["directory_path_test"] = str(Path(mask_dir).resolve())
        ns["RUN_SEED"] = int(seed)
        ns["IMAGE_DIR"] = str(Path(image_dir).resolve())
        if method in PROPOSED_METHODS:
            ns[CLASS_OF[method]] = getattr(methods, CLASS_OF[method])
            ns["derma_views"] = derma_views
            _model_parameters(ns, params)
        recorder = _install_mask_recorder(ns, record_masks, method) if record_masks else None
        _run(ns, code("derma", method), f"experiments/derma/{method}.py", log)
        if recorder is not None:
            if recorder["calls"] != len(ids):
                raise RuntimeError(f"mask recorder: compute_metrics called {recorder['calls']} times for {len(ids)} images")
            info["masks_saved"] = recorder["saved"]
    return _finish(target, info, t0)


# ----------------------------------------------------------------------------- COVID / non-COVID
def run_xray(method, dataset, rsua_dir, out_dir, seed, params=None):
    if dataset not in XRAY_SUBSETS:
        raise ValueError("dataset must be 'covid' or 'noncovid'")
    sub, first_id, end_id = XRAY_SUBSETS[dataset]
    if not (Path(rsua_dir) / sub / "images").is_dir():
        raise FileNotFoundError(f"{rsua_dir}: expected the extracted 'Data Chest X-Ray RSUA (Validated)' folder")
    target = Path(out_dir) / dataset / method
    target.mkdir(parents=True, exist_ok=True)
    for d in XRAY_OUTPUT_DIRS:
        (target / d).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    info = _info(dataset, method, seed, params)
    with open(target / "run.log", "w") as log, contextlib.chdir(target):
        _prepare_runtime(log)
        ns = _namespace(log)
        _run(ns, code("xray", "setup"), "experiments/xray/setup.py", log)
        ns["RUN_SEED"] = int(seed)
        ns["XRAY_DIR"] = f"{str(Path(rsua_dir).resolve()).rstrip('/')}/{sub}/"
        ns["FIRST_ID"], ns["END_ID"] = first_id, end_id
        ns["make_adapter"] = make_adapter
        if method in PROPOSED_METHODS:
            _model_parameters(ns, params)
        if method in XRAY_CLASS_NAME:
            ns[XRAY_CLASS_NAME[method]] = getattr(methods, CLASS_OF[method])
        _run(ns, code("xray", method), f"experiments/xray/{method}.py", log)
    return _finish(target, info, t0)


# ----------------------------------------------------------------------------- parameter sensitivity (dermatology)
def run_sensitivity(experiment, image_list, image_dir, mask_dir, out_dir, seed, grid=None, d_params=None,
                    target_name="sensitivity"):
    """grid: lambda or (lambda, sigma) values of a lambda experiment; d_params: (lambda, sigma) of a d experiment."""
    if experiment not in SENS_EXPERIMENTS:
        raise ValueError(f"experiment must be one of {sorted(SENS_EXPERIMENTS)}")
    ids = read_image_list(image_list)
    _check_derma_files(ids, image_dir, mask_dir)
    target = Path(out_dir) / target_name
    target.mkdir(parents=True, exist_ok=True)
    for d in SENS_OUTPUT_DIRS:
        (target / d).mkdir(parents=True, exist_ok=True)
    cls = SENS_EXPERIMENTS[experiment]
    t0 = time.time()
    info = _info("derma", experiment, seed, None, ids)
    with open(target / f"run_{experiment}.log", "w") as log, contextlib.chdir(target):
        _prepare_runtime(log)
        ns = _namespace(log)
        _run(ns, code("sensitivity", "setup"), "experiments/sensitivity/setup.py", log)
        ns[cls] = getattr(methods, cls)
        ns["images"], ns["ids"] = list(ids), list(range(len(ids)))
        ns["directory_path_test"] = str(Path(mask_dir).resolve())
        ns["RUN_SEED"] = int(seed)
        ns["IMAGE_DIR"] = str(Path(image_dir).resolve())
        ns["derma_views"] = derma_views
        if grid is not None:
            ns["LAMBDA_GRID"] = [tuple(float(a) for a in v) if isinstance(v, (tuple, list)) else float(v) for v in grid]
            info["grid"] = [list(g) if isinstance(g, (tuple, list)) else g for g in grid]
        if d_params is not None:
            _model_parameters(ns, {"lambda": d_params[0], "sigma": d_params[1]})
            info["d_params"] = {"lambda": d_params[0], "sigma": d_params[1]}
        _run(ns, code("sensitivity", experiment), f"experiments/sensitivity/{experiment}.py", log)
    info["seconds"] = round(time.time() - t0, 1)
    info["versions"] = _versions()
    (target / f"run_info_{experiment}.json").write_text(json.dumps(info, indent=1) + "\n")
    return info
