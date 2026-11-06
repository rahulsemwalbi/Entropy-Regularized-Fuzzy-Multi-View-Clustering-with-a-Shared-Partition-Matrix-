#!/usr/bin/env python
"""Experiments (configs/final.json), on the CPU.

  python scripts/run_experiments.py main          # all 10 methods x 3 datasets x 3 seeds (90 runs) + checks
  python scripts/run_experiments.py sensitivity   # parameter sensitivity (ISIC, seed 42): lambda / sigma -> best -> d
  python scripts/run_experiments.py summary       # results/final/SUMMARY.md
  python scripts/run_experiments.py check         # checks of the main runs
  python scripts/run_experiments.py one derma SPMVFCM --seed 42     # a single method
  (scripts/run_all.sh: main -> sensitivity -> summary)

Outputs (results/final/):
  seed_<s>/<dataset>/<dataset>_seed<s>_<method>.csv   one row per image (id, 9 measures, confusion matrix
                                                      [[TP, FP], [FN, TN]]) and a final 'mean' row
  seed_<s>/notebook_output/<dataset>/<method>/        the experiment's own CSV, run_info.json, run.log
  seed_42/segmented image/<method>/<id>_segmented.png and <id>_inverted.png  (scored mask, 200 x 200)
  seed_42/original image/, seed_42/ground truth/      the example images and masks, copied unchanged
  sensitivity_seed_42/                                one CSV per parameter value, best_parameters.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "configs" / "final.json").read_text())
OUT = ROOT / CFG["output"]
DATA = ROOT / "data"
ISIC_IMG = DATA / "ISBI2016_ISIC_Part1_Training_Data"
ISIC_GT = DATA / "ISBI2016_ISIC_Part1_Training_GroundTruth"
RSUA = DATA / "Data Chest X-Ray RSUA (Validated)"
PROPOSED = ["SPMVFCM", "SPMVKFCM", "SPWPMVFCM", "SPWPMVKFCM"]
DEEP = ["dcn", "dec", "dipencoder", "dipdeck"]
XRAY_IDS = {"covid": range(1, 208), "noncovid": range(208, 240)}
N_IMAGES = {"derma": 900, "covid": 207, "noncovid": 32}
METRICS = ["IoU", "Dice", "MSE", "RMSE", "PSNR", "accuracy", "precision", "recall", "F1"]


# ----------------------------------------------------------------------------- one job (runs in its own process)
def final_csv(seed, dataset, method):
    return OUT / f"seed_{seed}" / dataset / f"{dataset}_seed{seed}_{method}.csv"


def write_named_csv(src_csv, dest_csv, dataset):
    """Format 1 with the image id: the dermatology CSVs already have it; the X-ray CSVs get it (CXR_Images_<n>)."""
    import pandas as pd
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    if dataset == "derma":
        shutil.copyfile(src_csv, dest_csv)
        return
    df = pd.read_csv(src_csv, index_col=0)
    names = [f"CXR_Images_{i}" for i in XRAY_IDS[dataset]]
    if len(df) != len(names) + 1:
        raise ValueError(f"{src_csv}: {len(df)} rows, expected {len(names) + 1}")
    df.insert(0, "id", names + ["mean"])
    df.to_csv(dest_csv)


def run_job(spec):
    sys.path.insert(0, str(ROOT))
    from spmvfcm import runner as R
    if spec["kind"] == "main":
        seed, ds, m = spec["seed"], spec["dataset"], spec["method"]
        nb_out = OUT / f"seed_{seed}" / "notebook_output"
        params = CFG["params"].get(m)
        if ds == "derma":
            rec = None
            if seed == CFG["examples"]["seed"]:
                rec = (CFG["examples"]["images"], OUT / f"seed_{seed}" / "segmented image")
            R.run_derma(m, ROOT / CFG["derma_image_list"], ISIC_IMG, ISIC_GT, nb_out, seed=seed, params=params,
                        record_masks=rec)
        else:
            R.run_xray(m, ds, RSUA, nb_out, seed=seed, params=params)
        csvs = sorted((nb_out / ds / m).rglob("*.csv"))
        if len(csvs) != 1:
            raise RuntimeError(f"expected one CSV in {nb_out / ds / m}, found {len(csvs)}")
        write_named_csv(csvs[0], final_csv(seed, ds, m), ds)
    else:
        sens = CFG["sensitivity"]
        R.run_sensitivity(spec["experiment"], ROOT / sens["image_list"], ISIC_IMG, ISIC_GT, OUT,
                          seed=sens["seed"], grid=spec.get("grid"), d_params=spec.get("d_params"),
                          target_name=f"sensitivity_seed_{sens['seed']}")


# ----------------------------------------------------------------------------- scheduler
def schedule(jobs, label):
    """jobs: list of (name, spec, threads); started in order whenever the CPU thread budget allows."""
    budget = CFG["cpu"]["thread_budget"]
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    progress = open(OUT / f"{label}_progress.log", "a")

    def say(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        progress.write(line + "\n"); progress.flush(); print(line, flush=True)

    say(f"{label}: {len(jobs)} jobs, thread budget {budget}")
    pending, running, failed = list(jobs), {}, []
    while pending or running:
        used = sum(t for _, t, _ in running.values())
        while pending and used + pending[0][2] <= budget:
            name, spec, threads = pending.pop(0)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CLUSTPY_DEVICE="cpu", MPLBACKEND="Agg",
                       OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
            p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "job", json.dumps(spec)], env=env,
                                 stdout=open(logs / f"{name}.log", "w"), stderr=subprocess.STDOUT)
            running[p] = (name, threads, time.time())
            used += threads
            say(f"start  {name} ({threads} threads)")
        time.sleep(15)
        for p in list(running):
            if p.poll() is not None:
                name, _, t0 = running.pop(p)
                ok = p.returncode == 0
                say(f"{'done  ' if ok else 'FAILED'} {name} ({(time.time() - t0) / 60:.1f} min)")
                if not ok:
                    failed.append(name)
    say(f"{label}: finished, {len(failed)} failed {failed}")
    progress.close()
    return failed


# ----------------------------------------------------------------------------- main experiments
def main_jobs():
    jobs = []
    t = CFG["cpu"]
    order = ([("derma", m) for m in DEEP] + [("derma", m) for m in PROPOSED + ["kmeans", "fcm"]]
             + [(ds, m) for ds in ("covid", "noncovid") for m in DEEP]
             + [(ds, m) for ds in ("covid", "noncovid") for m in PROPOSED + ["kmeans", "fcm"]])
    for ds, m in order:
        for s in CFG["seeds"]:
            threads = t["deep_threads"] if m in DEEP else t["other_threads"]
            jobs.append((f"seed{s}_{ds}_{m}", {"kind": "main", "seed": s, "dataset": ds, "method": m}, threads))
    return jobs


def copy_examples():
    ex = CFG["examples"]
    base = OUT / f"seed_{ex['seed']}"
    (base / "original image").mkdir(parents=True, exist_ok=True)
    (base / "ground truth").mkdir(parents=True, exist_ok=True)
    for i in ex["images"]:
        shutil.copyfile(ISIC_IMG / f"{i}.jpg", base / "original image" / f"{i}.jpg")
        shutil.copyfile(ISIC_GT / f"{i}_Segmentation.png", base / "ground truth" / f"{i}_Segmentation.png")


def check_main():
    import cv2
    import numpy as np
    import pandas as pd
    problems = []
    for s in CFG["seeds"]:
        for ds in CFG["datasets"]:
            for m in CFG["methods"]:
                f = final_csv(s, ds, m)
                if not f.exists():
                    problems.append(f"missing {f.relative_to(OUT)}"); continue
                df = pd.read_csv(f, index_col=0)
                if len(df) != N_IMAGES[ds] + 1 or df["id"].iloc[-1] != "mean" or df["id"].iloc[:-1].duplicated().any():
                    problems.append(f"{f.name}: {len(df)} rows / ids not as expected")
                if df[METRICS].iloc[:-1].astype(float).isna().any().any():
                    problems.append(f"{f.name}: NaN metrics")
                info = json.loads((OUT / f"seed_{s}" / "notebook_output" / ds / m / "run_info.json").read_text())
                if info.get("seed") != s or info.get("method") != m or info.get("dataset") != ds:
                    problems.append(f"{f.name}: run_info does not match ({info.get('seed')}, {info.get('method')})")
                if info.get("params") != CFG["params"].get(m):
                    problems.append(f"{f.name}: params {info.get('params')}")
    ex = CFG["examples"]
    base = OUT / f"seed_{ex['seed']}"
    for m in CFG["methods"]:
        for i in ex["images"]:
            a = cv2.imread(str(base / "segmented image" / m / f"{i}_segmented.png"), cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(str(base / "segmented image" / m / f"{i}_inverted.png"), cv2.IMREAD_GRAYSCALE)
            if a is None or b is None:
                problems.append(f"mask missing: {m}/{i}"); continue
            if a.shape != (200, 200) or not np.array_equal(a.astype(int) + b.astype(int), np.full((200, 200), 255)):
                problems.append(f"mask pair not inverse / wrong size: {m}/{i}")
    for i in ex["images"]:
        for f in (base / "original image" / f"{i}.jpg", base / "ground truth" / f"{i}_Segmentation.png"):
            if not f.exists():
                problems.append(f"missing {f.relative_to(OUT)}")
    (OUT / "check_main.txt").write_text("\n".join(problems) + ("\n" if problems else "all checks passed\n"))
    print("\n".join(problems) if problems else "all checks passed")
    return problems


# ----------------------------------------------------------------------------- parameter sensitivity
SENS_FILES = {  # lambda experiment -> CSV of one parameter value (folder and name written by the experiment code)
    "SPMVKFCM_lambda_sigma": lambda l, s: f"MVKFCM/LAMDA_SIGMA/FINAL_RESULT_900_1_MVKFCM_LAMBDA_SIGMA_{l}_{s}.csv",
    "SPMVFCM_lambda": lambda l, s: f"MVJFCM/Lamda/FINAL_RESULT_no_kernal_MVJFC_parameter_{l}.csv",
    "SPWPMVKFCM_lambda_sigma": lambda l, s: f"WMVKFCM/LAMDA_SIGMA/FINAL_RESULT_900_1_MVKFCM_LAMBDA_SIGMA_{l}_{s}.csv",
    "SPWPMVFCM_lambda": lambda l, s: f"WMVJFCM/FINAL_RESULT_no_kernal_MVJFC_parameter_{l}.csv",
}


def sens_dir():
    return OUT / f"sensitivity_seed_{CFG['sensitivity']['seed']}"


def select_best():
    import pandas as pd
    metric = CFG["sensitivity"]["selection_metric"]
    best, table = {}, {}
    for name, spec in CFG["sensitivity"]["stage1"].items():
        rows = []
        for g in spec["grid"]:
            lam, sig = (float(g[0]), float(g[1])) if isinstance(g, list) else (float(g), None)
            f = sens_dir() / SENS_FILES[name](lam, sig)
            df = pd.read_csv(f, index_col=0)
            rows.append({"lambda": lam, "sigma": sig, **{k: float(df[k].iloc[-1]) for k in METRICS}})
        top = max(rows, key=lambda r: r[metric])            # first in grid order on ties
        best[spec["method"]] = {"lambda": top["lambda"], "sigma": top["sigma"], f"mean_{metric}": top[metric]}
        table[spec["method"]] = rows
    out = {"selection": f"highest mean {metric} over the {CFG['sensitivity']['image_list']} images",
           "best": best, "stage1": table}
    (sens_dir() / "best_parameters.json").write_text(json.dumps(out, indent=1) + "\n")
    return best


def sensitivity():
    sens, t = CFG["sensitivity"], CFG["cpu"]["other_threads"]
    stage1 = [(f"sens_stage1_{name}", {"kind": "sens", "experiment": name, "grid": v["grid"]}, t)
              for name, v in sens["stage1"].items()]
    failed = schedule(stage1, "sensitivity_stage1")
    if failed:
        raise SystemExit(f"stage 1 failed: {failed}")
    best = select_best()
    stage2 = [(f"sens_stage2_{name}", {"kind": "sens", "experiment": name,
                                       "d_params": [best[m]["lambda"], best[m]["sigma"]]}, t)
              for name, m in sens["stage2"].items()]
    failed = schedule(stage2, "sensitivity_stage2")
    if failed:
        raise SystemExit(f"stage 2 failed: {failed}")


# ----------------------------------------------------------------------------- summary
def summary():
    import numpy as np
    import pandas as pd
    lines, rows = [f"# Results (configs/final.json), seeds {', '.join(map(str, CFG['seeds']))}\n\n"
                   "Proposed methods: P -> h -> M updates; dermatology: 9 views (R, G, B, L, a, b, black-hat, "
                   "morphological gradient, external gradient); X-ray: 4 views. Comparison methods: pixel intensities "
                   "(dermatology: R, G, B; X-ray: grey value).\n"], []
    for ds, title in (("derma", "Dermatology (900 images)"), ("covid", "COVID (207 images)"),
                      ("noncovid", "Non-COVID (32 images)")):
        for s in CFG["seeds"]:
            for m in CFG["methods"]:
                f = final_csv(s, ds, m)
                if not f.exists():
                    continue
                df = pd.read_csv(f, index_col=0)
                per = df.iloc[:-1][METRICS].astype(float)
                r = {"dataset": ds, "seed": s, "method": m, **{k: float(df[k].iloc[-1]) for k in METRICS},
                     "psnr_inf": int(np.isinf(per["PSNR"]).sum())}
                r["PSNR_finite"] = float(per["PSNR"][np.isfinite(per["PSNR"])].mean())
                rows.append(r)
        d = pd.DataFrame([r for r in rows if r["dataset"] == ds])
        if d.empty:
            continue
        lines.append(f"\n## {title}: mean ± std over seeds {CFG['seeds']}\n\n| Method | " + " | ".join(METRICS)
                     + " |\n|" + "---|" * (len(METRICS) + 1) + "\n")
        for m in CFG["methods"]:
            g = d[d.method == m]
            if g.empty:
                continue
            cells = []
            for k in METRICS:
                v = g["PSNR_finite"] if k == "PSNR" and np.isinf(g["PSNR"]).any() else g[k]
                cells.append(f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" + ("*" if k == "PSNR" and v is g["PSNR_finite"] else ""))
            lines.append(f"| {m} | " + " | ".join(cells) + " |\n")
        for s in CFG["seeds"]:
            lines.append(f"\n### {title}, seed {s}\n\n| Method | " + " | ".join(METRICS) + " |\n|"
                         + "---|" * (len(METRICS) + 1) + "\n")
            for _, r in d[d.seed == s].iterrows():
                lines.append(f"| {r.method} | " + " | ".join(f"{r[k]:.4f}" for k in METRICS) + " |\n")
    lines.append("\n\\* PSNR mean over the images with finite PSNR (a perfect segmentation gives infinite PSNR).\n")
    pd.DataFrame(rows).to_csv(OUT / "summary_main.csv", index=False)
    bp = sens_dir() / "best_parameters.json"
    if bp.exists():
        b = json.loads(bp.read_text())
        lines.append(f"\n## Parameter sensitivity (dermatology, 900 images, seed {CFG['sensitivity']['seed']})\n")
        lines.append(f"\nBest lambda / sigma ({b['selection']}):\n\n| Method | lambda | sigma | mean IoU |\n|---|---|---|---|\n")
        for m, v in b["best"].items():
            lines.append(f"| {m} | {v['lambda']:g} | {'' if v['sigma'] is None else format(v['sigma'], 'g')} | "
                         f"{v['mean_IoU']:.4f} |\n")
        for m, tab in b["stage1"].items():
            lines.append(f"\n### {m}: lambda / sigma stage\n\n| lambda | sigma | " + " | ".join(METRICS) + " |\n|"
                         + "---|" * (len(METRICS) + 2) + "\n")
            for r in tab:
                lines.append(f"| {r['lambda']:g} | {'' if r['sigma'] is None else format(r['sigma'], 'g')} | "
                             + " | ".join(f"{r[k]:.4f}" for k in METRICS) + " |\n")
        d_dirs = {"SPMVKFCM": "MVKFCM/FUZZY", "SPMVFCM": "MVJFCM/FUZZY", "SPWPMVKFCM": "WMVKFCM/FUZZY",
                  "SPWPMVFCM": "WMVJFCM/FUZZY"}
        for m, sub in d_dirs.items():
            files = sorted((sens_dir() / sub).glob("*.csv"), key=lambda p: float(p.stem.split("_")[-1]))
            if not files:
                continue
            lines.append(f"\n### {m}: d stage (at the best lambda / sigma)\n\n| d | " + " | ".join(METRICS) + " |\n|"
                         + "---|" * (len(METRICS) + 1) + "\n")
            for f in files:
                df = pd.read_csv(f, index_col=0)
                lines.append(f"| {f.stem.split('_')[-1]} | " + " | ".join(f"{float(df[k].iloc[-1]):.4f}" for k in METRICS)
                             + " |\n")
    (OUT / "SUMMARY.md").write_text("".join(lines))
    print("".join(lines))


# ----------------------------------------------------------------------------- command line
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["main", "check", "sensitivity", "summary", "one", "job"])
    ap.add_argument("spec", nargs="*", help="one: DATASET METHOD (derma / covid / noncovid; a method of configs/final.json)")
    ap.add_argument("--seed", type=int, default=CFG["examples"]["seed"], help="one: the seed (default 42)")
    a = ap.parse_args()
    if a.step == "job":
        run_job(json.loads(a.spec[0]))
    elif a.step == "one":
        if len(a.spec) != 2 or a.spec[0] not in CFG["datasets"] or a.spec[1] not in CFG["methods"]:
            ap.error(f"one: DATASET METHOD with DATASET in {CFG['datasets']} and METHOD in {CFG['methods']}")
        os.environ.update(CUDA_VISIBLE_DEVICES="", CLUSTPY_DEVICE="cpu", MPLBACKEND="Agg")
        run_job({"kind": "main", "seed": a.seed, "dataset": a.spec[0], "method": a.spec[1]})
        print(final_csv(a.seed, a.spec[0], a.spec[1]))
    elif a.step == "main":
        OUT.mkdir(parents=True, exist_ok=True)
        copy_examples()
        failed = schedule(main_jobs(), "main")
        problems = check_main()
        sys.exit(1 if failed or problems else 0)
    elif a.step == "check":
        sys.exit(1 if check_main() else 0)
    elif a.step == "sensitivity":
        sensitivity()
    else:
        summary()
