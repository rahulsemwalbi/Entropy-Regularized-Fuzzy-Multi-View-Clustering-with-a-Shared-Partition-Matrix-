# Entropy-Regularized Fuzzy Multi-View Clustering with a Shared Partition Matrix 

Code and data for *Entropy-Regularized Fuzzy Multi-View Clustering with a Shared Partition Matrix*.

Four proposed methods:

- **SPMVFCM**: shared partition, view weights;
- **SPMVKFCM**: shared partition, Gaussian kernel, view weights;
- **SPWPMVFCM**: shared partition, point-wise weights;
- **SPWPMVKFCM**: shared partition, Gaussian kernel, point-wise weights.

They are compared with k-means, FCM, DCN, DEC, DipEncoder and DipDECK on three datasets:

- ISIC 2016 dermoscopic images;
- chest X-rays, COVID and non-COVID.

## Repository layout

```
configs/final.json           all settings
images.txt                   ISIC image IDs, in processing order
docs/METHODS.md              models, update rules and settings
experiments/derma/           ISIC: setup.py and one file per method
experiments/xray/            COVID / non-COVID: setup.py and one file per method
experiments/sensitivity/     parameter sensitivity: setup.py and one file per experiment
spmvfcm/models.py            update rules of the four models
spmvfcm/methods.py           the four methods: iterations P -> h -> M, objective, stopping test
spmvfcm/views.py             dermatology views of the proposed methods
spmvfcm/xray_adapters.py     X-ray shared-view models with the methods of spmvfcm/methods.py
spmvfcm/runner.py            runs one method on one dataset, or one sensitivity experiment
scripts/run_experiments.py   all methods, parameter sensitivity, summary, checks, single runs
scripts/run_all.sh           all methods -> parameter sensitivity -> summary
scripts/prepare_data.py      rebuilds data/ from the dataset zip files
tests/                       tests on synthetic images
```

## Installation

```bash
conda create -n spmvfcm python=3.11 -y && conda activate spmvfcm
pip install -r requirements.txt
```

Everything runs on the CPU.

## Running

```bash
python scripts/run_experiments.py main                         # all methods
bash scripts/run_all.sh                                        # all methods → parameter sensitivity → summary
python scripts/run_experiments.py one derma SPMVFCM --seed 42  # a single method
```

**Other steps:**

```bash
python scripts/run_experiments.py sensitivity   # parameter sensitivity only
python scripts/run_experiments.py summary       # results/final/SUMMARY.md
python scripts/run_experiments.py check         # checks of the all runs
```

**Names for `one`:**

- datasets: `derma`, `covid`, `noncovid`;
- methods: `SPMVFCM SPMVKFCM SPWPMVFCM SPWPMVKFCM kmeans fcm dcn dec dipencoder dipdeck`.

Runs are started in parallel within `cpu.thread_budget` of `configs/final.json` (default 120 threads); set it to the machine's number of cores. On CPU, the proposed methods take 7–25 minutes per seed on the 900 dermatology images, and the dermatology deep-learning runs 5–13 hours each. `main` takes about 13 hours and `sensitivity` about 8 hours.

## Settings (`configs/final.json`)

**Proposed methods**


| Setting | Parameters |
|---|---|
| Fuzzifier | d |
| Clusters | C  |
| Iterations | at most 40 |
| Stopping test | \|Θ(t) − Θ(t−1)\| < ε |
| Dermatology views | 9: R, G, B, L, a, b, black-hat, morphological gradient, external gradient. |
| X-ray views | 4: gamma-corrected image, black-hat, morphological gradient, external gradient. |

- **Example images:** ISIC_0001191, ISIC_0005555, ISIC_0003346, ISIC_0005564, ISIC_0002093.


## Reproducing the results exactly

The per-image CSVs are identical byte for byte when:

1. you use Python 3.11 with the versions in `requirements.txt` (torch 2.15.0, the standard PyPI Linux build, run on the CPU);
2. you keep `deep_threads: 4` and `other_threads: 1` in `configs/final.json`. The thread count can change the last digits of the deep-learning results; `thread_budget` only sets how many runs go in parallel;
3. you run on the CPU (the default).

On a different CPU type, deep-learning results may differ in the last digits.

## Tests

```bash
pytest -m "not slow"      # methods, views and complete runs on synthetic images (about 1 minute)
pytest -m slow            # one deep-learning run on synthetic images (about 10 minutes)
```

The tests never read the datasets.

## License

Code: MIT License (see [LICENSE](LICENSE)).
