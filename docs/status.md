# Reproducer status & roadmap

This document tracks the engineering status of `dual-pathway-pwm` as a public
reproducer for the paper. We are explicit about what is and is not validated.

## Current state — `v0.1.0a1` (alpha)

| Area | Status |
|------|--------|
| Paper LaTeX source | Hosted on arXiv (see top-level `README.md`) -- **not in this repo by design**: readers are not expected to read the source, and arXiv is the canonical preprint host. |
| Per-experiment Python scripts | ✅ Published. 16 directories under `experiments/` covering §5.4 / §5.5 / §5.7 / §5.8. Each contains `run.py` + `config.json` + `run.sh`. |
| Per-experiment launchers | ✅ Path-portable. `run.sh` defaults to the repo root and a `.venv` at the same root; override via `REPO`, `VENV`, `OUT` env vars. |
| Top-level dependency manifest | ✅ `requirements.txt` and `pyproject.toml` committed. |
| Shared simulator / encoder library (`src/pwm/`) | 🟡 **Skeleton only.** The 16 experiment scripts each carry their own copy of common code (Duffing simulator, encoder MLPs, alignment metrics). Refactoring into `src/pwm/` is **T2** below. |
| End-to-end smoke validation on a fresh environment | 🔴 **Not yet performed.** No CI run. The scripts were last validated on the authors' EC2 with `JAX[cuda12]` and a specific layout; we do not yet certify that a reader can `pip install -r requirements.txt && bash experiments/A/run.sh` and reproduce the paper's headline numbers. |
| Numerical reproduction of paper tables | 🔴 **Not yet certified.** §5.4's `EQUIVALENCE` (ΔAUROC = -0.001) and `PASS` (+0.298) numbers, §5.5's 0.50 → 0.735 lift, and §5.8's CKA = 0.93 / 0.46 ablation come from a controlled run on the authors' machine. Re-running on a fresh environment is **T3** below. |

## Why we publish before T2/T3 finish

Releasing the script-set immediately lets:

* reviewers and collaborators read the actual code while reading the paper, rather than wait for a polished release;
* readers cite the repo (with this status doc) honestly, not believe a curated demo is the full method;
* downstream researchers fork, modify, and probe assumptions without us being a bottleneck.

We will not silently rewrite history under the same tag. T2 and T3 will land as `v0.2.0` and `v1.0.0` respectively, with tagged releases.

## T2 -- Engineering polish (next)

* Factor common Duffing simulator into `src/pwm/sim/duffing.py`.
* Factor common encoder pair into `src/pwm/models/dual_pathway.py`.
* Factor alignment metrics (CKA, transfer probe, identifiability score) into `src/pwm/metrics/`.
* Replace per-experiment local copies with imports from `pwm.*`.
* Add a `Dockerfile` pinning JAX/CUDA and Python versions.
* Add CI smoke tests: at least the CPU-runnable subset (likely `D1`, `D2`, `J`'s `--smoke` mode).

## T3 -- End-to-end validation (after T2)

* Rerun every experiment from a fresh `.venv` using `requirements.txt`.
* Diff the produced `results-*.json` against the values cited in the paper.
* Document expected wall-time, GPU memory, and disk-checkpoint sizes per experiment.
* Publish the validation log as `docs/validation-v1.0.md`.

## Reporting issues

If a script fails to run or numbers drift on a clean install, please open an
issue at https://github.com/EpistemicaLab/dual-pathway-pwm/issues with:

1. The exact command you ran.
2. Your Python / JAX / GPU stack (`python --version`, `pip freeze | grep jax`, `nvidia-smi` if applicable).
3. The full traceback or the diff between observed numbers and paper numbers.

We will not silently fix and force-push; corrections land as commits with `fix(<exp>): ...` messages.
