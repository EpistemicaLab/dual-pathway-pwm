# Experiment A

**Paper section:** §5.8 (twelve-method alignment grid: DeepCCA, independent g_A/g_B)

## Summary

Method A — DeepCCA loss with INDEPENDENT per-pathway projection MLPs.

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/A/run.sh
# Outputs ./results-A.json

# Or directly:
python experiments/A/run.py --out results-A.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
