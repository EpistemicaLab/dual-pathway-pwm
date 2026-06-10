# Experiment AC

**Paper section:** §5.8 (twelve-method alignment grid: DeepCCA, shared g_theta -- control for A)

## Summary

AC -- DeepCCA loss + weight-shared g_theta core.

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/AC/run.sh
# Outputs ./results-AC.json

# Or directly:
python experiments/AC/run.py --out results-AC.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
