# Experiment C

**Paper section:** §5.8 (twelve-method alignment grid: method C)

## Summary

C -- Weight-shared g_theta dynamics core with MSE alignment.

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/C/run.sh
# Outputs ./results-C.json

# Or directly:
python experiments/C/run.py --out results-C.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
