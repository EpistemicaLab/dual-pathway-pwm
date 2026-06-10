# Experiment D

**Paper section:** §5.5 + Theorem 4.1 (calibration-fair encoder sweep used to fit δ_1 / ε_2 constants)

## Summary

D -- JEPA cross-prediction with stop-gradient + EMA target encoders.

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/D/run.sh
# Outputs ./results-D.json

# Or directly:
python experiments/D/run.py --out results-D.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
