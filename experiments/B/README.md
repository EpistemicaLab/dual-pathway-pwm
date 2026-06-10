# Experiment B

**Paper section:** §5.8 (twelve-method alignment grid: method B)

## Summary

Method B — Procrustes-MSE alignment (Schönemann 1966).

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/B/run.sh
# Outputs ./results-B.json

# Or directly:
python experiments/B/run.py --out results-B.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
