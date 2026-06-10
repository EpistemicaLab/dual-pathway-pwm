# Experiment J

**Paper section:** §5.4 calibration-fair target-not-loss sweep (THE HEADLINE: loss swap = EQUIVALENCE, target swap = PASS)

## Summary

J -- Translator-network alignment with cycle consistency.

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/J/run.sh
# Outputs ./results-J.json

# Or directly:
python experiments/J/run.py --out results-J.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
