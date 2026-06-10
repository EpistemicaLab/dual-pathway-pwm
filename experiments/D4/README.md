# Experiment D4

**Paper section:** §5.5 / per-coordinate ablation (u-coord hierarchy)

## Summary

D4 -- Translator-Network Capacity probe.

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/D4/run.sh
# Outputs ./results-D4.json

# Or directly:
python experiments/D4/run.py --out results-D4.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
