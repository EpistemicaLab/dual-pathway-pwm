# Experiment D2

**Paper section:** §5.5 / Theorem 4.3 ablation

## Summary

D2 -- Representational Similarity Analysis (RSA) diagnostic.

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/D2/run.sh
# Outputs ./results-D2.json

# Or directly:
python experiments/D2/run.py --out results-D2.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
