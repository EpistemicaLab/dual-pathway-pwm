# Experiment H

**Paper section:** §5.7 determinism logbook (task-equivalence ≠ representation-equivalence)

## Summary

H -- Optimal Transport / Sinkhorn divergence alignment (Cuturi 2013, Genevay 2018).

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/H/run.sh
# Outputs ./results-H.json

# Or directly:
python experiments/H/run.py --out results-H.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
