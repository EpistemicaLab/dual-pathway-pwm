# Experiment E

**Paper section:** §5.7 real-trajectory diagnostic (asymmetric-pathway failure mode)

## Summary

E -- Equivariant by construction (POST-HOC orbit-averaging surrogate).

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/E/run.sh
# Outputs ./results-E.json

# Or directly:
python experiments/E/run.py --out results-E.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
