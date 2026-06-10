# Experiment D3

**Paper section:** §5.5 / Theorem 4.6 ablation

## Summary

D3 -- Linear-Probe-Transfer diagnostic.

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/D3/run.sh
# Outputs ./results-D3.json

# Or directly:
python experiments/D3/run.py --out results-D3.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
