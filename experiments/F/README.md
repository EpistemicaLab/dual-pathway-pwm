# Experiment F

**Paper section:** §5.8 main result (iVAE conditioned on u — the only method that closes §5.7 gap, CKA=0.93)

## Summary

F -- iVAE with intervention u as auxiliary variable.

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/F/run.sh
# Outputs ./results-F.json

# Or directly:
python experiments/F/run.py --out results-F.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
