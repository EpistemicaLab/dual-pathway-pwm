# Experiment F0

**Paper section:** §5.8 ablation (iVAE without u -- CKA drops to 0.46)

## Summary

F -- iVAE with intervention u as auxiliary variable.

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/F0/run.sh
# Outputs ./results-F0.json

# Or directly:
python experiments/F0/run.py --out results-F0.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
