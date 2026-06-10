# Experiment I

**Paper section:** §5.5 do-intervention vs passive observation (structure-ID 0.50 → 0.735)

## Summary

I -- MINE / Donsker-Varadhan MI lower bound (Belghazi+ ICML 2018).

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/I/run.sh
# Outputs ./results-I.json

# Or directly:
python experiments/I/run.py --out results-I.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
