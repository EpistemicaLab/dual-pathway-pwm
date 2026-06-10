# Experiment G

**Paper section:** §5.8 conditional baseline grid

## Summary

G -- InfoNCE / SimCLR contrastive (van den Oord 2018).

## Run

```bash
# From the repo root, with a venv at .venv:
./experiments/G/run.sh
# Outputs ./results-G.json

# Or directly:
python experiments/G/run.py --out results-G.json [--smoke]
```

## Files

* `run.py`     -- the experiment.
* `config.json` -- hyperparameters and seeds.
* `run.sh`     -- launcher (sets up `.venv` paths, calls `run.py`).

## Notes

This script was last validated on the authors' EC2 with `JAX[cuda12]`. See
[`../../docs/status.md`](../../docs/status.md) for the public-release
validation status.
