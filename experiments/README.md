# Experiments index

Each subdirectory is a standalone experiment with `config.json`, `run.py`, and
`run.sh`. Run any of them via:

```bash
# From the repo root:
./experiments/<EXP>/run.sh
# Outputs ./results-<EXP>.json
```

Override env vars to control where outputs land or which venv is used:

```bash
REPO=$PWD VENV=/path/to/venv OUT=/path/to/results.json ./experiments/A/run.sh
```

> **Validation status:** these scripts were last validated on the authors'
> EC2 with `JAX[cuda12]`. End-to-end reproduction on a clean environment is
> the T3 milestone in [`../docs/status.md`](../docs/status.md). Until T3 ships,
> readers should expect to debug environment issues.

## Map: paper section → experiment directories

The paper's empirical contribution is organised as a calibration-fair sweep
of alignment methods plus a pair of identifiability anchors. Each row below
corresponds to one paper claim and the experiment directory that produced it.

### §5.4 — Target-not-loss hypothesis (the headline)

The single falsifiable prediction: *physical-knowledge transfer is governed by
what the cross-pathway coupling's target encodes, not by the surrogate-loss
form*.

| Experiment | Paper role |
|------------|-----------|
| `J` | The calibration-fair sweep. Loss-family swap (BCE↔MSE) yields ΔAUROC = -0.001 (`EQUIVALENCE`); target swap yields +0.298 (`PASS`). |

### §5.5 — Do-intervention identifiability

| Experiment | Paper role |
|------------|-----------|
| `I` | Structure-ID lifts from chance (0.50) under passive observation to 0.735 under do-intervention. |
| `D` / `D1` / `D2` / `D3` / `D4` | Per-coordinate and noise-class ablations of the identifiability bound (Theorem 4.2 / 4.3 / 4.6). |

### §5.7 — Asymmetric-pathway failure mode

| Experiment | Paper role |
|------------|-----------|
| `H` | Determinism logbook -- the "task-equivalence ≠ representation-equivalence" negative result that motivates §5.8. |
| `E` | Real-trajectory diagnostic: `LC` alone fails on asymmetric pathways. |

### §5.8 — Twelve-method alignment grid

The paper's main claim is that, among twelve alignment architectures, only
iVAE conditioned on the intervention variable u closes the §5.7 gap.

| Experiment | Method (one-line) |
|------------|-------------------|
| `A` | DeepCCA loss with INDEPENDENT per-pathway projection MLPs. |
| `AC` | DeepCCA with SHARED projection (control for A). |
| `B` | Method B baseline. |
| `C` | Method C baseline. |
| `F` | iVAE conditioned on u (the **only** method that achieves CKA = 0.93 and transfer = 0.99). |
| `F0` | iVAE without u-conditioning (the ablation that drops CKA to 0.46). |
| `G` | Conditional baseline grid. |

### Theory anchors

| Experiment | Paper role |
|------------|-----------|
| `D` | Calibration-fair encoder sweep used to fit the constants in Theorem 4.1's δ_1 / ε_2. |

### Cross-method audits

The `*_audit` directories (if present in your fork) re-run prior cells against
new fixtures to detect drift. They are not part of the public release at
v0.1.0a1.

## Per-experiment files

* `run.py` -- the actual experiment. Self-contained NumPy/JAX code.
* `config.json` -- hyperparameters and seeds. Edit here to sweep.
* `run.sh` -- launcher that activates `$REPO/.venv` and runs `run.py`. Override
  `REPO`, `VENV`, `OUT` via env vars.
