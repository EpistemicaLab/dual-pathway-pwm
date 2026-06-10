# Dual-Pathway Physical World Models

Code release for the paper:

> **A Dual-Pathway Theory of Physical World Models: Why Symbolic Laws Need Intervention, and Why Cross-Pathway Consistency is the Mechanism (Not the Loss)**
> arXiv preprint, 2026.

The paper argues that physical-knowledge transfer in modern world models is
governed not by the form of the cross-pathway alignment loss but by what the
coupling's *target* encodes — and that closing the asymmetric-pathway gap
between passive observation and active intervention requires conditioning the
representation on the intervention variable, not adding more loss terms.

This repository reproduces the empirical claims that ground that argument.

## Repository layout

```
dual-pathway-pwm/
├── README.md                  this file
├── LICENSE                    MIT
├── pyproject.toml             package metadata
├── requirements.txt           pinnable dependency manifest
├── docs/
│   └── status.md              honest reproducer status & roadmap (READ ME)
├── src/
│   └── pwm/                   shared library skeleton (T2: refactor pending)
└── experiments/
    ├── README.md              maps each experiment to its paper section
    ├── A/  AC/  B/  C/        twelve-method alignment grid (§5.8)
    ├── D/ D1/ D2/ D3/ D4/     identifiability ablations  (§5.5, Theorems 4.2–4.6)
    ├── E/ F/ F0/ G/ H/        further alignment / asymmetry experiments
    ├── I/                     do-intervention vs passive  (§5.5)
    └── J/                     calibration-fair target-not-loss sweep  (§5.4)
```

## What this release contains

| | |
|---|---|
| ✅ | All 16 experiment scripts that produced the paper's numbers (`run.py` + `config.json` + `run.sh`). |
| ✅ | A path-portable launcher that defaults to the repo root and a local `.venv`. |
| ✅ | `requirements.txt` and `pyproject.toml`. |
| 🟡 | Skeleton `src/pwm/` package. The 16 experiments still carry their own copy of common utilities; refactoring is the next milestone (**T2**). |
| 🔴 | End-to-end reproduction of paper tables on a clean environment is **not yet certified** (**T3** — see [`docs/status.md`](docs/status.md)). |

We chose to publish the script-set immediately, before T2/T3 ship, so readers
can audit the actual code rather than wait for a curated demo. The cost of
that choice is that, until T3 lands, you may need to debug environment issues
to reproduce a given table.

## Quick start

```bash
# 1. Clone and set up a venv at the repo root.
git clone https://github.com/EpistemicaLab/dual-pathway-pwm.git
cd dual-pathway-pwm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run a single experiment. Outputs ./results-<EXP>.json.
./experiments/J/run.sh         # §5.4 calibration-fair sweep (target-not-loss)

# 3. Or run a smoke / faster cell directly via Python:
python experiments/J/run.py --smoke --out smoke.json
```

> If `./experiments/J/run.sh` fails because no `.venv` is found, that's the
> launcher behaving correctly — the message tells you to create one. If it
> fails further along (e.g. JAX device errors), see
> [`docs/status.md`](docs/status.md) for the engineering caveats and the open
> issues we're tracking.

GPU is recommended for §5.4 (experiment `J`) and the §5.8 alignment grid (`A`,
`B`, `C`, `F`, `F0`, `G`). The §5.5 identifiability cells (`I`, `D*`) are
practical on CPU.

## Headline results (from the paper)

These are the numbers each experiment was authored to produce. T3 will
certify that a fresh checkout reproduces them; until then, treat them as
the authors' reported numbers, not as repo guarantees.

| Section | Experiment | Claim | Number |
|---------|-----------|-------|-------:|
| §5.4 | `J` | Target swap (positive control) | ΔAUROC = +0.298 (PASS) |
| §5.4 | `J` | Loss swap BCE↔MSE (negative control) | ΔAUROC = -0.001 (EQUIVALENCE) |
| §5.5 | `I` | Passive structure-ID baseline | 0.50 (chance) |
| §5.5 | `I` | + do-intervention | 0.735 |
| §5.7 | `H`, `E` | LC alone on asymmetric pathways | task-eq ≠ repr-eq (negative) |
| §5.8 | `F` | iVAE conditioned on u (primary method) | CKA = 0.93, transfer = 0.99 |
| §5.8 | `F0` | u → 0 ablation | CKA = 0.46 |

See [`experiments/README.md`](experiments/README.md) for the full map of
experiments to paper sections.

## Citation

```bibtex
@article{dual_pathway_pwm_2026,
  title  = {A Dual-Pathway Theory of Physical World Models: Why Symbolic
            Laws Need Intervention, and Why Cross-Pathway Consistency
            is the Mechanism (Not the Loss)},
  author = {Authors},
  journal= {arXiv preprint},
  year   = {2026},
  note   = {Anonymous during peer review}
}
```

The arXiv ID will be added once the preprint is announced.

## License

MIT — see [LICENSE](LICENSE).

## Maintainer

This repository is part of [EpistemicaLab](https://epistemicalab.github.io/),
an independent research lab. Reproducer issues, fixes, and extensions are
welcome via GitHub issues and pull requests.
