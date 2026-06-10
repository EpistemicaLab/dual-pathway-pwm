"""Dual-pathway physical world models -- shared utilities.

This package will host shared simulators, encoders, and metrics extracted
from the per-experiment scripts under ``experiments/``. In v0.1.0a1 (the
current release) the experiments are self-contained and import only
``numpy``/``jax``/``sklearn`` directly. T2 of the reproducer roadmap
(see ``docs/status.md``) will refactor common code into this package.

The arXiv paper that this code accompanies introduces:

  * The target-not-loss hypothesis (Theorem 4.1, evaluated in
    experiment ``J``): physical-knowledge transfer is governed by what
    the cross-pathway coupling's *target* encodes, not by the
    surrogate-loss form.
  * A do-intervention identifiability result (Theorem 4.2, evaluated
    in experiments ``I`` and ``D*``): SINDy plateaus at chance under
    passive observation; do-intervention lifts structure-ID to 0.735.
  * A calibration-fair 12-method alignment grid (experiments
    ``A``..``J``, ``F0``, ``AC``): only iVAE conditioned on the
    intervention variable u closes the asymmetric-pathway gap
    (CKA = 0.93, transfer = 0.99); ablation u -> 0 drops CKA to 0.46.
"""

__version__ = "0.1.0a1"
