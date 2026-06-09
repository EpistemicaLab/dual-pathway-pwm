# Dual-Pathway Physical World Models

Code and data release for the paper:

> **A Dual-Pathway Theory of Physical World Models: Why Symbolic Laws Need Intervention, and Why Cross-Pathway Consistency is the Mechanism (Not the Loss)**
> arXiv preprint, 2026.

## Status

This repository will be populated with:
- experiment code (Duffing oscillator do-intervention sweeps, vision encoder ablations, iVAE alignment baselines)
- data-generating processes (passive observation pathway B, active intervention pathway A)
- pre-trained checkpoints (the calibration-fair sweep of §5.4 and §5.8 alignment grid)
- reproduction scripts for all reported numbers

Full release upon arXiv announcement and paper publication.

## Paper highlights

- **Target-not-loss hypothesis (§5.4)**: physical-knowledge transfer is governed by what the cross-pathway coupling's *target* encodes, not by the surrogate-loss form. Loss-family swap (BCE↔MSE) yields ΔAUROC = -0.001 (EQUIVALENCE); target swap yields +0.298 (PASS).
- **Dual-pathway architecture**: Aristotelian passive-observation pathway B + Newtonian active-intervention pathway A, coupled by a Δ-inequality (Theorem 4.1).
- **Identifiability under do-intervention (§5.5)**: structure-ID lifts from chance (0.50) under passive observation to 0.735 under do-intervention; SINDy underperforms chance.
- **Twelve-method alignment ablation (§5.8)**: only iVAE conditioned on intervention variable u achieves CKA = 0.93 and transfer = 0.99; ablation u→0 drops CKA to 0.46.

## Citation

```bibtex
@article{dual_pathway_pwm_2026,
  title={A Dual-Pathway Theory of Physical World Models: Why Symbolic Laws Need Intervention, and Why Cross-Pathway Consistency is the Mechanism (Not the Loss)},
  author={Authors},
  journal={arXiv preprint},
  year={2026},
  note={Anonymous during peer review}
}
```

## License

MIT -- see [LICENSE](LICENSE).
