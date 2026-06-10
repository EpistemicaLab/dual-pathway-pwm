"""D2 -- Representational Similarity Analysis (RSA) diagnostic.

Reads hyperparameters from config.json in the same dir. Trains the §5.7 dual-pathway
baseline at lambda_C=0.0 (pre-alignment condition: no L_C term), extracts paired
(zA, zB) latents on a held-out eval set, then computes Spearman rank-correlation
between the upper-triangle pairwise-distance vectors of zA and zB. Reports both
Euclidean and cosine variants with 95% bootstrap CIs.

RSA is rank-based and so robust to monotone (and many arbitrary-monotone)
transformations of either pathway. It complements D1 (CKA): D1 picks up
basis-aligned linear similarity (and, in RBF form, also nonlinear smooth
distortions); D2 picks up second-order distance-rank preservation.

This is a diagnostic, not a PASS gate -- numbers feed the `decide` phase's tree.

Honest-negative outcomes:
  * n_eval_paired < 100: insufficient samples (matches D1 floor).
  * tie_fraction > 0.25 in either pdist vector: Spearman degrades; result emits
    a warning field but still records the rho values for transparency.

The runner reuses D1's build_baseline_pair() to keep the (zA, zB) pair byte-equivalent
across diagnostics (same hyperparameters in config.json).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


HERE = Path(__file__).resolve().parent
D1_DIR = HERE.parents[0] / "D1"
E1_DIR = HERE.parents[1] / "E1-numerical-tightness"
W4_DIR = HERE.parents[1] / "W4-learned-B"
W5_DIR = HERE.parents[1] / "W5-C13-LC-ablation"
for p in (D1_DIR, E1_DIR, W4_DIR, W5_DIR):
    sys.path.insert(0, str(p))


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def rsa(X: np.ndarray, Y: np.ndarray, metric: str = "euclidean") -> float:
    """Spearman rank-correlation between pairwise-distance vectors of X and Y."""
    dx = pdist(X, metric=metric)
    dy = pdist(Y, metric=metric)
    if dx.size == 0 or dy.size == 0:
        return float("nan")
    rho, _ = spearmanr(dx, dy)
    return float(rho)


def tie_fraction(X: np.ndarray, metric: str = "euclidean") -> float:
    """Fraction of pdist entries that share their value with at least one other entry.

    Spearman degrades when ties dominate; if this exceeds ~0.25 we record a warning.
    """
    d = pdist(X, metric=metric)
    if d.size == 0:
        return 0.0
    _, counts = np.unique(d, return_counts=True)
    tied = int((counts[counts > 1]).sum())
    return float(tied) / float(d.size)


def bootstrap_ci_rsa(X: np.ndarray, Y: np.ndarray, metric: str, n_boot: int, seed: int,
                     alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap 95% CI for RSA Spearman by resampling rows of (X, Y) jointly."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = rsa(X[idx], Y[idx], metric=metric)
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Reuse D1's baseline-pair builder so the (zA, zB) substrate is identical
    # across diagnostics (modulo bootstrap RNG).
    import run as d1_run  # type: ignore[import-not-found]  # D1/run.py on sys.path

    t0 = time.perf_counter()
    z_a, z_b, meta = d1_run.build_baseline_pair(cfg)
    n = z_a.shape[0]

    if n < 100:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": f"n_eval_samples={n} < 100; RSA estimate is uninformative.",
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    print(f"[D2] computing RSA on n={n}, dimA={meta['dim_A']}, dimB={meta['dim_B']}…",
          flush=True)
    rho_euc = rsa(z_a, z_b, metric="euclidean")
    rho_cos = rsa(z_a, z_b, metric="cosine")
    n_boot = int(cfg.get("n_bootstrap", 1000))
    seed = int(cfg.get("bootstrap_seed", 24690))
    euc_lo, euc_hi = bootstrap_ci_rsa(z_a, z_b, "euclidean", n_boot=n_boot, seed=seed)
    cos_lo, cos_hi = bootstrap_ci_rsa(z_a, z_b, "cosine", n_boot=n_boot, seed=seed + 1)

    tf_a_euc = tie_fraction(z_a, metric="euclidean")
    tf_b_euc = tie_fraction(z_b, metric="euclidean")
    tf_a_cos = tie_fraction(z_a, metric="cosine")
    tf_b_cos = tie_fraction(z_b, metric="cosine")
    max_tie = max(tf_a_euc, tf_b_euc, tf_a_cos, tf_b_cos)
    tie_warning = max_tie > 0.25

    # Spec interpretation table -> decision-tree input. Take the max of euc/cos.
    rho_max = max(rho_euc, rho_cos)
    if rho_max >= 0.70:
        interp = "strong_shared_geometry__rank_preserving"
    elif rho_max >= 0.40:
        interp = "partial_alignment"
    else:
        interp = "geometries_genuinely_differ__information_mismatch"

    result = {
        "verdict": "diagnostic_complete",
        **meta,
        "euclidean_rsa": rho_euc,
        "euclidean_rsa_ci_95": [euc_lo, euc_hi],
        "cosine_rsa": rho_cos,
        "cosine_rsa_ci_95": [cos_lo, cos_hi],
        "n_bootstrap": n_boot,
        "interpretation": interp,
        "tie_fraction": {
            "zA_euclidean": tf_a_euc, "zB_euclidean": tf_b_euc,
            "zA_cosine": tf_a_cos, "zB_cosine": tf_b_cos,
            "max": max_tie,
        },
        "tie_warning": tie_warning,
        "wall_s": round(time.perf_counter() - t0, 1),
        "git_head_at_run": _git_sha(HERE),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
