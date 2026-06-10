"""D3 -- Linear-Probe-Transfer diagnostic.

Trains per-bit logistic regression on zA to predict the 11-bit ORACLE_MASKS label,
then evaluates the SAME probe (no retrain) on zB to measure linear transfer. Symmetric
B→A direction also computed. The transfer_ratio drives the decide-phase top_3 routing:

  transfer_ratio ≥ 0.95   → basis-ambiguity        (top_3 = [A, AC, G])
  0.50 ≤ ratio ≤ 0.94    → partial linear compat.  (top_3 = [F, G, AC])
  transfer_ratio < 0.50   → linearly-incompatible  (top_3 = [F, C, AC])

Reuses D1's build_baseline_pair() to keep the (zA, zB) substrate byte-equivalent
across diagnostics. Recovers per-eval-sample 11-bit basis labels by re-deriving
the deterministic eval class assignments via e1._eval_set with the same args.

This is a diagnostic, not a PASS gate — output feeds the `decide` phase's tree.

Honest-negative outcomes:
  * n_eval_paired < 100: insufficient samples (matches D1/D2 floor).
  * within_auroc_mean < 0.6 (either direction): §5.7 baseline did not learn
    the task; this diagnostic is uninformative; abort per spec.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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


def _eval_labels_11bit(cfg: dict) -> np.ndarray:
    """Reproduce e1._eval_set's class assignment deterministically and look up
    each sample's 11-bit ORACLE_MASKS basis label.

    e1._eval_set iterates ALL_CLASSES × c_grid × k_grid × seeds with seed_base=50_000.
    Class index for sample i is determined by (ALL_CLASSES, c_grid, k_grid, n_eval_seeds).
    """
    import measure_bound_real_lc as e1
    if cfg["smoke"]:
        c_grid, k_grid = (0.05, 0.10), (1.0, 2.0)
        n_eval = 12
    else:
        c_grid, k_grid = (0.05, 0.10, 0.20), (1.0, 2.0, 5.0)
        n_eval = int(cfg["n_eval_seeds"])

    # Recover eval_cls from a deterministic re-call of e1._eval_set.
    # _eval_set is pure-numpy renderer; call is cheap (~secs, no encoder forward).
    _imgs, _feats, eval_cls = e1._eval_set(c_grid, k_grid, n_eval, seed_base=50_000)
    # Look up 11-bit mask per sample.
    y_bits = np.array(
        [e1.ORACLE_MASKS[e1.ALL_CLASSES[ci]] for ci in eval_cls],
        dtype=np.int8,
    )  # shape (n_eval_paired, 11)
    return y_bits


def _train_probes_per_bit(
    z_train: np.ndarray, y_train_bits: np.ndarray, C: float, max_iter: int
) -> tuple[list, np.ndarray]:
    """Fit one LogisticRegression per bit. Return (probes_list, skipped_bits_mask)."""
    n_bits = y_train_bits.shape[1]
    probes: list = [None] * n_bits
    skipped = np.zeros(n_bits, dtype=bool)
    for b in range(n_bits):
        y = y_train_bits[:, b]
        if y.min() == y.max():  # degenerate: all 0 or all 1
            skipped[b] = True
            continue
        clf = LogisticRegression(C=float(C), max_iter=int(max_iter))
        clf.fit(z_train, y)
        probes[b] = clf
    return probes, skipped


def _predict_proba_matrix(
    probes: list, z_test: np.ndarray, skipped: np.ndarray
) -> np.ndarray:
    """Return (n_test, n_bits) of P(y=1) per bit; NaN for skipped bits."""
    n_test = z_test.shape[0]
    n_bits = len(probes)
    out = np.full((n_test, n_bits), np.nan, dtype=np.float64)
    for b in range(n_bits):
        if skipped[b]:
            continue
        # Class label order in sklearn: clf.classes_; column for y=1 is where == 1.
        clf = probes[b]
        proba = clf.predict_proba(z_test)
        pos_col = int(np.argmax(clf.classes_ == 1))
        out[:, b] = proba[:, pos_col]
    return out


def _auroc_per_bit(
    proba: np.ndarray, y_test_bits: np.ndarray, skipped: np.ndarray, idx: np.ndarray
) -> np.ndarray:
    """AUROC per bit on the row-subset idx; NaN for bits where y_test[idx] is degenerate."""
    n_bits = proba.shape[1]
    out = np.full(n_bits, np.nan, dtype=np.float64)
    for b in range(n_bits):
        if skipped[b]:
            continue
        y = y_test_bits[idx, b]
        if y.min() == y.max():
            continue
        out[b] = roc_auc_score(y, proba[idx, b])
    return out


def _direction(
    z_train_src: np.ndarray, z_train_dst: np.ndarray,
    z_test_src: np.ndarray, z_test_dst: np.ndarray,
    y_train_bits: np.ndarray, y_test_bits: np.ndarray,
    C: float, max_iter: int, n_boot: int, boot_seed: int,
) -> dict:
    """Train probe on z_train_src, eval within=z_test_src, transfer=z_test_dst."""
    probes, skipped = _train_probes_per_bit(z_train_src, y_train_bits, C=C, max_iter=max_iter)
    proba_within = _predict_proba_matrix(probes, z_test_src, skipped)
    proba_transfer = _predict_proba_matrix(probes, z_test_dst, skipped)

    n_test = z_test_src.shape[0]
    full_idx = np.arange(n_test)
    within_per_bit = _auroc_per_bit(proba_within, y_test_bits, skipped, full_idx)
    transfer_per_bit = _auroc_per_bit(proba_transfer, y_test_bits, skipped, full_idx)
    valid = ~np.isnan(within_per_bit) & ~np.isnan(transfer_per_bit)
    n_valid = int(valid.sum())
    within_mean = float(np.nanmean(within_per_bit[valid])) if n_valid else float("nan")
    transfer_mean = float(np.nanmean(transfer_per_bit[valid])) if n_valid else float("nan")
    ratio_per_bit = transfer_per_bit / np.where(within_per_bit > 1e-6, within_per_bit, np.nan)
    ratio_mean = float(np.nanmean(ratio_per_bit[valid])) if n_valid else float("nan")

    # Bootstrap CI on ratio_mean: resample test-set ROWS, recompute per-bit AUROCs, mean ratio.
    rng = np.random.default_rng(boot_seed)
    boot_ratios = np.empty(n_boot, dtype=np.float64)
    for bi in range(n_boot):
        idx = rng.integers(0, n_test, size=n_test)
        w = _auroc_per_bit(proba_within, y_test_bits, skipped, idx)
        t = _auroc_per_bit(proba_transfer, y_test_bits, skipped, idx)
        v = ~np.isnan(w) & ~np.isnan(t)
        if not v.any():
            boot_ratios[bi] = np.nan
            continue
        r = t[v] / np.where(w[v] > 1e-6, w[v], np.nan)
        boot_ratios[bi] = float(np.nanmean(r))
    ci_lo = float(np.nanpercentile(boot_ratios, 2.5))
    ci_hi = float(np.nanpercentile(boot_ratios, 97.5))

    per_bit_records = []
    for b in range(y_test_bits.shape[1]):
        rec = {
            "bit": b,
            "skipped": bool(skipped[b]) or not bool(valid[b] if b < len(valid) else False),
            "within_auroc": (None if np.isnan(within_per_bit[b]) else float(within_per_bit[b])),
            "transfer_auroc": (None if np.isnan(transfer_per_bit[b]) else float(transfer_per_bit[b])),
            "ratio": (None if np.isnan(ratio_per_bit[b]) else float(ratio_per_bit[b])),
        }
        per_bit_records.append(rec)

    return {
        "within_auroc_mean": within_mean,
        "transfer_auroc_mean": transfer_mean,
        "transfer_ratio_mean": ratio_mean,
        "transfer_ratio_ci_95": [ci_lo, ci_hi],
        "n_valid_bits": n_valid,
        "skipped_bits": [int(b) for b in np.where(skipped)[0]],
        "per_bit": per_bit_records,
    }


def _diagnosis_and_top3(ratio_AtoB: float, ratio_BtoA: float) -> tuple[str, list, bool]:
    asymmetric = abs(ratio_AtoB - ratio_BtoA) > 0.20
    rmean = float(np.mean([ratio_AtoB, ratio_BtoA]))
    if rmean >= 0.95:
        return "basis-ambiguity", ["A", "AC", "G"], asymmetric
    if rmean >= 0.50:
        return "partial-linear-compatibility", ["F", "G", "AC"], asymmetric
    return "linearly-incompatible", ["F", "C", "AC"], asymmetric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build (zA, zB) pair via D1's helper -- byte-equivalent substrate across diagnostics.
    import run as d1_run  # type: ignore[import-not-found]  # D1/run.py on sys.path

    t0 = time.perf_counter()
    z_a, z_b, meta = d1_run.build_baseline_pair(cfg)
    n = z_a.shape[0]

    if n < 100:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": f"n_eval_samples={n} < 100; probe-transfer estimate is uninformative.",
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # Recover 11-bit basis labels for the held-out eval samples.
    y_bits = _eval_labels_11bit(cfg)
    assert y_bits.shape[0] == n, f"label/feature size mismatch: {y_bits.shape[0]} vs {n}"
    n_bits = int(y_bits.shape[1])

    # Train/test split (same indices used for both directions for fair comparison).
    test_frac = float(cfg.get("test_fraction", 0.30))
    split_rng = np.random.default_rng(int(cfg.get("split_seed", 271828)))
    perm = split_rng.permutation(n)
    n_test = max(1, int(round(n * test_frac)))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]
    z_a_tr, z_a_te = z_a[train_idx], z_a[test_idx]
    z_b_tr, z_b_te = z_b[train_idx], z_b[test_idx]
    y_tr, y_te = y_bits[train_idx], y_bits[test_idx]

    print(
        f"[D3] computing probe-transfer on n={n} (train={len(train_idx)}, test={n_test}); "
        f"dimA={meta['dim_A']}, dimB={meta['dim_B']}, n_bits={n_bits}",
        flush=True,
    )

    C_canon = float(cfg.get("probe_C_canonical", 1.0))
    max_iter = int(cfg.get("probe_max_iter", 1000))
    n_boot = int(cfg.get("n_bootstrap", 1000))
    boot_seed = int(cfg.get("bootstrap_seed", 31415))

    # Canonical-C run with full bootstrap CI for both directions.
    A_to_B = _direction(
        z_a_tr, z_b_tr, z_a_te, z_b_te, y_tr, y_te,
        C=C_canon, max_iter=max_iter, n_boot=n_boot, boot_seed=boot_seed,
    )
    B_to_A = _direction(
        z_b_tr, z_a_tr, z_b_te, z_a_te, y_tr, y_te,
        C=C_canon, max_iter=max_iter, n_boot=n_boot, boot_seed=boot_seed + 1,
    )

    # Honest-negative gate: within-pathway AUROC < 0.6 in EITHER direction means the
    # baseline didn't learn the task, so the diagnostic is uninformative.
    within_min = float(min(A_to_B["within_auroc_mean"], B_to_A["within_auroc_mean"]))
    if within_min < 0.6:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": (
                f"within-pathway AUROC < 0.6 (min={within_min:.3f}); §5.7 baseline did not "
                "learn the basis-mask task at this lambda_C=0.0 condition. Re-run baseline first."
            ),
            **meta,
            "n_test_samples": n_test,
            "n_train_probe_samples": int(len(train_idx)),
            "C_used": C_canon,
            "A_to_B": A_to_B,
            "B_to_A": B_to_A,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    diagnosis, top_3, asymmetric = _diagnosis_and_top3(
        A_to_B["transfer_ratio_mean"], B_to_A["transfer_ratio_mean"]
    )

    # C-sweep robustness (no bootstrap CI; just mean ratios per C, both directions).
    C_sweep_robustness: dict = {}
    for C in cfg.get("probe_C_sweep", [0.01, 0.1, 1.0, 10.0]):
        Cf = float(C)
        if abs(Cf - C_canon) < 1e-12:
            C_sweep_robustness[str(Cf)] = {
                "A_to_B_ratio": A_to_B["transfer_ratio_mean"],
                "B_to_A_ratio": B_to_A["transfer_ratio_mean"],
                "is_canonical": True,
            }
            continue
        ab = _direction(
            z_a_tr, z_b_tr, z_a_te, z_b_te, y_tr, y_te,
            C=Cf, max_iter=max_iter, n_boot=0, boot_seed=boot_seed,
        )
        ba = _direction(
            z_b_tr, z_a_tr, z_b_te, z_a_te, y_tr, y_te,
            C=Cf, max_iter=max_iter, n_boot=0, boot_seed=boot_seed + 1,
        )
        C_sweep_robustness[str(Cf)] = {
            "A_to_B_ratio": ab["transfer_ratio_mean"],
            "B_to_A_ratio": ba["transfer_ratio_mean"],
            "is_canonical": False,
        }

    result = {
        "verdict": "diagnostic_complete",
        **meta,
        "n_test_samples": n_test,
        "n_train_probe_samples": int(len(train_idx)),
        "n_bits": n_bits,
        "C_used": C_canon,
        "C_sweep_robustness": C_sweep_robustness,
        "n_bootstrap": n_boot,
        "A_to_B": A_to_B,
        "B_to_A": B_to_A,
        "asymmetric": asymmetric,
        "diagnosis": diagnosis,
        "top_3_implied": top_3,
        "wall_s": round(time.perf_counter() - t0, 1),
        "git_head_at_run": _git_sha(HERE),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
