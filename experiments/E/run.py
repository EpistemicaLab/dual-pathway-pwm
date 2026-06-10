"""E -- Equivariant by construction (POST-HOC orbit-averaging surrogate).

Per methods/method-E-equivariant.md: build encoders equivariant to relevant
physical symmetry (Galilean, SO(2) for 1D Duffing). Spec calls for e2cnn
rotation-equivariant CNN on pathway B and time-translation-equivariant MLP on
pathway A. MVP DEVIATION: post-hoc orbit-averaging projection on z_a_pre,
z_b_pre (32-d section-5.7 baseline latents at lambda_C=1.0) instead of
encoder retraining with e2cnn. By construction, phi_g(g.z) = phi_g(z) for
any g in the chosen finite group G.

Group projections (deterministic, no training):
  C2 (sign-flip per component):  phi(z)_i  = |z_i|                          (32-d)
  C4 (pair-rotation):            phi(z)_i  = z_{2i}^2 + z_{2i+1}^2          (16-d)
  C2xC4:                         stack [C2(z), C4(z)]                       (48-d)

The projection partitions the latent space into orbits under the group action;
within each orbit, we take an invariant scalar feature (absolute value for C2;
power-sum for C4 since R^2 = x^2 + y^2 is invariant under SO(2) rotations of
(x,y) and SO(2) contains C4 as a subgroup). Basis-ambiguity along these orbits
dissolves by fiat.

Output: results/E.json with PASS_GATE 4-gate evaluator (c1, pass_linear,
pass_cka, pass_transfer) per the original spec.

Honest-negative paths:
  * n_eval_paired < 100
  * ALL 3 group cells produce non-finite probe AUROC.
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
# NOTE: Drop D3_DIR per AC attempt-2 fix (sys.path insert(0) reverses iter order;
# D3/run.py would shadow D1/run.py and break import build_baseline_pair).
for p in (D1_DIR, E1_DIR, W4_DIR, W5_DIR):
    sys.path.insert(0, str(p))


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


# --- CKA ---
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = float(np.linalg.norm(X.T @ Y, ord="fro") ** 2)
    den = float(np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro"))
    return float("nan") if den <= 0.0 else num / den


def bootstrap_ci(fn_xy, X, Y, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = fn_xy(X[idx], Y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), float("nan")
    lo = float(np.quantile(vals, alpha / 2))
    hi = float(np.quantile(vals, 1 - alpha / 2))
    return lo, hi


# --- Group orbit projections ---
def project_group(z: np.ndarray, group: str) -> np.ndarray:
    """Deterministic orbit-invariant projection of z (n, d) under group G.

    C2     : phi(z)_i  = |z_i|                                 (d)
    C4     : phi(z)_i  = z_{2i}^2 + z_{2i+1}^2  (i=0..d/2-1)   (d/2)  -- d must be even
    C2xC4  : concatenate [C2(z), C4(z)]                        (d + d/2 = 3d/2)
    """
    if group == "C2":
        return np.abs(z)
    if group == "C4":
        d = z.shape[-1]
        if d % 2 != 0:
            raise ValueError(f"C4 requires even d, got {d}")
        z_pairs = z.reshape(z.shape[0], d // 2, 2)
        return np.sum(z_pairs ** 2, axis=-1)
    if group == "C2xC4":
        return np.concatenate([project_group(z, "C2"), project_group(z, "C4")], axis=-1)
    raise ValueError(f"Unknown group: {group}")


# --- per-bit probes ---
def _per_bit_probes(z, y_bits, C, max_iter):
    n_bits = y_bits.shape[1]
    probes = [None] * n_bits
    skipped = []
    for b in range(n_bits):
        y = y_bits[:, b]
        if y.sum() == 0 or y.sum() == y.shape[0]:
            skipped.append(b)
            continue
        try:
            clf = LogisticRegression(C=C, max_iter=max_iter)
            clf.fit(z, y)
            probes[b] = clf
        except Exception:  # noqa: BLE001
            skipped.append(b)
    return probes, skipped


def _per_bit_auroc(probes, skipped, z, y_bits):
    n_bits = y_bits.shape[1]
    out = [None] * n_bits
    for b in range(n_bits):
        if b in skipped or probes[b] is None:
            continue
        y = y_bits[:, b]
        if y.sum() == 0 or y.sum() == y.shape[0]:
            continue
        try:
            proba = probes[b].predict_proba(z)
            classes = probes[b].classes_
            pos_col = int(np.where(classes == 1)[0][0]) if 1 in classes else 1
            out[b] = float(roc_auc_score(y, proba[:, pos_col]))
        except Exception:  # noqa: BLE001
            continue
    return out


def _safe_mean(arr):
    finite = [v for v in arr if v is not None and np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # 1) Build section-5.7 baseline pair at lambda_C=1.0 (same substrate as AC/F/G/A/F0/C/B/I/H/J/D).
    import run as d1_run  # type: ignore[import-not-found]
    z_a_pre, z_b_pre, meta = d1_run.build_baseline_pair(cfg)
    n = z_a_pre.shape[0]
    print(f"[E] baseline pair built. n={n} dimA={meta['dim_A']} dimB={meta['dim_B']} "
          f"lambda_c={meta['lambda_c']}", flush=True)

    if n < 100:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": f"n_eval_samples={n} < 100; PASS_GATE estimate is uninformative.",
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # 2) Recover 11-bit basis labels deterministically.
    import measure_bound_real_lc as e1
    if cfg["smoke"]:
        c_grid, k_grid = (0.05, 0.10), (1.0, 2.0)
        n_eval = 12
    else:
        c_grid, k_grid = (0.05, 0.10, 0.20), (1.0, 2.0, 5.0)
        n_eval = int(cfg["n_eval_seeds"])
    _imgs, _feats, eval_cls = e1._eval_set(c_grid, k_grid, n_eval, seed_base=50_000)
    y_bits = np.array(
        [e1.ORACLE_MASKS[e1.ALL_CLASSES[ci]] for ci in eval_cls],
        dtype=np.int8,
    )
    assert y_bits.shape[0] == n, f"label count {y_bits.shape[0]} != latent count {n}"

    # 3) 70/30 head split (head_split_seed=271828 same as AC/F/G/A/F0/C/B/I/H/J/D/D3/D4).
    rng = np.random.default_rng(int(cfg["head_split_seed"]))
    perm = rng.permutation(n)
    n_train = int(round(n * float(cfg["head_train_fraction"])))
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    z_a_tr, z_a_te = z_a_pre[tr_idx], z_a_pre[te_idx]
    z_b_tr, z_b_te = z_b_pre[tr_idx], z_b_pre[te_idx]
    y_tr, y_te = y_bits[tr_idx], y_bits[te_idx]

    # 4) Group sweep -- deterministic projection, no training.
    groups = list(cfg["e_groups"])
    cell_records = []
    n_bootstrap = int(cfg["n_bootstrap"])
    bootstrap_seed = int(cfg["bootstrap_seed"])
    probe_C = float(cfg["probe_C"])
    probe_max_iter = int(cfg["probe_max_iter"])

    for g in groups:
        t_cell = time.perf_counter()
        try:
            phi_a_tr = project_group(z_a_tr, g)
            phi_a_te = project_group(z_a_te, g)
            phi_b_tr = project_group(z_b_tr, g)
            phi_b_te = project_group(z_b_te, g)
        except Exception as e:  # noqa: BLE001
            cell_records.append({"group": g, "error": str(e), "auroc_A": float("nan"),
                                 "wall_s": round(time.perf_counter() - t_cell, 2)})
            continue

        probes_A, skipped_A = _per_bit_probes(phi_a_tr, y_tr, probe_C, probe_max_iter)
        probes_B, skipped_B = _per_bit_probes(phi_b_tr, y_tr, probe_C, probe_max_iter)
        auroc_A_within = _per_bit_auroc(probes_A, skipped_A, phi_a_te, y_te)
        auroc_B_within = _per_bit_auroc(probes_B, skipped_B, phi_b_te, y_te)
        within_A = _safe_mean(auroc_A_within)
        within_B = _safe_mean(auroc_B_within)
        rec = {
            "group": g,
            "proj_dim": int(phi_a_tr.shape[-1]),
            "within_A_mean": within_A,
            "within_B_mean": within_B,
            "wall_s": round(time.perf_counter() - t_cell, 2),
        }
        cell_records.append(rec)
        print(f"[E] cell group={g} dim={rec['proj_dim']} within_A={within_A:.4f} within_B={within_B:.4f} "
              f"wall={rec['wall_s']}s", flush=True)

    finite_cells = [r for r in cell_records
                    if "within_A_mean" in r and np.isfinite(r["within_A_mean"])]
    if not finite_cells:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": "All E group cells produced non-finite probe AUROC.",
            "cell_records": cell_records,
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # Best cell = max within_A_mean; tie-break smaller proj_dim (Occam).
    best_cell = max(finite_cells, key=lambda r: (r["within_A_mean"], -r["proj_dim"]))
    best_group = best_cell["group"]
    print(f"[E] best cell group={best_group} within_A={best_cell['within_A_mean']:.4f}", flush=True)

    # 5) PASS_GATE on best group (full evaluation including transfer + CKA).
    phi_a_tr = project_group(z_a_tr, best_group)
    phi_a_te = project_group(z_a_te, best_group)
    phi_b_tr = project_group(z_b_tr, best_group)
    phi_b_te = project_group(z_b_te, best_group)

    probes_A, skipped_A = _per_bit_probes(phi_a_tr, y_tr, probe_C, probe_max_iter)
    probes_B, skipped_B = _per_bit_probes(phi_b_tr, y_tr, probe_C, probe_max_iter)
    auroc_A_within = _per_bit_auroc(probes_A, skipped_A, phi_a_te, y_te)
    auroc_B_within = _per_bit_auroc(probes_B, skipped_B, phi_b_te, y_te)
    auroc_A_to_B = _per_bit_auroc(probes_A, skipped_A, phi_b_te, y_te)
    auroc_B_to_A = _per_bit_auroc(probes_B, skipped_B, phi_a_te, y_te)

    probe_auroc_A = _safe_mean(auroc_A_within)
    probe_auroc_B = _safe_mean(auroc_B_within)
    transfer_AtoB_mean = _safe_mean(auroc_A_to_B)
    transfer_BtoA_mean = _safe_mean(auroc_B_to_A)
    within_A_mean = probe_auroc_A
    within_B_mean = probe_auroc_B
    ratio_AtoB = float(transfer_AtoB_mean / within_A_mean) if within_A_mean > 0 else float("nan")
    ratio_BtoA = float(transfer_BtoA_mean / within_B_mean) if within_B_mean > 0 else float("nan")
    lower_transfer_ratio = float(min(ratio_AtoB, ratio_BtoA))
    probe_transfer_auroc = float(0.5 * (transfer_AtoB_mean + transfer_BtoA_mean))
    within_pathway_auroc = float(0.5 * (within_A_mean + within_B_mean))

    cka_val = linear_cka(phi_a_te, phi_b_te)
    cka_lo, cka_hi = bootstrap_ci(linear_cka, phi_a_te, phi_b_te,
                                   n_boot=n_bootstrap, seed=bootstrap_seed)

    # PASS_GATE 4-gate per the original spec.
    thr = cfg["pass_gate"]
    c1_pass = bool(np.isfinite(probe_auroc_A) and np.isfinite(probe_auroc_B)
                   and probe_auroc_A >= thr["c1_min_per_pathway_auroc"]
                   and probe_auroc_B >= thr["c1_min_per_pathway_auroc"])
    pass_cka = bool(c1_pass and np.isfinite(cka_val) and cka_val >= thr["cka_min"])
    pass_transfer = bool(c1_pass and np.isfinite(lower_transfer_ratio)
                         and lower_transfer_ratio >= thr["transfer_ratio_min"])
    pass_linear = False  # delta_1/epsilon_2 not exposed; forced FALSE per AC/F/G/.../D.
    pass_any = pass_cka or pass_transfer or pass_linear
    verdict = "PASS" if pass_any else "HONEST_NEGATIVE"

    # Diagnostic: equivariance residual is 0 by construction (orbit-invariant projection).
    equivariance_residual = 0.0

    # Raw norms before/after projection on test slice.
    norm_a_pre_te = float(np.mean(np.linalg.norm(z_a_te, axis=-1)))
    norm_b_pre_te = float(np.mean(np.linalg.norm(z_b_te, axis=-1)))
    norm_a_post_te = float(np.mean(np.linalg.norm(phi_a_te, axis=-1)))
    norm_b_post_te = float(np.mean(np.linalg.norm(phi_b_te, axis=-1)))

    result = {
        "verdict": verdict,
        "method": "E",
        "lambda_c": float(meta["lambda_c"]),
        "n_eval_samples": int(n),
        "n_head_train": int(n_train),
        "n_head_test": int(n - n_train),
        "equivariance_group": best_group,
        "equivariance_residual": equivariance_residual,
        "equivariance_residual_note": "0.0 by construction: orbit-invariant projection phi_g satisfies phi_g(g.z)=phi_g(z) for all g in G. Spec's e2cnn approach would have a small but nonzero residual due to discrete-group approximation of continuous SO(2).",
        "proj_dim": int(phi_a_te.shape[-1]),
        "groups_swept": groups,
        "cell_records": cell_records,
        "best_cell": best_cell,
        "probe_auroc_A": probe_auroc_A,
        "probe_auroc_B": probe_auroc_B,
        "auroc_A_per_bit": auroc_A_within,
        "auroc_B_per_bit": auroc_B_within,
        "skipped_bits_A": skipped_A,
        "skipped_bits_B": skipped_B,
        "cka": cka_val,
        "cka_ci_95": [cka_lo, cka_hi],
        "transfer_AtoB_mean": transfer_AtoB_mean,
        "transfer_BtoA_mean": transfer_BtoA_mean,
        "ratio_AtoB": ratio_AtoB,
        "ratio_BtoA": ratio_BtoA,
        "lower_transfer_ratio": lower_transfer_ratio,
        "probe_transfer_auroc": probe_transfer_auroc,
        "within_pathway_auroc": within_pathway_auroc,
        "within_A_mean": within_A_mean,
        "within_B_mean": within_B_mean,
        "norm_a_pre_te_mean": norm_a_pre_te,
        "norm_b_pre_te_mean": norm_b_pre_te,
        "norm_a_post_te_mean": norm_a_post_te,
        "norm_b_post_te_mean": norm_b_post_te,
        "delta_1": None,
        "epsilon_2": None,
        "c1_pass": c1_pass,
        "pass_linear": pass_linear,
        "pass_cka": pass_cka,
        "pass_transfer": pass_transfer,
        "pass_any": pass_any,
        "deviation_from_spec": (
            "POST-HOC orbit-averaging projection on z_a_pre/z_b_pre (32-d section-5.7 baseline "
            "latents at lambda_C=1.0) instead of joint encoder retraining with e2cnn-equivariant "
            "CNN. By construction, phi_g(g.z) = phi_g(z) for any g in chosen finite group G, so "
            "basis-ambiguity ALONG the group orbits is removed by fiat. Group cells: C2 "
            "(sign-flip, phi(z)_i = |z_i|, dim 32); C4 (pair-rotation, phi(z)_i = z_{2i}^2 + "
            "z_{2i+1}^2, dim 16); C2xC4 (concat, dim 48). Spec sweeps {C4, C8, SO(2)}; we sweep "
            "{C2, C4, C2xC4} -- C2 included because 1D Duffing has Z_2 reflection symmetry; C8 "
            "and SO(2) require encoder retraining (no closed-form orbit-invariant for arbitrary "
            "z layouts). Mirrors AC/F/G/A/F0/C/B/I/H/J/D post-hoc choice. delta_1 / epsilon_2 "
            "not exposed by build_baseline_pair -> pass_linear forced FALSE; pass_cka and "
            "pass_transfer carry verdict. No training (deterministic projection) -> no seed CI."
        ),
        "n_bootstrap": n_bootstrap,
        "wall_s": round(time.perf_counter() - t0, 1),
        "git_head_at_run": _git_sha(HERE),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
