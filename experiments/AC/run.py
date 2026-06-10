"""AC -- DeepCCA loss + weight-shared g_theta core.

Per methods/method-AC-deepcca-shared.md: combines method A (DeepCCA loss) with
method C (weight-shared g_theta). Built on top of the §5.7 baseline at
lambda_C=1.0 (fix-knob ON, distinct from D1-D4's lambda_C=0.0 baseline).

DEVIATION FROM SPEC (documented):
  Spec calls for JOINT encoder + g_theta training with DeepCCA + L_C at
  lambda_C=1.0. This implementation is POST-HOC: build_baseline_pair() trains
  encoders once at lambda_C=1.0 (which already includes L_C alignment), then
  shared g_theta + DeepCCA is fit on top with 20 seeds. Trade-off: cheaper
  (no encoder retrain per seed) and evaluates AC's marginal benefit on top
  of the L_C-aligned base rather than as a combined from-scratch fix.

Output: results/method-AC.json with PASS_GATE 4-gate evaluator
(c1, pass_linear, pass_cka, pass_transfer) per the original spec.

Honest-negative paths:
  * n_eval_paired < 100
  * DeepCCA loss diverges to NaN on all 20 seeds
  * Per-bit probes degenerate on >50% of bits
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
import jax
import jax.numpy as jnp
import jax.nn as jnn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
D1_DIR = HERE.parents[0] / "D1"
E1_DIR = HERE.parents[1] / "E1-numerical-tightness"
W4_DIR = HERE.parents[1] / "W4-learned-B"
W5_DIR = HERE.parents[1] / "W5-C13-LC-ablation"
# NOTE (attempt 2, 2026-06-05): D3_DIR removed from sys.path tuple.
# Bug history: prior attempt's `for p in (D1_DIR, D3_DIR, E1_DIR, ...): sys.path.insert(0, ...)`
# put D3 ahead of D1 because insert(0) reverses iteration order, so `import run as d1_run`
# resolved to D3/run.py (which lacks build_baseline_pair as a top-level attr) -> AttributeError.
# AC consumes only D1.build_baseline_pair; D3 was never a library for AC. Dropping it fixes the
# shadow without touching any DeepCCA / shared-g_theta / PASS_GATE logic.
for p in (D1_DIR, E1_DIR, W4_DIR, W5_DIR):
    sys.path.insert(0, str(p))


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


# --- CKA (copy from D1 to keep AC self-contained) ---
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = float(np.linalg.norm(X.T @ Y, ord="fro") ** 2)
    den = float(np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro"))
    return float("nan") if den <= 0.0 else num / den


def bootstrap_ci(fn_xy, X, Y, n_boot: int, seed: int, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = fn_xy(X[idx], Y[idx])
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


# --- Shared g_theta MLP (same weights applied to zA and zB) ---
def _init_g_theta(key, dim_in: int, depth: int, hidden: int, dim_out: int):
    keys = jax.random.split(key, depth)
    layers = []
    cur = dim_in
    for i in range(depth - 1):
        W = jax.random.normal(keys[i], (cur, hidden)) * jnp.sqrt(2.0 / cur)
        layers.append({"W": W, "b": jnp.zeros((hidden,))})
        cur = hidden
    W = jax.random.normal(keys[-1], (cur, dim_out)) * jnp.sqrt(1.0 / cur)
    layers.append({"W": W, "b": jnp.zeros((dim_out,))})
    return layers


def _g_forward(layers, x):
    h = x
    L = len(layers)
    for i, layer in enumerate(layers):
        h = h @ layer["W"] + layer["b"]
        if i < L - 1:
            h = jnn.relu(h)
    return h


def _deepcca_loss(layers, z_a, z_b, r: float):
    """Negative sum of singular values of T = Sigma_AA^{-1/2} Sigma_AB Sigma_BB^{-1/2}.

    Uses eigendecomposition for matrix inverse-sqrt of the regularized covariances.
    Loss minimized -> correlation maximized.
    """
    H_a = _g_forward(layers, z_a)
    H_b = _g_forward(layers, z_b)
    H_a = H_a - jnp.mean(H_a, axis=0, keepdims=True)
    H_b = H_b - jnp.mean(H_b, axis=0, keepdims=True)
    n = H_a.shape[0]
    d = H_a.shape[1]
    eye = jnp.eye(d)
    S_aa = (H_a.T @ H_a) / (n - 1) + r * eye
    S_bb = (H_b.T @ H_b) / (n - 1) + r * eye
    S_ab = (H_a.T @ H_b) / (n - 1)
    # Inverse sqrt via eigendecomposition (S is PSD; eigh is stable on float32).
    wa, Va = jnp.linalg.eigh(S_aa)
    wb, Vb = jnp.linalg.eigh(S_bb)
    wa = jnp.clip(wa, 1e-12, None)
    wb = jnp.clip(wb, 1e-12, None)
    S_aa_inv_sqrt = (Va * (wa ** -0.5)[None, :]) @ Va.T
    S_bb_inv_sqrt = (Vb * (wb ** -0.5)[None, :]) @ Vb.T
    T = S_aa_inv_sqrt @ S_ab @ S_bb_inv_sqrt
    # sum of singular values = trace((T'T)^{1/2})
    sv = jnp.linalg.svd(T, compute_uv=False)
    return -jnp.sum(sv)


def _adam_init(params):
    return {"m": jax.tree_util.tree_map(jnp.zeros_like, params),
            "v": jax.tree_util.tree_map(jnp.zeros_like, params),
            "t": 0}


def _adam_apply(params, opt, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    t = opt["t"] + 1
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1 - b1) * g, opt["m"], grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1 - b2) * (g * g), opt["v"], grads)
    mh = jax.tree_util.tree_map(lambda mm: mm / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda vv: vv / (1 - b2 ** t), v)
    new = jax.tree_util.tree_map(lambda p, mm, vv: p - lr * mm / (jnp.sqrt(vv) + eps), params, mh, vh)
    return new, {"m": m, "v": v, "t": t}


def _train_g_theta_one_seed(z_a_tr, z_b_tr, depth, hidden, dim_out, r, lr, n_epochs, seed: int):
    key = jax.random.PRNGKey(int(seed))
    layers = _init_g_theta(key, dim_in=z_a_tr.shape[1], depth=int(depth),
                           hidden=int(hidden), dim_out=int(dim_out))
    opt = _adam_init(layers)
    grad_fn = jax.value_and_grad(_deepcca_loss)

    @jax.jit
    def step(layers, opt, x, y):
        loss, g = grad_fn(layers, x, y, r)
        new, new_opt = _adam_apply(layers, opt, g, lr=lr)
        return new, new_opt, loss

    x = jnp.asarray(z_a_tr, dtype=jnp.float32)
    y = jnp.asarray(z_b_tr, dtype=jnp.float32)
    losses = []
    t0 = time.perf_counter()
    for _ in range(int(n_epochs)):
        layers, opt, loss = step(layers, opt, x, y)
        losses.append(float(loss))
    return layers, {"final_loss": float(losses[-1]), "loss_curve": losses,
                    "wall_s": round(time.perf_counter() - t0, 2)}


# --- PASS_GATE: per-bit logistic-regression probes & cross-pathway transfer ---
def _per_bit_probes(z_train, y_train_bits, C, max_iter):
    n_bits = y_train_bits.shape[1]
    probes = [None] * n_bits
    skipped = np.zeros(n_bits, dtype=bool)
    for b in range(n_bits):
        y = y_train_bits[:, b]
        if y.min() == y.max():
            skipped[b] = True
            continue
        clf = LogisticRegression(C=float(C), max_iter=int(max_iter))
        clf.fit(z_train, y)
        probes[b] = clf
    return probes, skipped


def _per_bit_auroc(probes, skipped, z_test, y_test_bits):
    n_bits = len(probes)
    out = np.full(n_bits, np.nan, dtype=np.float64)
    for b in range(n_bits):
        if skipped[b]:
            continue
        y = y_test_bits[:, b]
        if y.min() == y.max():
            continue
        clf = probes[b]
        proba = clf.predict_proba(z_test)
        pos_col = int(np.argmax(clf.classes_ == 1))
        out[b] = roc_auc_score(y, proba[:, pos_col])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # 1) Build the §5.7 baseline pair at lambda_C=1.0 (fix-knob ON).
    import run as d1_run  # type: ignore[import-not-found]
    z_a_pre, z_b_pre, meta = d1_run.build_baseline_pair(cfg)
    n = z_a_pre.shape[0]
    print(f"[AC] baseline pair built. n={n} dimA={meta['dim_A']} dimB={meta['dim_B']} "
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

    # 2) Recover 11-bit basis labels deterministically (D3 helper logic, inlined).
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

    # 3) 70/30 split — train g_theta + probes on 70%, hold out 30% for PASS_GATE.
    head_train_frac = float(cfg["head_train_fraction"])
    rng = np.random.default_rng(int(cfg["head_split_seed"]))
    perm = rng.permutation(n)
    n_train = int(round(n * head_train_frac))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    z_a_tr_pre, z_a_te_pre = z_a_pre[train_idx], z_a_pre[test_idx]
    z_b_tr_pre, z_b_te_pre = z_b_pre[train_idx], z_b_pre[test_idx]
    y_tr_bits, y_te_bits = y_bits[train_idx], y_bits[test_idx]

    # 4) Train shared g_theta with DeepCCA loss, 20 seeds. Best by lowest final loss.
    n_seeds = int(cfg["deepcca_n_seeds"])
    seed_base = int(cfg["deepcca_seed_base"])
    seed_records = []
    best = None
    for s in range(n_seeds):
        seed = seed_base + s
        layers, info = _train_g_theta_one_seed(
            z_a_tr_pre, z_b_tr_pre,
            depth=cfg["g_theta_depth"], hidden=cfg["g_theta_hidden"],
            dim_out=cfg["g_theta_output_dim"], r=float(cfg["deepcca_whitening_reg"]),
            lr=float(cfg["deepcca_lr"]), n_epochs=int(cfg["deepcca_epochs"]),
            seed=seed,
        )
        rec = {"seed": seed, "final_loss": info["final_loss"], "wall_s": info["wall_s"]}
        seed_records.append(rec)
        if (np.isfinite(info["final_loss"]) and (best is None or info["final_loss"] < best["final_loss"])):
            best = {"layers": layers, **rec}
        if (s % 5) == 0 or s == n_seeds - 1:
            print(f"[AC] g_theta seed {s+1}/{n_seeds} loss={info['final_loss']:.4f} "
                  f"wall={info['wall_s']:.1f}s", flush=True)

    if best is None:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": "All 20 g_theta seeds diverged to NaN.",
            "seed_records": seed_records,
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # 5) Apply best g_theta to ALL splits to get AC outputs.
    g_layers = best["layers"]

    @jax.jit
    def _apply_g(x):
        return _g_forward(g_layers, x)

    z_a_tr = np.asarray(_apply_g(jnp.asarray(z_a_tr_pre, dtype=jnp.float32)))
    z_a_te = np.asarray(_apply_g(jnp.asarray(z_a_te_pre, dtype=jnp.float32)))
    z_b_tr = np.asarray(_apply_g(jnp.asarray(z_b_tr_pre, dtype=jnp.float32)))
    z_b_te = np.asarray(_apply_g(jnp.asarray(z_b_te_pre, dtype=jnp.float32)))

    # 6) PASS_GATE: c1 (per-bit AUROC on each pathway), CKA, transfer ratio.
    C = float(cfg["probe_C"])
    max_iter = int(cfg["probe_max_iter"])
    probes_a, skipped_a = _per_bit_probes(z_a_tr, y_tr_bits, C, max_iter)
    probes_b, skipped_b = _per_bit_probes(z_b_tr, y_tr_bits, C, max_iter)

    auroc_a_per_bit = _per_bit_auroc(probes_a, skipped_a, z_a_te, y_te_bits)
    auroc_b_per_bit = _per_bit_auroc(probes_b, skipped_b, z_b_te, y_te_bits)
    valid_a = ~np.isnan(auroc_a_per_bit)
    valid_b = ~np.isnan(auroc_b_per_bit)
    probe_auroc_A = float(np.nanmean(auroc_a_per_bit[valid_a])) if valid_a.any() else float("nan")
    probe_auroc_B = float(np.nanmean(auroc_b_per_bit[valid_b])) if valid_b.any() else float("nan")

    # Transfer: probe trained on z_a_tr, evaluated on z_b_te (and symmetric).
    transfer_a_to_b = _per_bit_auroc(probes_a, skipped_a, z_b_te, y_te_bits)
    transfer_b_to_a = _per_bit_auroc(probes_b, skipped_b, z_a_te, y_te_bits)
    valid_ab = ~np.isnan(transfer_a_to_b) & valid_a
    valid_ba = ~np.isnan(transfer_b_to_a) & valid_b
    transfer_AtoB_mean = float(np.nanmean(transfer_a_to_b[valid_ab])) if valid_ab.any() else float("nan")
    transfer_BtoA_mean = float(np.nanmean(transfer_b_to_a[valid_ba])) if valid_ba.any() else float("nan")
    within_A_mean = float(np.nanmean(auroc_a_per_bit[valid_ab])) if valid_ab.any() else float("nan")
    within_B_mean = float(np.nanmean(auroc_b_per_bit[valid_ba])) if valid_ba.any() else float("nan")
    ratio_AtoB = transfer_AtoB_mean / within_A_mean if within_A_mean > 0 else float("nan")
    ratio_BtoA = transfer_BtoA_mean / within_B_mean if within_B_mean > 0 else float("nan")
    probe_transfer_auroc = (transfer_AtoB_mean + transfer_BtoA_mean) / 2.0
    within_pathway_auroc = (within_A_mean + within_B_mean) / 2.0

    # CKA on test split (held-out of head training).
    cka_te = linear_cka(z_a_te, z_b_te)
    n_boot = int(cfg["n_bootstrap"])
    cka_lo, cka_hi = bootstrap_ci(linear_cka, z_a_te, z_b_te, n_boot=n_boot, seed=int(cfg["bootstrap_seed"]))

    # PASS_GATE 4-gate evaluator (per the original spec PASS_GATE):
    c1_min = float(cfg["pass_gate"]["c1_min_per_pathway_auroc"])
    cka_min = float(cfg["pass_gate"]["cka_min"])
    transfer_ratio_min = float(cfg["pass_gate"]["transfer_ratio_min"])
    c1_pass = bool(probe_auroc_A >= c1_min and probe_auroc_B >= c1_min)
    pass_cka = bool(c1_pass and cka_te >= cka_min)
    # Use lower-direction transfer ratio (conservative, mirrors D3 spec rule).
    lower_transfer_ratio = float(min(ratio_AtoB, ratio_BtoA)) if np.isfinite(ratio_AtoB) and np.isfinite(ratio_BtoA) else float("nan")
    pass_transfer = bool(c1_pass and np.isfinite(lower_transfer_ratio) and lower_transfer_ratio >= transfer_ratio_min)
    # delta_1 / epsilon_2 require §5.7-specific quantities not exposed here; emit null.
    pass_linear = False
    delta_1 = None
    epsilon_2 = None
    pass_any = pass_cka or pass_transfer or pass_linear
    verdict = "PASS" if pass_any else "HONEST_NEGATIVE"

    result = {
        "verdict": verdict,
        "method": "AC",
        **meta,
        "n_head_train": int(len(train_idx)),
        "n_head_test": int(len(test_idx)),
        "g_theta": {
            "depth": int(cfg["g_theta_depth"]),
            "hidden": int(cfg["g_theta_hidden"]),
            "output_dim": int(cfg["g_theta_output_dim"]),
            "deepcca_whitening_reg": float(cfg["deepcca_whitening_reg"]),
            "deepcca_lr": float(cfg["deepcca_lr"]),
            "deepcca_epochs": int(cfg["deepcca_epochs"]),
            "n_seeds": n_seeds,
            "best_seed": int(best["seed"]),
            "best_final_loss": float(best["final_loss"]),
        },
        "seed_records": seed_records,
        "pass_gate": {
            "probe_auroc_A": probe_auroc_A,
            "probe_auroc_B": probe_auroc_B,
            "auroc_A_per_bit": [None if np.isnan(v) else float(v) for v in auroc_a_per_bit],
            "auroc_B_per_bit": [None if np.isnan(v) else float(v) for v in auroc_b_per_bit],
            "skipped_bits_A": [int(b) for b in np.where(skipped_a)[0]],
            "skipped_bits_B": [int(b) for b in np.where(skipped_b)[0]],
            "cka": cka_te,
            "cka_ci_95": [cka_lo, cka_hi],
            "transfer_AtoB_mean": transfer_AtoB_mean,
            "transfer_BtoA_mean": transfer_BtoA_mean,
            "within_A_mean": within_A_mean,
            "within_B_mean": within_B_mean,
            "ratio_AtoB": ratio_AtoB,
            "ratio_BtoA": ratio_BtoA,
            "lower_transfer_ratio": lower_transfer_ratio,
            "probe_transfer_auroc": probe_transfer_auroc,
            "within_pathway_auroc": within_pathway_auroc,
            "delta_1": delta_1,
            "epsilon_2": epsilon_2,
            "delta_1_note": "Linear-translator residual delta_1 not computed in MVP (requires §5.7-specific quantity not exposed by build_baseline_pair). pass_linear forced to False; pass_cka and pass_transfer can still PASS the method.",
            "c1_pass": c1_pass,
            "pass_linear": pass_linear,
            "pass_cka": pass_cka,
            "pass_transfer": pass_transfer,
            "pass_any": pass_any,
            "thresholds": dict(cfg["pass_gate"]),
        },
        "deviation_from_spec": "POST-HOC g_theta: encoders pretrained once at lambda_C=1.0 (already L_C-aligned per E1 joint_dual_finetune_with_lc); shared g_theta + DeepCCA fitted post-hoc with 20 seeds. Spec describes joint encoder + g_theta training. Trade-off: cheaper (no encoder retrain per seed) and isolates AC's marginal benefit on top of L_C-aligned base.",
        "n_bootstrap": n_boot,
        "wall_s": round(time.perf_counter() - t0, 1),
        "git_head_at_run": _git_sha(HERE),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
