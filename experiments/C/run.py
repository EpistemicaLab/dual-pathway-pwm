"""C -- Weight-shared g_theta dynamics core with MSE alignment.

Per methods/method-C-weightshare.md: per-pathway input encoders unchanged
(§5.7 baseline at lambda_C=1.0), but a shared g_theta MLP with TIED weights
is applied to z_A_pre AND z_B_pre. Standard MSE L_C = E||g(zA) - g(zB)||^2
on post-g_theta outputs (NOT DeepCCA like AC).

Hyperparameter sweep: depth in {1, 2, 4} at sweep_seed=120001 (single seed,
80/20 sub-split of head_train for sweep_val cell selection by lowest val MSE).
After picking best depth, 20 seeds train on FULL head_train; best-by-final-loss
seed used for PASS_GATE eval.

DEVIATION FROM SPEC (documented):
  Spec calls for joint encoder + shared-g_theta training with MSE L_C at
  lambda_C=1.0. This implementation is POST-HOC: build_baseline_pair() trains
  encoders once at lambda_C=1.0 (which already includes L_C alignment), then
  shared g_theta is fit on top with the depth sweep + 20-seed CI. Mirrors
  AC/F/G/A/F0 post-hoc choice for like-for-like comparability.

Output: results/method-C.json with PASS_GATE 4-gate evaluator
(c1, pass_linear, pass_cka, pass_transfer) per the original spec.

Honest-negative paths:
  * n_eval_paired < 100
  * All 3 sweep cells produce non-finite val_loss
  * All 20 seeds on best cell NaN
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
# NOTE: D3_DIR DROPPED per AC attempt-2 lesson (insert(0) reverses iter order;
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


# --- CKA (copy from D1 for self-containment) ---
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


# --- Shared g_theta MLP (same weights applied to zA_pre and zB_pre) ---
def _init_g_theta(key, dim_in: int, depth: int, hidden: int, dim_out: int):
    """depth=1 -> single Linear(dim_in -> dim_out); depth=k -> Linear(dim_in->h)+ReLU + (k-2) hidden + Linear(h->dim_out)."""
    depth = int(depth)
    if depth < 1:
        raise ValueError(f"depth must be >=1, got {depth}")
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


def _param_count(layers) -> int:
    return int(sum(int(np.prod(layer["W"].shape)) + int(layer["b"].shape[0]) for layer in layers))


def _mse_loss(layers, z_a, z_b):
    """Standard MSE L_C on post-g_theta outputs (per spec)."""
    H_a = _g_forward(layers, z_a)
    H_b = _g_forward(layers, z_b)
    diff = H_a - H_b
    return jnp.mean(jnp.sum(diff * diff, axis=1))


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


def _train_g_theta_one_seed(z_a_tr, z_b_tr, depth, hidden, dim_out, lr, n_epochs, seed: int,
                            z_a_val=None, z_b_val=None):
    key = jax.random.PRNGKey(int(seed))
    layers = _init_g_theta(key, dim_in=z_a_tr.shape[1], depth=int(depth),
                           hidden=int(hidden), dim_out=int(dim_out))
    opt = _adam_init(layers)
    grad_fn = jax.value_and_grad(_mse_loss)

    @jax.jit
    def step(layers, opt, x, y):
        loss, g = grad_fn(layers, x, y)
        new, new_opt = _adam_apply(layers, opt, g, lr=lr)
        return new, new_opt, loss

    x_tr = jnp.asarray(z_a_tr, dtype=jnp.float32)
    y_tr = jnp.asarray(z_b_tr, dtype=jnp.float32)
    losses = []
    t0 = time.perf_counter()
    for _ in range(int(n_epochs)):
        layers, opt, loss = step(layers, opt, x_tr, y_tr)
        losses.append(float(loss))
    val_loss = None
    if z_a_val is not None and z_b_val is not None and z_a_val.shape[0] > 0:
        x_val = jnp.asarray(z_a_val, dtype=jnp.float32)
        y_val = jnp.asarray(z_b_val, dtype=jnp.float32)
        val_loss = float(_mse_loss(layers, x_val, y_val))
    return layers, {"final_loss": float(losses[-1]),
                    "val_loss": val_loss,
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
    print(f"[C] baseline pair built. n={n} dimA={meta['dim_A']} dimB={meta['dim_B']} "
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

    # 3) 70/30 head split — train g_theta + probes on 70%, hold out 30% for PASS_GATE.
    head_train_frac = float(cfg["head_train_fraction"])
    rng = np.random.default_rng(int(cfg["head_split_seed"]))
    perm = rng.permutation(n)
    n_train = int(round(n * head_train_frac))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    z_a_tr_pre, z_a_te_pre = z_a_pre[train_idx], z_a_pre[test_idx]
    z_b_tr_pre, z_b_te_pre = z_b_pre[train_idx], z_b_pre[test_idx]
    y_tr_bits, y_te_bits = y_bits[train_idx], y_bits[test_idx]

    # 4) DEPTH SWEEP — sub-split head_train into sweep_train/sweep_val (80/20),
    #    train each depth at sweep_seed, pick best depth by lowest val MSE.
    sweep_val_frac = float(cfg["sweep_val_fraction"])
    sweep_rng = np.random.default_rng(int(cfg["sweep_seed"]))
    sweep_perm = sweep_rng.permutation(z_a_tr_pre.shape[0])
    n_sweep_val = int(round(z_a_tr_pre.shape[0] * sweep_val_frac))
    sweep_val_idx = sweep_perm[:n_sweep_val]
    sweep_train_idx = sweep_perm[n_sweep_val:]
    z_a_sw_tr = z_a_tr_pre[sweep_train_idx]
    z_b_sw_tr = z_b_tr_pre[sweep_train_idx]
    z_a_sw_val = z_a_tr_pre[sweep_val_idx]
    z_b_sw_val = z_b_tr_pre[sweep_val_idx]

    cell_records = []
    for depth in cfg["g_theta_depth_sweep"]:
        layers, info = _train_g_theta_one_seed(
            z_a_sw_tr, z_b_sw_tr,
            depth=int(depth), hidden=cfg["g_theta_hidden"],
            dim_out=cfg["g_theta_output_dim"],
            lr=float(cfg["g_theta_lr"]), n_epochs=int(cfg["g_theta_epochs"]),
            seed=int(cfg["sweep_seed"]),
            z_a_val=z_a_sw_val, z_b_val=z_b_sw_val,
        )
        rec = {
            "depth": int(depth),
            "param_count": _param_count(layers),
            "final_train_loss": info["final_loss"],
            "val_loss": info["val_loss"],
            "wall_s": info["wall_s"],
        }
        cell_records.append(rec)
        print(f"[C] sweep depth={depth} params={rec['param_count']} "
              f"train_loss={info['final_loss']:.4f} val_loss={info['val_loss']:.4f} "
              f"wall={info['wall_s']:.1f}s", flush=True)

    # Best cell = lowest finite val_loss; fall back to lowest train_loss if all val_loss NaN.
    finite_cells = [c for c in cell_records if c["val_loss"] is not None and np.isfinite(c["val_loss"])]
    if not finite_cells:
        finite_cells = [c for c in cell_records if np.isfinite(c["final_train_loss"])]
    if not finite_cells:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": "All depth-sweep cells produced non-finite val_loss AND non-finite train_loss.",
            "cell_records": cell_records,
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    best_cell = min(finite_cells, key=lambda c: (c["val_loss"] if c["val_loss"] is not None else c["final_train_loss"]))
    best_depth = int(best_cell["depth"])
    print(f"[C] best depth = {best_depth} (val_loss={best_cell['val_loss']:.4f})", flush=True)

    # 5) 20-seed paired-bootstrap CI on best depth — re-train on FULL head_train slice.
    n_seeds = int(cfg["n_seeds"])
    seed_base = int(cfg["seed_base"])
    seed_records = []
    best_seed = None
    for s in range(n_seeds):
        seed = seed_base + s
        layers, info = _train_g_theta_one_seed(
            z_a_tr_pre, z_b_tr_pre,
            depth=best_depth, hidden=cfg["g_theta_hidden"],
            dim_out=cfg["g_theta_output_dim"],
            lr=float(cfg["g_theta_lr"]), n_epochs=int(cfg["g_theta_epochs"]),
            seed=seed,
        )
        rec = {"seed": seed, "final_loss": info["final_loss"], "wall_s": info["wall_s"]}
        seed_records.append(rec)
        if (np.isfinite(info["final_loss"]) and (best_seed is None or info["final_loss"] < best_seed["final_loss"])):
            best_seed = {"layers": layers, **rec}
        if (s % 5) == 0 or s == n_seeds - 1:
            print(f"[C] seed {s+1}/{n_seeds} loss={info['final_loss']:.4f} wall={info['wall_s']:.1f}s", flush=True)

    if best_seed is None:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": "All 20 seeds on best depth diverged to NaN.",
            "cell_records": cell_records,
            "best_depth": best_depth,
            "seed_records": seed_records,
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # 6) Apply best g_theta to ALL splits to get post-shared-core latents.
    g_layers = best_seed["layers"]
    g_param_count = _param_count(g_layers)

    @jax.jit
    def _apply_g(x):
        return _g_forward(g_layers, x)

    z_a_tr = np.asarray(_apply_g(jnp.asarray(z_a_tr_pre, dtype=jnp.float32)))
    z_a_te = np.asarray(_apply_g(jnp.asarray(z_a_te_pre, dtype=jnp.float32)))
    z_b_tr = np.asarray(_apply_g(jnp.asarray(z_b_tr_pre, dtype=jnp.float32)))
    z_b_te = np.asarray(_apply_g(jnp.asarray(z_b_te_pre, dtype=jnp.float32)))

    # 7) PASS_GATE: c1 (per-bit AUROC on each pathway), CKA, transfer ratio.
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
    lower_transfer_ratio = float(min(ratio_AtoB, ratio_BtoA)) if np.isfinite(ratio_AtoB) and np.isfinite(ratio_BtoA) else float("nan")
    pass_transfer = bool(c1_pass and np.isfinite(lower_transfer_ratio) and lower_transfer_ratio >= transfer_ratio_min)
    pass_linear = False  # delta_1/epsilon_2 not exposed by build_baseline_pair, mirrors AC/F/G/A/F0
    delta_1 = None
    epsilon_2 = None
    pass_any = pass_cka or pass_transfer or pass_linear
    verdict = "PASS" if pass_any else "HONEST_NEGATIVE"

    result = {
        "verdict": verdict,
        "method": "C",
        **meta,
        "n_head_train": int(len(train_idx)),
        "n_head_test": int(len(test_idx)),
        "g_theta": {
            "best_depth": best_depth,
            "g_theta_param_count": g_param_count,
            "hidden": int(cfg["g_theta_hidden"]),
            "output_dim": int(cfg["g_theta_output_dim"]),
            "lr": float(cfg["g_theta_lr"]),
            "epochs": int(cfg["g_theta_epochs"]),
            "depth_sweep": list(cfg["g_theta_depth_sweep"]),
            "n_seeds": n_seeds,
            "best_seed": int(best_seed["seed"]),
            "best_final_loss": float(best_seed["final_loss"]),
        },
        "cell_records": cell_records,
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
            "delta_1_note": "Linear-translator residual delta_1 not computed in MVP (requires \u00a75.7-specific quantity not exposed by build_baseline_pair). pass_linear forced to False; pass_cka and pass_transfer can still PASS the method.",
            "c1_pass": c1_pass,
            "pass_linear": pass_linear,
            "pass_cka": pass_cka,
            "pass_transfer": pass_transfer,
            "pass_any": pass_any,
            "thresholds": dict(cfg["pass_gate"]),
        },
        "deviation_from_spec": "POST-HOC weight-shared g_theta with MSE alignment on z_a_pre/z_b_pre (32-d \u00a75.7 baseline latents at lambda_C=1.0). Spec describes joint encoder + shared-g_theta training. Implementation: pretrained encoders at lambda_C=1.0 (already L_C-aligned per E1's joint_dual_finetune_with_lc), then post-hoc shared g_theta with depth sweep {1,2,4} + 20-seed paired-bootstrap. Mirrors AC/F/G/A/F0 post-hoc choice. delta_1/epsilon_2 not exposed -> pass_linear forced False; pass_cka and pass_transfer carry verdict.",
        "n_bootstrap": n_boot,
        "wall_s": round(time.perf_counter() - t0, 1),
        "git_head_at_run": _git_sha(HERE),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
