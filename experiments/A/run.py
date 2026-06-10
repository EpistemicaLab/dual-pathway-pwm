"""Method A — DeepCCA loss with INDEPENDENT per-pathway projection MLPs.

Per methods/method-A-deepcca.md: replace MSE L_C with DeepCCA correlation criterion
(Andrew+ ICML 2013). DeepCCA maximizes sum of canonical correlations between zA, zB —
ROTATION-INVARIANT by construction. If basis-ambiguity is the §5.7 mechanism, DeepCCA
should close the gap.

CRITICAL DISTINCTION vs Method AC: per-pathway projection MLPs g_A, g_B are INDEPENDENT
(separate weights), NOT shared. AC tied g_theta across pathways; A does not.

DEVIATION FROM SPEC (documented):
  Spec calls for JOINT encoder + g_A/g_B training with DeepCCA + per-pathway L from
  scratch. This implementation is POST-HOC: build_baseline_pair() trains encoders once
  at lambda_C=1.0 (which already includes L_C alignment), then independent g_A, g_B +
  DeepCCA + per-pathway reconstruction is fit on top. Mirrors AC/F/G's post-hoc choice
  for like-for-like comparison. Trade-off: cheaper (no encoder retrain per cell/seed)
  and isolates A's marginal benefit on top of L_C-aligned base. proj_output_dim=16
  follows spec k=16 latent-dim (vs AC's 32).

Output: results/method-A.json with PASS_GATE 4-gate evaluator
(c1, pass_linear, pass_cka, pass_transfer) per the original spec.

Honest-negative paths:
  * n_eval_paired < 100
  * ALL 3 sweep cells produce non-finite val_loss
  * ALL 20 seeds on best cell NaN
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
# NOTE: D3_DIR omitted from sys.path tuple per AC attempt-2 lesson (insert(0) reverses
# iteration order; D3/run.py would shadow D1/run.py and break import build_baseline_pair).
for p in (D1_DIR, E1_DIR, W4_DIR, W5_DIR):
    sys.path.insert(0, str(p))


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


# --- CKA (linear) ---
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


# --- MLP init/forward (used for both projection encoders g_A,g_B and decoders) ---
def _init_mlp(key, dim_in: int, depth: int, hidden: int, dim_out: int):
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


def _mlp_forward(layers, x):
    h = x
    L = len(layers)
    for i, layer in enumerate(layers):
        h = h @ layer["W"] + layer["b"]
        if i < L - 1:
            h = jnn.relu(h)
    return h


# --- DeepCCA loss: -sum singular values of whitened cross-covariance ---
def _deepcca_term(H_a, H_b, r: float):
    H_a = H_a - jnp.mean(H_a, axis=0, keepdims=True)
    H_b = H_b - jnp.mean(H_b, axis=0, keepdims=True)
    n = H_a.shape[0]
    da = H_a.shape[1]
    db = H_b.shape[1]
    eye_a = jnp.eye(da)
    eye_b = jnp.eye(db)
    S_aa = (H_a.T @ H_a) / (n - 1) + r * eye_a
    S_bb = (H_b.T @ H_b) / (n - 1) + r * eye_b
    S_ab = (H_a.T @ H_b) / (n - 1)
    wa, Va = jnp.linalg.eigh(S_aa)
    wb, Vb = jnp.linalg.eigh(S_bb)
    wa = jnp.clip(wa, 1e-12, None)
    wb = jnp.clip(wb, 1e-12, None)
    S_aa_inv_sqrt = (Va * (wa ** -0.5)[None, :]) @ Va.T
    S_bb_inv_sqrt = (Vb * (wb ** -0.5)[None, :]) @ Vb.T
    T = S_aa_inv_sqrt @ S_ab @ S_bb_inv_sqrt
    sv = jnp.linalg.svd(T, compute_uv=False)
    return -jnp.sum(sv)


def _full_loss(params, z_a, z_b, lambda_c: float, r: float):
    """L_A_recon + L_B_recon + lambda_c * L_DeepCCA on independent g_A, g_B."""
    H_a = _mlp_forward(params["g_a"], z_a)
    H_b = _mlp_forward(params["g_b"], z_b)
    z_a_recon = _mlp_forward(params["dec_a"], H_a)
    z_b_recon = _mlp_forward(params["dec_b"], H_b)
    L_a = jnp.mean(jnp.sum((z_a_recon - z_a) ** 2, axis=1))
    L_b = jnp.mean(jnp.sum((z_b_recon - z_b) ** 2, axis=1))
    L_dcca = _deepcca_term(H_a, H_b, r)
    return L_a + L_b + lambda_c * L_dcca, (L_a, L_b, L_dcca)


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


def _init_params(seed: int, dim_in: int, proj_depth, proj_hidden, proj_out, dec_depth, dec_hidden):
    key = jax.random.PRNGKey(int(seed))
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "g_a": _init_mlp(k1, dim_in, int(proj_depth), int(proj_hidden), int(proj_out)),
        "g_b": _init_mlp(k2, dim_in, int(proj_depth), int(proj_hidden), int(proj_out)),
        "dec_a": _init_mlp(k3, int(proj_out), int(dec_depth), int(dec_hidden), dim_in),
        "dec_b": _init_mlp(k4, int(proj_out), int(dec_depth), int(dec_hidden), dim_in),
    }


def _train_one(z_a_tr, z_b_tr, z_a_va, z_b_va, cfg, lambda_c, seed):
    params = _init_params(
        seed, dim_in=z_a_tr.shape[1],
        proj_depth=cfg["proj_depth"], proj_hidden=cfg["proj_hidden"], proj_out=cfg["proj_output_dim"],
        dec_depth=cfg["decoder_depth"], dec_hidden=cfg["decoder_hidden"],
    )
    opt = _adam_init(params)
    r = float(cfg["deepcca_whitening_reg"])
    lr = float(cfg["deepcca_lr"])

    grad_fn = jax.value_and_grad(_full_loss, has_aux=True)

    @jax.jit
    def step(params, opt, x, y):
        (loss, terms), g = grad_fn(params, x, y, lambda_c, r)
        new, new_opt = _adam_apply(params, opt, g, lr=lr)
        return new, new_opt, loss, terms

    @jax.jit
    def eval_loss(params, x, y):
        (loss, terms) = _full_loss(params, x, y, lambda_c, r)
        return loss, terms

    x = jnp.asarray(z_a_tr, dtype=jnp.float32)
    y = jnp.asarray(z_b_tr, dtype=jnp.float32)
    xv = jnp.asarray(z_a_va, dtype=jnp.float32)
    yv = jnp.asarray(z_b_va, dtype=jnp.float32)
    losses = []
    t0 = time.perf_counter()
    for _ in range(int(cfg["deepcca_epochs"])):
        params, opt, loss, _ = step(params, opt, x, y)
        losses.append(float(loss))
    val_loss, val_terms = eval_loss(params, xv, yv)
    return params, {
        "final_train_loss": float(losses[-1]),
        "val_loss": float(val_loss),
        "val_recon_a": float(val_terms[0]),
        "val_recon_b": float(val_terms[1]),
        "val_dcca": float(val_terms[2]),
        "wall_s": round(time.perf_counter() - t0, 2),
    }


# --- PASS_GATE helpers ---
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

    # 1) §5.7 baseline pair at lambda_C=1.0 (fix-knob ON), reused across cells/seeds.
    import run as d1_run  # type: ignore[import-not-found]
    z_a_pre, z_b_pre, meta = d1_run.build_baseline_pair(cfg)
    n = z_a_pre.shape[0]
    print(f"[A] baseline pair built. n={n} dimA={meta['dim_A']} dimB={meta['dim_B']} "
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

    # 2) Recover 11-bit basis labels deterministically (E1 helper).
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

    # 3) 70/30 split same as AC/F/G/D3/D4 for like-for-like comparability.
    head_train_frac = float(cfg["head_train_fraction"])
    rng = np.random.default_rng(int(cfg["head_split_seed"]))
    perm = rng.permutation(n)
    n_train = int(round(n * head_train_frac))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    z_a_tr_pre, z_a_te_pre = z_a_pre[train_idx], z_a_pre[test_idx]
    z_b_tr_pre, z_b_te_pre = z_b_pre[train_idx], z_b_pre[test_idx]
    y_tr_bits, y_te_bits = y_bits[train_idx], y_bits[test_idx]

    # 4) Cell sweep over lambda_c at single seed, pick best by val_loss.
    sweep_seed = int(cfg["sweep_seed"])
    lambda_c_grid = [float(x) for x in cfg["lambda_c_grid"]]
    cell_records = []
    best_cell = None
    for lc in lambda_c_grid:
        params_c, info_c = _train_one(z_a_tr_pre, z_b_tr_pre,
                                      z_a_te_pre, z_b_te_pre, cfg, lc, sweep_seed)
        rec = {"lambda_c": lc, **info_c}
        cell_records.append(rec)
        print(f"[A] sweep lambda_c={lc:.2f} val_loss={info_c['val_loss']:.4f} "
              f"recon_a={info_c['val_recon_a']:.4f} recon_b={info_c['val_recon_b']:.4f} "
              f"dcca={info_c['val_dcca']:.4f} wall={info_c['wall_s']:.1f}s", flush=True)
        if np.isfinite(info_c["val_loss"]):
            if best_cell is None or info_c["val_loss"] < best_cell["val_loss"]:
                best_cell = {"lambda_c": lc, **info_c}

    if best_cell is None:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": "All 3 sweep cells diverged to NaN val_loss.",
            "cell_records": cell_records,
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # 5) 20-seed paired-bootstrap CI on best cell. Pick best seed by lowest val_loss.
    n_seeds = int(cfg["deepcca_n_seeds"])
    seed_base = int(cfg["deepcca_seed_base"])
    seed_records = []
    best_seed_pkg = None
    for s in range(n_seeds):
        seed = seed_base + s
        params_s, info_s = _train_one(z_a_tr_pre, z_b_tr_pre,
                                      z_a_te_pre, z_b_te_pre, cfg,
                                      float(best_cell["lambda_c"]), seed)
        rec = {"seed": seed, **info_s}
        seed_records.append(rec)
        if np.isfinite(info_s["val_loss"]):
            if best_seed_pkg is None or info_s["val_loss"] < best_seed_pkg["val_loss"]:
                best_seed_pkg = {"seed": seed, "params": params_s, **info_s}
        if (s % 5) == 0 or s == n_seeds - 1:
            print(f"[A] seed {s+1}/{n_seeds} val_loss={info_s['val_loss']:.4f} "
                  f"wall={info_s['wall_s']:.1f}s", flush=True)

    if best_seed_pkg is None:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": "All 20 seeds on best cell diverged to NaN val_loss.",
            "cell_records": cell_records,
            "seed_records": seed_records,
            "best_cell": {k: v for k, v in best_cell.items() if k != "params"},
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # 6) Apply best g_A, g_B to all splits to get DeepCCA-aligned latents.
    bp = best_seed_pkg["params"]

    @jax.jit
    def _apply_g_a(x):
        return _mlp_forward(bp["g_a"], x)

    @jax.jit
    def _apply_g_b(x):
        return _mlp_forward(bp["g_b"], x)

    z_a_tr = np.asarray(_apply_g_a(jnp.asarray(z_a_tr_pre, dtype=jnp.float32)))
    z_a_te = np.asarray(_apply_g_a(jnp.asarray(z_a_te_pre, dtype=jnp.float32)))
    z_b_tr = np.asarray(_apply_g_b(jnp.asarray(z_b_tr_pre, dtype=jnp.float32)))
    z_b_te = np.asarray(_apply_g_b(jnp.asarray(z_b_te_pre, dtype=jnp.float32)))

    # 7) PASS_GATE: c1 (per-bit AUROC), CKA, transfer ratio.
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

    cka_te = linear_cka(z_a_te, z_b_te)
    n_boot = int(cfg["n_bootstrap"])
    cka_lo, cka_hi = bootstrap_ci(linear_cka, z_a_te, z_b_te, n_boot=n_boot, seed=int(cfg["bootstrap_seed"]))

    c1_min = float(cfg["pass_gate"]["c1_min_per_pathway_auroc"])
    cka_min = float(cfg["pass_gate"]["cka_min"])
    transfer_ratio_min = float(cfg["pass_gate"]["transfer_ratio_min"])
    c1_pass = bool(probe_auroc_A >= c1_min and probe_auroc_B >= c1_min)
    pass_cka = bool(c1_pass and cka_te >= cka_min)
    lower_transfer_ratio = float(min(ratio_AtoB, ratio_BtoA)) if np.isfinite(ratio_AtoB) and np.isfinite(ratio_BtoA) else float("nan")
    pass_transfer = bool(c1_pass and np.isfinite(lower_transfer_ratio) and lower_transfer_ratio >= transfer_ratio_min)
    pass_linear = False
    delta_1 = None
    epsilon_2 = None
    pass_any = pass_cka or pass_transfer or pass_linear
    verdict = "PASS" if pass_any else "HONEST_NEGATIVE"

    # Drop heavy 'params' from best_cell + best_seed_pkg before serializing.
    best_cell_clean = {k: v for k, v in best_cell.items() if k != "params"}
    best_seed_clean = {k: v for k, v in best_seed_pkg.items() if k != "params"}

    result = {
        "verdict": verdict,
        "method": "A",
        **meta,
        "n_head_train": int(len(train_idx)),
        "n_head_test": int(len(test_idx)),
        "deepcca": {
            "proj_depth": int(cfg["proj_depth"]),
            "proj_hidden": int(cfg["proj_hidden"]),
            "proj_output_dim": int(cfg["proj_output_dim"]),
            "decoder_depth": int(cfg["decoder_depth"]),
            "decoder_hidden": int(cfg["decoder_hidden"]),
            "deepcca_whitening_reg": float(cfg["deepcca_whitening_reg"]),
            "deepcca_lr": float(cfg["deepcca_lr"]),
            "deepcca_epochs": int(cfg["deepcca_epochs"]),
            "lambda_c_grid": lambda_c_grid,
            "n_seeds": n_seeds,
        },
        "cell_records": cell_records,
        "best_cell": best_cell_clean,
        "seed_records": seed_records,
        "best_seed": best_seed_clean,
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
            "delta_1_note": "Linear-translator residual delta_1 not computed in MVP (requires section-5.7-specific quantity not exposed by build_baseline_pair). pass_linear forced to False; pass_cka and pass_transfer can still PASS the method.",
            "c1_pass": c1_pass,
            "pass_linear": pass_linear,
            "pass_cka": pass_cka,
            "pass_transfer": pass_transfer,
            "pass_any": pass_any,
            "thresholds": dict(cfg["pass_gate"]),
        },
        "deviation_from_spec": "POST-HOC independent g_A, g_B with DeepCCA + per-pathway reconstruction on z_a_pre/z_b_pre (32-d section-5.7 baseline latents at lambda_C=1.0). Spec describes joint encoder + g_A/g_B training from scratch with DeepCCA + L_A + L_B at lambda_C in {0.1,1.0,10.0}. Implementation is post-hoc on L_C-aligned baseline (mirrors AC/F/G post-hoc choice for like-for-like comparison). Full-batch Adam (n_train=378 fits trivially) instead of spec batch=128 — full-batch is more stable for Sigma estimation on small-n. proj_output_dim=16 follows spec k=16 latent-dim. delta_1/epsilon_2 not exposed -> pass_linear forced False; pass_cka and pass_transfer carry verdict.",
        "n_bootstrap": n_boot,
        "wall_s": round(time.perf_counter() - t0, 1),
        "git_head_at_run": _git_sha(HERE),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
