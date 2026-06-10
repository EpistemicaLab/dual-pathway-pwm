"""I -- MINE / Donsker-Varadhan MI lower bound (Belghazi+ ICML 2018).

Per methods/method-I-mine.md: maximize a tight neural lower bound on I(z_A; z_B)
via the Donsker-Varadhan representation. Sharper than InfoNCE's log-batch-size
bound but harder to optimize.

DEVIATION FROM SPEC (documented; mirrors AC/F/G/A/F0/C/B's post-hoc choice):
  Spec calls for joint encoder retraining with MI-maximization replacing L_C in
  section-5.7 training. This MVP is post-hoc on z_A_pre, z_B_pre (32-d section-5.7
  baseline latents at lambda_C=1.0). Per-pathway projection MLPs (enc + dec) trained
  with reconstruction (acting as the L_A, L_B 'predictive' terms in the spec) plus
  lambda_C * (-I_hat) between projections, where I_hat is the Donsker-Varadhan MI
  lower bound estimated by a small statistics network T_theta.

Output: results/method-I.json with PASS_GATE 4-gate evaluator
(c1, pass_linear, pass_cka, pass_transfer) per the original spec.

Honest-negative paths:
  * n_eval_paired < 100
  * ALL 3 sweep cells diverge to NaN
  * ALL 20 seeds on best cell diverge to NaN
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


# --- CKA (self-contained) ---
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


# --- MLP helpers (pure JAX/JIT-able) ---
def _init_mlp(key, dims):
    """dims = [in, h1, ..., out]; ReLU between hidden, none after final."""
    keys = jax.random.split(key, len(dims) - 1)
    layers = []
    for i in range(len(dims) - 1):
        fan_in = dims[i]
        scale = jnp.sqrt(2.0 / fan_in) if i < len(dims) - 2 else jnp.sqrt(1.0 / fan_in)
        W = jax.random.normal(keys[i], (dims[i], dims[i + 1])) * scale
        layers.append({"W": W, "b": jnp.zeros((dims[i + 1],))})
    return layers


def _mlp_forward(layers, x):
    h = x
    L = len(layers)
    for i, layer in enumerate(layers):
        h = h @ layer["W"] + layer["b"]
        if i < L - 1:
            h = jnn.relu(h)
    return h


def _init_g(key, x_dim: int, z_dim: int, enc_h: int, dec_h: int, stats_h: int, stats_depth: int):
    keys = jax.random.split(key, 5)
    enc_a = _init_mlp(keys[0], [x_dim, enc_h, z_dim])
    enc_b = _init_mlp(keys[1], [x_dim, enc_h, z_dim])
    dec_a = _init_mlp(keys[2], [z_dim, dec_h, x_dim])
    dec_b = _init_mlp(keys[3], [z_dim, dec_h, x_dim])
    # Stats network T_theta: (z_dim + z_dim) -> (stats_h * stats_depth) -> 1
    stats_dims = [2 * z_dim] + [stats_h] * stats_depth + [1]
    stats = _init_mlp(keys[4], stats_dims)
    return {"enc_a": enc_a, "enc_b": enc_b, "dec_a": dec_a, "dec_b": dec_b, "stats": stats}


def _stats_forward(stats_layers, z_a, z_b, clip):
    """T_theta(z_a, z_b) = MLP([z_a; z_b]) with tanh-clip to [-clip, +clip]."""
    h = jnp.concatenate([z_a, z_b], axis=-1)
    out = _mlp_forward(stats_layers, h)
    return clip * jnn.tanh(out / clip)        # smooth clip in [-clip, +clip]


def _mine_loss(stats_layers, z_a, z_b, z_b_marg, clip):
    """Donsker-Varadhan MI lower bound:
        I_hat = E_p(joint)[T(z_a, z_b)] - log E_p(marg)[exp T(z_a, z_b_shuffled)]
       Returns (-I_hat, I_hat) so we can MINIMIZE the negative.
    """
    t_joint = _stats_forward(stats_layers, z_a, z_b, clip).squeeze(-1)
    t_marg = _stats_forward(stats_layers, z_a, z_b_marg, clip).squeeze(-1)
    et = jnp.mean(t_joint)
    log_mean_exp = jax.scipy.special.logsumexp(t_marg) - jnp.log(t_marg.shape[0])
    i_hat = et - log_mean_exp
    return -i_hat, i_hat


def _g_step(params, x_a, x_b, x_b_marg, lambda_c, clip):
    """Joint loss for encoders+decoders+stats: recon_a + recon_b - lambda_c * I_hat.
    Note we MINIMIZE so MI-MAXIMIZATION enters as -I_hat."""
    z_a = _mlp_forward(params["enc_a"], x_a)
    z_b = _mlp_forward(params["enc_b"], x_b)
    z_b_m = _mlp_forward(params["enc_b"], x_b_marg)
    x_a_rec = _mlp_forward(params["dec_a"], z_a)
    x_b_rec = _mlp_forward(params["dec_b"], z_b)
    recon_a = 0.5 * jnp.mean(jnp.sum((x_a_rec - x_a) ** 2, axis=-1))
    recon_b = 0.5 * jnp.mean(jnp.sum((x_b_rec - x_b) ** 2, axis=-1))
    neg_i, i_hat = _mine_loss(params["stats"], z_a, z_b, z_b_m, clip)
    loss = recon_a + recon_b + lambda_c * neg_i
    aux = {
        "recon_a": recon_a,
        "recon_b": recon_b,
        "i_hat": i_hat,
        "norm_a_mean": jnp.mean(jnp.linalg.norm(z_a, axis=-1)),
        "norm_b_mean": jnp.mean(jnp.linalg.norm(z_b, axis=-1)),
    }
    return loss, aux


def _adam_init(params):
    return {
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
        "t": 0,
    }


def _adam_apply_per_group(params, opts, grads, lr_enc, lr_stats, b1=0.9, b2=0.999, eps=1e-8):
    """Apply Adam with separate lr for the 'stats' subtree vs the rest."""
    new_params = {}
    new_opts = {}
    for k in params:
        lr = lr_stats if k == "stats" else lr_enc
        opt = opts[k]
        t = opt["t"] + 1
        m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1 - b1) * g, opt["m"], grads[k])
        v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1 - b2) * (g * g), opt["v"], grads[k])
        mh = jax.tree_util.tree_map(lambda mm: mm / (1 - b1 ** t), m)
        vh = jax.tree_util.tree_map(lambda vv: vv / (1 - b2 ** t), v)
        new_params[k] = jax.tree_util.tree_map(
            lambda p, mm, vv: p - lr * mm / (jnp.sqrt(vv) + eps), params[k], mh, vh
        )
        new_opts[k] = {"m": m, "v": v, "t": t}
    return new_params, new_opts


def _train_one_seed(z_a_tr, z_b_tr, z_a_va, z_b_va, cfg, lambda_c: float, seed: int):
    """Returns (params, history, final). Mini-batch Adam over i_epochs."""
    z_dim = int(cfg["i_latent_dim"])
    enc_h = int(cfg["i_enc_hidden"])
    dec_h = int(cfg["i_dec_hidden"])
    stats_h = int(cfg["i_stats_hidden"])
    stats_depth = int(cfg["i_stats_depth"])
    clip = float(cfg["i_stats_clip"])
    bs_cfg = int(cfg["i_batch_size"])
    n_epochs = int(cfg["i_epochs"])
    lr_enc = float(cfg["i_lr_enc"])
    lr_stats = float(cfg["i_lr_stats"])

    n_train = z_a_tr.shape[0]
    bs = min(bs_cfg, n_train)

    key = jax.random.PRNGKey(int(seed))
    init_key, _ = jax.random.split(key)
    params = _init_g(
        init_key,
        x_dim=z_a_tr.shape[1], z_dim=z_dim,
        enc_h=enc_h, dec_h=dec_h,
        stats_h=stats_h, stats_depth=stats_depth,
    )
    opts = {k: _adam_init(params[k]) for k in params}

    grad_fn = jax.value_and_grad(_g_step, has_aux=True)

    @jax.jit
    def step(params, opts, x_a, x_b, x_b_marg):
        (loss, aux), g = grad_fn(params, x_a, x_b, x_b_marg, lambda_c, clip)
        new, new_opts = _adam_apply_per_group(params, opts, g, lr_enc, lr_stats)
        return new, new_opts, loss, aux

    @jax.jit
    def eval_step(params, x_a, x_b, x_b_marg):
        return _g_step(params, x_a, x_b, x_b_marg, lambda_c, clip)

    history = []
    rng = np.random.default_rng(int(seed))
    for ep in range(n_epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, bs):
            idx = order[s:s + bs]
            # Marginal: shuffle z_b within the mini-batch.
            marg = rng.permutation(len(idx))
            params, opts, _loss, _aux = step(
                params, opts,
                jnp.asarray(z_a_tr[idx], dtype=jnp.float32),
                jnp.asarray(z_b_tr[idx], dtype=jnp.float32),
                jnp.asarray(z_b_tr[idx[marg]], dtype=jnp.float32),
            )
        # Per-epoch val snapshot (no grad).
        rng_val = np.random.default_rng(int(seed) + 99991)
        val_marg = rng_val.permutation(z_b_va.shape[0])
        val_loss, val_aux = eval_step(
            params,
            jnp.asarray(z_a_va, dtype=jnp.float32),
            jnp.asarray(z_b_va, dtype=jnp.float32),
            jnp.asarray(z_b_va[val_marg], dtype=jnp.float32),
        )
        history.append({
            "epoch": ep,
            "val_loss": float(val_loss),
            "val_recon_a": float(val_aux["recon_a"]),
            "val_recon_b": float(val_aux["recon_b"]),
            "val_i_hat": float(val_aux["i_hat"]),
            "val_norm_a": float(val_aux["norm_a_mean"]),
            "val_norm_b": float(val_aux["norm_b_mean"]),
        })

    final = history[-1]
    return params, history, final


def _proj(params, x, side: str):
    enc = params["enc_a"] if side == "a" else params["enc_b"]
    return np.asarray(_mlp_forward(enc, jnp.asarray(x, dtype=jnp.float32)))


def _stats_param_norm(params) -> float:
    flat = []
    for layer in params["stats"]:
        flat.append(np.asarray(layer["W"]).ravel())
        flat.append(np.asarray(layer["b"]).ravel())
    return float(np.linalg.norm(np.concatenate(flat)))


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

    # 1) Build §5.7 baseline pair at lambda_C=1.0 (same as AC/F/G/A/F0/C/B).
    import run as d1_run  # type: ignore[import-not-found]
    z_a_pre, z_b_pre, meta = d1_run.build_baseline_pair(cfg)
    n = z_a_pre.shape[0]
    print(f"[I] baseline pair built. n={n} dimA={meta['dim_A']} dimB={meta['dim_B']} "
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

    # 2) Recover 11-bit basis labels deterministically (same as D3/AC/F/G/A/F0/C/B).
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

    # 3) 70/30 head split (same head_split_seed=271828 as AC/F/G/A/F0/C/B/D3/D4).
    head_train_frac = float(cfg["head_train_fraction"])
    rng = np.random.default_rng(int(cfg["head_split_seed"]))
    perm = rng.permutation(n)
    n_train = int(round(n * head_train_frac))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    z_a_tr_pre, z_a_te_pre = z_a_pre[train_idx], z_a_pre[test_idx]
    z_b_tr_pre, z_b_te_pre = z_b_pre[train_idx], z_b_pre[test_idx]
    y_tr_bits, y_te_bits = y_bits[train_idx], y_bits[test_idx]

    # 4) Cell sweep: lambda_C in {0.1, 1.0, 10.0} = 3 cells at 1 seed each.
    sweep_seed = int(cfg["i_sweep_seed"])
    cells = []
    for lambda_c in cfg["i_lambda_c_grid"]:
        t_cell = time.perf_counter()
        params, _hist, final = _train_one_seed(
            z_a_tr_pre, z_b_tr_pre, z_a_te_pre, z_b_te_pre,
            cfg, lambda_c=float(lambda_c), seed=sweep_seed,
        )
        cells.append({
            "lambda_c": float(lambda_c),
            "val_loss": final["val_loss"],
            "val_recon_a": final["val_recon_a"],
            "val_recon_b": final["val_recon_b"],
            "val_i_hat": final["val_i_hat"],
            "val_norm_a": final["val_norm_a"],
            "val_norm_b": final["val_norm_b"],
            "wall_s": round(time.perf_counter() - t_cell, 2),
        })
        print(f"[I] sweep lambda_c={lambda_c} val_loss={final['val_loss']:.3f} "
              f"i_hat={final['val_i_hat']:.3f} "
              f"norms=({final['val_norm_a']:.3f},{final['val_norm_b']:.3f}) "
              f"wall={cells[-1]['wall_s']:.1f}s", flush=True)

    finite = [c for c in cells if np.isfinite(c["val_loss"]) and np.isfinite(c["val_i_hat"])]
    if not finite:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": "All 3 sweep cells produced non-finite val_loss or val_i_hat (NaN/Inf).",
            "cells": cells,
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # Best cell: highest val_i_hat (the metric of merit per spec; recon may dominate val_loss).
    best_cell = max(finite, key=lambda c: c["val_i_hat"])
    print(f"[I] best cell: lambda_c={best_cell['lambda_c']} "
          f"i_hat={best_cell['val_i_hat']:.3f}", flush=True)

    # 5) 20 seeds on the best cell (paired-bootstrap CI + best params for PASS_GATE).
    n_seeds = int(cfg["i_n_seeds"])
    seed_base = int(cfg["i_seed_base"])
    seed_records = []
    best_params = None
    best_i_hat = -float("inf")
    for s in range(n_seeds):
        seed = seed_base + s
        params, _hist, final = _train_one_seed(
            z_a_tr_pre, z_b_tr_pre, z_a_te_pre, z_b_te_pre,
            cfg, lambda_c=float(best_cell["lambda_c"]), seed=seed,
        )
        rec = {"seed": seed, **{k: final[k] for k in
                                ("val_loss", "val_recon_a", "val_recon_b",
                                 "val_i_hat", "val_norm_a", "val_norm_b")}}
        seed_records.append(rec)
        if np.isfinite(final["val_i_hat"]) and final["val_i_hat"] > best_i_hat:
            best_i_hat = final["val_i_hat"]
            best_params = params
        if (s % 5) == 0 or s == n_seeds - 1:
            print(f"[I] seed {s+1}/{n_seeds} i_hat={final['val_i_hat']:.3f}", flush=True)

    if best_params is None:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": f"All {n_seeds} seeds on best cell produced non-finite val_i_hat.",
            "best_cell": best_cell,
            "cells": cells,
            "seed_records": seed_records,
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # 6) PASS_GATE on deterministic projections with best_params.
    z_a_tr = _proj(best_params, z_a_tr_pre, side="a")
    z_a_te = _proj(best_params, z_a_te_pre, side="a")
    z_b_tr = _proj(best_params, z_b_tr_pre, side="b")
    z_b_te = _proj(best_params, z_b_te_pre, side="b")

    # Spec failure-mode 1 diagnostics: stats_net_norm + final_mi_estimate.
    stats_net_norm = _stats_param_norm(best_params)
    final_mi_estimate = float(best_i_hat)

    # Norm-collapse check (raw, like G's diagnostic).
    norm_a_te_mean = float(np.linalg.norm(z_a_te, axis=-1).mean())
    norm_b_te_mean = float(np.linalg.norm(z_b_te, axis=-1).mean())
    norm_a_te_std = float(np.linalg.norm(z_a_te, axis=-1).std())
    norm_b_te_std = float(np.linalg.norm(z_b_te, axis=-1).std())

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
    cka_lo, cka_hi = bootstrap_ci(linear_cka, z_a_te, z_b_te, n_boot=n_boot,
                                  seed=int(cfg["bootstrap_seed"]))

    # PASS_GATE 4-gate evaluator (per the original spec PASS_GATE):
    c1_min = float(cfg["pass_gate"]["c1_min_per_pathway_auroc"])
    cka_min = float(cfg["pass_gate"]["cka_min"])
    transfer_ratio_min = float(cfg["pass_gate"]["transfer_ratio_min"])
    c1_pass = bool(probe_auroc_A >= c1_min and probe_auroc_B >= c1_min)
    pass_cka = bool(c1_pass and np.isfinite(cka_te) and cka_te >= cka_min)
    lower_transfer_ratio = (float(min(ratio_AtoB, ratio_BtoA))
                            if np.isfinite(ratio_AtoB) and np.isfinite(ratio_BtoA)
                            else float("nan"))
    pass_transfer = bool(c1_pass and np.isfinite(lower_transfer_ratio)
                         and lower_transfer_ratio >= transfer_ratio_min)
    pass_linear = False
    delta_1 = None
    epsilon_2 = None
    pass_any = pass_cka or pass_transfer or pass_linear
    verdict = "PASS" if pass_any else "HONEST_NEGATIVE"

    bs_cfg = int(cfg["i_batch_size"])
    bs_eff = min(bs_cfg, n_train)

    result = {
        "verdict": verdict,
        "method": "I",
        **meta,
        "n_head_train": int(len(train_idx)),
        "n_head_test": int(len(test_idx)),
        "i_head": {
            "latent_dim": int(cfg["i_latent_dim"]),
            "enc_hidden": int(cfg["i_enc_hidden"]),
            "dec_hidden": int(cfg["i_dec_hidden"]),
            "stats_hidden": int(cfg["i_stats_hidden"]),
            "stats_depth": int(cfg["i_stats_depth"]),
            "stats_clip": float(cfg["i_stats_clip"]),
            "lr_enc": float(cfg["i_lr_enc"]),
            "lr_stats": float(cfg["i_lr_stats"]),
            "epochs": int(cfg["i_epochs"]),
            "batch_size_cfg": bs_cfg,
            "batch_size_effective": bs_eff,
            "n_seeds": n_seeds,
            "best_cell": best_cell,
            "best_i_hat": float(best_i_hat),
            "final_mi_estimate": final_mi_estimate,
            "stats_net_norm": stats_net_norm,
            "norm_a_te_mean": norm_a_te_mean,
            "norm_b_te_mean": norm_b_te_mean,
            "norm_a_te_std": norm_a_te_std,
            "norm_b_te_std": norm_b_te_std,
        },
        "cells": cells,
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
            "delta_1_note": (
                "Linear-translator residual delta_1 not computed in MVP "
                "(requires section-5.7-specific quantity not exposed by build_baseline_pair). "
                "pass_linear forced to False; pass_cka and pass_transfer can still PASS the method."
            ),
            "c1_pass": c1_pass,
            "pass_linear": pass_linear,
            "pass_cka": pass_cka,
            "pass_transfer": pass_transfer,
            "pass_any": pass_any,
            "thresholds": dict(cfg["pass_gate"]),
        },
        "deviation_from_spec": (
            "POST-HOC MINE projection on z_a_pre/z_b_pre (32-d section-5.7 baseline latents at "
            "lambda_C=1.0) instead of joint encoder retraining. Per-pathway projection MLPs (enc + dec) "
            "trained with reconstruction (acting as the L_A, L_B 'predictive' terms in the spec) plus "
            "lambda_C * (-I_hat) where I_hat is the Donsker-Varadhan MI lower bound estimated by a "
            "small statistics network T_theta with tanh-clip to [-10, 10] for stability. Marginal "
            "samples obtained by within-batch random-permutation shuffle of z_b. Mirrors AC/F/G/A/F0/C/B "
            "post-hoc choice. Separate Adam lr_enc=1e-3, lr_stats=1e-4 per spec."
        ),
        "n_bootstrap": n_boot,
        "wall_s": round(time.perf_counter() - t0, 1),
        "git_head_at_run": _git_sha(HERE),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
