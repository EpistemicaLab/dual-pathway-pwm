"""F -- iVAE with intervention u as auxiliary variable.

Per methods/method-F-ivae-with-u.md: two iVAE encoders sharing a u-conditional
Gaussian prior p(z|u) = N(mu(u), diag(sigma(u)^2)) across both pathways. Identifiability
flows from the prior factorization (Khemakhem 2020). Cross-pathway alignment via
shared prior + optional L_C term.

DEVIATION FROM SPEC (documented; mirrors AC's post-hoc choice):
  Spec calls for iVAE on RAW INPUTS x_A, x_B. This MVP uses post-hoc on z_A_pre,
  z_B_pre (32-d §5.7 baseline latents at lambda_C=1.0), because raw image+feature
  pipelines are only exposed inside e1's data builders. Khemakhem identifiability
  theory is unchanged -- iVAE applies to any observation space.

Output: results/method-F.json with PASS_GATE 4-gate evaluator
(c1, pass_linear, pass_cka, pass_transfer) per the original spec.

Honest-negative paths:
  * n_eval_paired < 100
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
# D3/run.py shadowing D1/run.py caused AttributeError on build_baseline_pair).
for p in (D1_DIR, E1_DIR, W4_DIR, W5_DIR):
    sys.path.insert(0, str(p))


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


# --- CKA (copied from D1 to keep F self-contained) ---
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


# --- iVAE module: hand-rolled JAX MLPs (matches AC's pattern; pure JAX/JIT-able) ---
def _init_mlp(key, dims):
    """dims = [in, h1, h2, ..., out]; ReLU between hidden, none after final."""
    keys = jax.random.split(key, len(dims) - 1)
    layers = []
    for i in range(len(dims) - 1):
        fan_in = dims[i]
        # He-init for ReLU layers, Glorot-like for the final.
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


def _init_ivae(key, x_dim: int, u_dim: int, z_dim: int, enc_h: int, dec_h: int, prior_h: int):
    keys = jax.random.split(key, 5)
    # Encoders: input is (x_pre || u) -> hidden -> (mu, log_var) of size z_dim.
    enc_a = _init_mlp(keys[0], [x_dim + u_dim, enc_h, 2 * z_dim])
    enc_b = _init_mlp(keys[1], [x_dim + u_dim, enc_h, 2 * z_dim])
    # Decoders: z -> hidden -> x_recon.
    dec_a = _init_mlp(keys[2], [z_dim, dec_h, x_dim])
    dec_b = _init_mlp(keys[3], [z_dim, dec_h, x_dim])
    # Prior network: u -> hidden -> (mu_p, log_var_p).
    prior = _init_mlp(keys[4], [u_dim, prior_h, 2 * z_dim])
    return {"enc_a": enc_a, "enc_b": enc_b, "dec_a": dec_a, "dec_b": dec_b, "prior": prior}


def _enc(layers, x, u, lv_min, lv_max, z_dim):
    out = _mlp_forward(layers, jnp.concatenate([x, u], axis=-1))
    mu = out[:, :z_dim]
    lv = jnp.clip(out[:, z_dim:], lv_min, lv_max)
    return mu, lv


def _prior(layers, u, lv_min, lv_max, z_dim):
    out = _mlp_forward(layers, u)
    mu = out[:, :z_dim]
    lv = jnp.clip(out[:, z_dim:], lv_min, lv_max)
    return mu, lv


def _kl_diag_gauss(mu_q, lv_q, mu_p, lv_p):
    # KL(N(mu_q, diag(exp(lv_q))) || N(mu_p, diag(exp(lv_p)))) per dim, then sum.
    var_q = jnp.exp(lv_q)
    var_p = jnp.exp(lv_p)
    return 0.5 * jnp.sum(lv_p - lv_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1.0, axis=-1)


def _ivae_step(params, key, x_a, x_b, u, beta, lambda_c, lv_min, lv_max, z_dim):
    """Returns (loss, aux). Reparameterization on both pathways."""
    mu_a, lv_a = _enc(params["enc_a"], x_a, u, lv_min, lv_max, z_dim)
    mu_b, lv_b = _enc(params["enc_b"], x_b, u, lv_min, lv_max, z_dim)
    mu_p, lv_p = _prior(params["prior"], u, lv_min, lv_max, z_dim)

    k1, k2 = jax.random.split(key)
    eps_a = jax.random.normal(k1, mu_a.shape)
    eps_b = jax.random.normal(k2, mu_b.shape)
    z_a = mu_a + jnp.exp(0.5 * lv_a) * eps_a
    z_b = mu_b + jnp.exp(0.5 * lv_b) * eps_b

    x_a_rec = _mlp_forward(params["dec_a"], z_a)
    x_b_rec = _mlp_forward(params["dec_b"], z_b)
    recon_a = 0.5 * jnp.mean(jnp.sum((x_a_rec - x_a) ** 2, axis=-1))
    recon_b = 0.5 * jnp.mean(jnp.sum((x_b_rec - x_b) ** 2, axis=-1))

    kl_a = jnp.mean(_kl_diag_gauss(mu_a, lv_a, mu_p, lv_p))
    kl_b = jnp.mean(_kl_diag_gauss(mu_b, lv_b, mu_p, lv_p))

    align = jnp.mean(jnp.sum((z_a - z_b) ** 2, axis=-1))

    loss = recon_a + recon_b + beta * (kl_a + kl_b) + lambda_c * align
    aux = {"recon_a": recon_a, "recon_b": recon_b,
           "kl_a": kl_a, "kl_b": kl_b, "align": align,
           "elbo_a": recon_a + beta * kl_a, "elbo_b": recon_b + beta * kl_b}
    return loss, aux


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


def _train_one_seed(z_a_tr, z_b_tr, u_tr, z_a_va, z_b_va, u_va,
                    cfg, beta: float, lambda_c: float, seed: int):
    """Returns (params, history). Mini-batch Adam over 50 epochs, full data each epoch."""
    z_dim = int(cfg["ivae_latent_dim"])
    lv_min = float(cfg["ivae_log_var_min"])
    lv_max = float(cfg["ivae_log_var_max"])
    bs = int(cfg["ivae_batch_size"])
    n_epochs = int(cfg["ivae_epochs"])
    lr = float(cfg["ivae_lr"])

    key = jax.random.PRNGKey(int(seed))
    init_key, key = jax.random.split(key)
    params = _init_ivae(init_key,
                        x_dim=z_a_tr.shape[1], u_dim=u_tr.shape[1], z_dim=z_dim,
                        enc_h=int(cfg["ivae_enc_hidden"]),
                        dec_h=int(cfg["ivae_dec_hidden"]),
                        prior_h=int(cfg["ivae_prior_hidden"]))
    opt = _adam_init(params)

    grad_fn = jax.value_and_grad(_ivae_step, has_aux=True)

    @jax.jit
    def step(params, opt, k, x_a, x_b, u):
        (loss, aux), g = grad_fn(params, k, x_a, x_b, u, beta, lambda_c, lv_min, lv_max, z_dim)
        new, new_opt = _adam_apply(params, opt, g, lr=lr)
        return new, new_opt, loss, aux

    @jax.jit
    def eval_step(params, k, x_a, x_b, u):
        return _ivae_step(params, k, x_a, x_b, u, beta, lambda_c, lv_min, lv_max, z_dim)

    n_train = z_a_tr.shape[0]
    history = []
    rng = np.random.default_rng(int(seed))
    for ep in range(n_epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, bs):
            idx = order[s:s + bs]
            sub_key, key = jax.random.split(key)
            params, opt, _loss, _aux = step(
                params, opt, sub_key,
                jnp.asarray(z_a_tr[idx], dtype=jnp.float32),
                jnp.asarray(z_b_tr[idx], dtype=jnp.float32),
                jnp.asarray(u_tr[idx], dtype=jnp.float32),
            )
        # Per-epoch val ELBO snapshot (no grad).
        sub_key, key = jax.random.split(key)
        val_loss, val_aux = eval_step(
            params, sub_key,
            jnp.asarray(z_a_va, dtype=jnp.float32),
            jnp.asarray(z_b_va, dtype=jnp.float32),
            jnp.asarray(u_va, dtype=jnp.float32),
        )
        history.append({
            "epoch": ep,
            "val_loss": float(val_loss),
            "val_elbo_a": float(val_aux["elbo_a"]),
            "val_elbo_b": float(val_aux["elbo_b"]),
            "val_kl_a": float(val_aux["kl_a"]),
            "val_kl_b": float(val_aux["kl_b"]),
            "val_align": float(val_aux["align"]),
        })

    final = history[-1]
    return params, history, final


def _posterior_mean(params, x, u, lv_min, lv_max, z_dim, side: str):
    enc = params["enc_a"] if side == "a" else params["enc_b"]
    mu, _ = _enc(enc, jnp.asarray(x, dtype=jnp.float32), jnp.asarray(u, dtype=jnp.float32),
                 lv_min, lv_max, z_dim)
    return np.asarray(mu)


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


def _build_aux_u(n_eval_seeds: int, smoke: bool) -> np.ndarray:
    """Reconstruct auxiliary u = [one_hot(class), c] per _eval_set's iteration order:
        for ci in ALL_CLASSES (3): for c in c_grid (3 or 2): for k in k_grid (3 or 2): for s in n_eval_seeds.
    """
    if smoke:
        c_grid = (0.05, 0.10)
        k_grid = (1.0, 2.0)
    else:
        c_grid = (0.05, 0.10, 0.20)
        k_grid = (1.0, 2.0, 5.0)
    rows = []
    n_classes = 3
    for ci in range(n_classes):
        oh = np.zeros(n_classes, dtype=np.float64)
        oh[ci] = 1.0
        for c in c_grid:
            for _k in k_grid:
                for _s in range(n_eval_seeds):
                    rows.append(np.concatenate([oh, [float(c)]]))
    return np.stack(rows).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # 1) Build §5.7 baseline pair at lambda_C=1.0 (same as AC).
    import run as d1_run  # type: ignore[import-not-found]
    z_a_pre, z_b_pre, meta = d1_run.build_baseline_pair(cfg)
    n = z_a_pre.shape[0]
    print(f"[F] baseline pair built. n={n} dimA={meta['dim_A']} dimB={meta['dim_B']} "
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

    # 2) Recover 11-bit basis labels deterministically (same as D3/AC).
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

    # 3) Build auxiliary u = [one_hot(class), c] (4-dim) reconstructed from _eval_set order.
    u = _build_aux_u(n_eval_seeds=n_eval, smoke=bool(cfg["smoke"]))
    assert u.shape[0] == n, f"aux count {u.shape[0]} != latent count {n}"
    assert u.shape[1] == int(cfg["ivae_aux_dim"]), \
        f"aux dim {u.shape[1]} != cfg ivae_aux_dim {cfg['ivae_aux_dim']}"

    # ============================================================
    # F0 CAUSAL ABLATION: zero u immediately, preserving shape+dtype.
    # If F's PASS verdict (cka=0.9311, lower_ratio=0.9906) is causally
    # driven by u, F0 must HONEST_NEGATIVE on identical pipeline.
    # If F0 PASSes, the auxiliary u is NOT the load-bearing mechanism
    # and the paper's central claim must be revised.
    # ============================================================
    u = np.zeros_like(u)

    # 4) 70/30 head split (same head_split_seed=271828 as AC).
    head_train_frac = float(cfg["head_train_fraction"])
    rng = np.random.default_rng(int(cfg["head_split_seed"]))
    perm = rng.permutation(n)
    n_train = int(round(n * head_train_frac))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    z_a_tr_pre, z_a_te_pre = z_a_pre[train_idx], z_a_pre[test_idx]
    z_b_tr_pre, z_b_te_pre = z_b_pre[train_idx], z_b_pre[test_idx]
    u_tr, u_te = u[train_idx], u[test_idx]
    y_tr_bits, y_te_bits = y_bits[train_idx], y_bits[test_idx]

    # 5) Cell sweep: 6 cells (lambda_C in {0,1}, beta in {0.5, 1, 2}) at 1 seed each.
    sweep_seed = int(cfg["ivae_sweep_seed"])
    cells = []
    for lambda_c in cfg["ivae_lambda_c_grid"]:
        for beta in cfg["ivae_beta_grid"]:
            t_cell = time.perf_counter()
            params, _hist, final = _train_one_seed(
                z_a_tr_pre, z_b_tr_pre, u_tr,
                z_a_te_pre, z_b_te_pre, u_te,
                cfg, beta=float(beta), lambda_c=float(lambda_c), seed=sweep_seed,
            )
            cells.append({
                "lambda_c": float(lambda_c),
                "beta": float(beta),
                "val_loss": final["val_loss"],
                "val_elbo_a": final["val_elbo_a"],
                "val_elbo_b": final["val_elbo_b"],
                "val_kl_a": final["val_kl_a"],
                "val_kl_b": final["val_kl_b"],
                "val_align": final["val_align"],
                "wall_s": round(time.perf_counter() - t_cell, 2),
            })
            print(f"[F] sweep cell lambda_c={lambda_c} beta={beta} val_loss={final['val_loss']:.3f} "
                  f"kl_a={final['val_kl_a']:.3f} kl_b={final['val_kl_b']:.3f} "
                  f"wall={cells[-1]['wall_s']:.1f}s", flush=True)

    # Pick cell with lowest val_loss (== best ELBO).
    finite = [c for c in cells if np.isfinite(c["val_loss"])]
    if not finite:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": "All 6 sweep cells produced non-finite val_loss (NaN/Inf).",
            "cells": cells,
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return
    best_cell = min(finite, key=lambda c: c["val_loss"])
    print(f"[F] best cell: lambda_c={best_cell['lambda_c']} beta={best_cell['beta']} "
          f"val_loss={best_cell['val_loss']:.3f}", flush=True)

    # 6) 20 seeds on the best cell (paired-bootstrap CI + best params for PASS_GATE).
    n_seeds = int(cfg["ivae_n_seeds"])
    seed_base = int(cfg["ivae_seed_base"])
    seed_records = []
    best_params = None
    best_final_loss = float("inf")
    for s in range(n_seeds):
        seed = seed_base + s
        params, _hist, final = _train_one_seed(
            z_a_tr_pre, z_b_tr_pre, u_tr,
            z_a_te_pre, z_b_te_pre, u_te,
            cfg, beta=float(best_cell["beta"]), lambda_c=float(best_cell["lambda_c"]),
            seed=seed,
        )
        rec = {"seed": seed, **{k: final[k] for k in
                                ("val_loss", "val_elbo_a", "val_elbo_b",
                                 "val_kl_a", "val_kl_b", "val_align")}}
        seed_records.append(rec)
        if np.isfinite(final["val_loss"]) and final["val_loss"] < best_final_loss:
            best_final_loss = final["val_loss"]
            best_params = params
        if (s % 5) == 0 or s == n_seeds - 1:
            print(f"[F] seed {s+1}/{n_seeds} val_loss={final['val_loss']:.3f}", flush=True)

    if best_params is None:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": f"All {n_seeds} seeds on best cell produced non-finite val_loss.",
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

    # Posterior collapse check on best params.
    z_dim = int(cfg["ivae_latent_dim"])
    mean_kl = (best_cell["val_kl_a"] + best_cell["val_kl_b"]) / 2.0
    mean_kl_per_dim = mean_kl / float(z_dim)
    collapse_warning = bool(mean_kl_per_dim < float(cfg["ivae_collapse_kl_threshold"]))

    # 7) PASS_GATE on deterministic posterior means with best_params.
    lv_min = float(cfg["ivae_log_var_min"])
    lv_max = float(cfg["ivae_log_var_max"])
    z_a_tr = _posterior_mean(best_params, z_a_tr_pre, u_tr, lv_min, lv_max, z_dim, side="a")
    z_a_te = _posterior_mean(best_params, z_a_te_pre, u_te, lv_min, lv_max, z_dim, side="a")
    z_b_tr = _posterior_mean(best_params, z_b_tr_pre, u_tr, lv_min, lv_max, z_dim, side="b")
    z_b_te = _posterior_mean(best_params, z_b_te_pre, u_te, lv_min, lv_max, z_dim, side="b")

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

    # CKA on test split (held-out of head training).
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
    # === Track A audit patch (2026-06-06) ===
    # Compute δ_1 (linear-translator residual) and ε_2 (cross-pathway consistency)
    # for finite-sample identifiability assessment per Track A §A.1.4.
    # CRITICAL: F0 trained on zeroed u; δ_1 measures whether the encoder still
    # recovers the TRUE (un-ablated) u from holdout. Rebuild u_true via _build_aux_u.
    from sklearn.linear_model import LinearRegression as _LR

    u_true_audit = _build_aux_u(n_eval_seeds=n_eval, smoke=bool(cfg["smoke"]))
    u_tr_true = u_true_audit[train_idx]
    u_te_true = u_true_audit[test_idx]

    # ε_2: cross-pathway consistency (mean squared distance between posterior means)
    epsilon_2 = float(np.mean(np.sum((z_a_te - z_b_te) ** 2, axis=1)))

    # δ_1: linear-translator residual on holdout, both pathways → un-ablated u_true
    T_a_audit = _LR().fit(z_a_tr, u_tr_true)
    res_a_audit = u_te_true - T_a_audit.predict(z_a_te)
    delta_1_A_l2 = float(np.sqrt(np.mean(np.sum(res_a_audit ** 2, axis=1))))

    T_b_audit = _LR().fit(z_b_tr, u_tr_true)
    res_b_audit = u_te_true - T_b_audit.predict(z_b_te)
    delta_1_B_l2 = float(np.sqrt(np.mean(np.sum(res_b_audit ** 2, axis=1))))

    # Binding pathway: max of A and B (the worse pathway's u-recovery)
    delta_1 = max(delta_1_A_l2, delta_1_B_l2)

    # α_1: cross-encoder σ_min via LSQ fit z_A → z_B on training fold
    T_cross_audit, _, _, _ = np.linalg.lstsq(z_a_tr, z_b_tr, rcond=None)
    alpha_1_none = float(np.linalg.svd(T_cross_audit, compute_uv=False).min())

    # K^* per ATTACK #3 cancellation: L_{K*} = α_1^{-1} √(1 + ‖Σ‖_op/q)
    APP_B5_Q = 0.02173
    APP_B5_SIGMA_OP = 5.008e-4
    APP_B5_M_R = 1.0
    L_KSTAR_FACTOR = float(np.sqrt(1.0 + APP_B5_SIGMA_OP / APP_B5_Q))  # ≈ 1.01146

    if alpha_1_none > 1e-10:
        k_star = (1.0 / alpha_1_none) * L_KSTAR_FACTOR * (delta_1 + np.sqrt(epsilon_2))
    else:
        k_star = float("inf")

    pass_linear = bool(k_star < APP_B5_M_R)

    # === Track E audit patch (2026-06-06) ===
    # Per-component σ_min(J_θ̂) check for Theorem 4.E (C) dynamical richness.
    # F0 encoder was trained on zeroed u; we probe its Jacobian using un-ablated u_true
    # to test whether the encoder still has full-rank c_other Jacobian without u-shaping.
    n_jacob = min(10000, int(z_a_te_pre.shape[0]))
    rng_audit = np.random.default_rng(int(cfg.get("head_split_seed", 271828)) + 1)
    jacob_idx = rng_audit.choice(int(z_a_te_pre.shape[0]), size=n_jacob, replace=(n_jacob > z_a_te_pre.shape[0]))
    x_base = z_a_te_pre[jacob_idx]
    u_aud = u_te_true[jacob_idx]
    in_dim = int(x_base.shape[1])
    u_dim = int(u_aud.shape[1])
    eps_x = 1e-3
    eps_u = 1e-3

    n_p = 1 + in_dim + u_dim
    x_pert = np.tile(x_base[:, None, :], (1, n_p, 1))
    u_pert = np.tile(u_aud[:, None, :], (1, n_p, 1))
    for i in range(in_dim):
        x_pert[:, 1 + i, i] += eps_x
    for j in range(u_dim):
        u_pert[:, 1 + in_dim + j, j] += eps_u

    x_flat = x_pert.reshape(-1, in_dim)
    u_flat = u_pert.reshape(-1, u_dim)
    z_flat = _posterior_mean(best_params, x_flat, u_flat, lv_min, lv_max, z_dim, side="a")
    z_pert_all = np.asarray(z_flat).reshape(n_jacob, n_p, z_dim)

    z_b = z_pert_all[:, 0, :]
    z_x = z_pert_all[:, 1:1+in_dim, :]
    z_u = z_pert_all[:, 1+in_dim:, :]

    J_full = (z_x - z_b[:, None, :]) / eps_x
    J_full = J_full.transpose(0, 2, 1)
    J_u = (z_u - z_b[:, None, :]) / eps_u
    J_u = J_u.transpose(0, 2, 1)

    svs_full = np.linalg.svd(J_full, compute_uv=False)
    sigma_min_full = svs_full[:, -1]
    sigma_min_full_median = float(np.median(sigma_min_full))
    sigma_min_full_p25 = float(np.percentile(sigma_min_full, 25))
    sigma_min_full_p75 = float(np.percentile(sigma_min_full, 75))

    sigma_min_other_list = []
    for i in range(n_jacob):
        U_u, S_u, _ = np.linalg.svd(J_u[i], full_matrices=True)
        c_other_basis = U_u[:, u_dim:]
        J_restricted = c_other_basis.T @ J_full[i]
        sigma_min_other_list.append(float(np.linalg.svd(J_restricted, compute_uv=False).min()))
    sigma_min_c_other = np.array(sigma_min_other_list)
    sigma_min_other_median = float(np.median(sigma_min_c_other))
    sigma_min_other_p25 = float(np.percentile(sigma_min_c_other, 25))
    sigma_min_other_p75 = float(np.percentile(sigma_min_c_other, 75))

    SIGMA_MIN_THRESHOLD = 1e-2
    pass_richness_full = bool(sigma_min_full_median >= SIGMA_MIN_THRESHOLD)
    pass_richness_other = bool(sigma_min_other_median >= SIGMA_MIN_THRESHOLD)

    # === Track F audit patch (2026-06-06) ===
    # V2 — TV-shifted OOD test via importance reweighting (mirror of F).
    # F0 trained on zeroed u; eval at multiple TV values to test F.1-Gap hypothesis.
    n_classes_audit = 3
    p_tr_audit = np.full(n_classes_audit, 1.0 / n_classes_audit)
    eval_cls_arr = np.array(eval_cls, dtype=np.int64)
    eval_cls_te = eval_cls_arr[test_idx]
    eval_cls_tr = eval_cls_arr[train_idx]
    tv_cells = []
    for tv_cu_audit in [0.0, 0.1, 0.2, 0.4]:
        a_audit = 1.0 / n_classes_audit + tv_cu_audit
        if a_audit > 1.0 - 1e-9:
            a_audit = 0.99
        p_te_audit = np.array([a_audit, (1.0 - a_audit) / 2.0, (1.0 - a_audit) / 2.0])
        actual_tv_audit = 0.5 * float(np.abs(p_te_audit - p_tr_audit).sum())
        weights_te_audit = np.array([p_te_audit[c] / p_tr_audit[c] for c in eval_cls_te])
        sq_diffs_audit = np.sum((z_a_te - z_b_te) ** 2, axis=1)
        epsilon_2_tv = float(np.average(sq_diffs_audit, weights=weights_te_audit))
        T_a_v2 = _LR().fit(z_a_tr, u_tr_true)
        res_a_v2 = u_te_true - T_a_v2.predict(z_a_te)
        sq_res_a_v2 = np.sum(res_a_v2 ** 2, axis=1)
        delta_1_A_tv = float(np.sqrt(np.average(sq_res_a_v2, weights=weights_te_audit)))
        T_b_v2 = _LR().fit(z_b_tr, u_tr_true)
        res_b_v2 = u_te_true - T_b_v2.predict(z_b_te)
        sq_res_b_v2 = np.sum(res_b_v2 ** 2, axis=1)
        delta_1_B_tv = float(np.sqrt(np.average(sq_res_b_v2, weights=weights_te_audit)))
        delta_1_tv = max(delta_1_A_tv, delta_1_B_tv)
        identifiability_err_tv = delta_1_tv + float(np.sqrt(epsilon_2_tv))
        tv_cells.append({
            "tv_cu": tv_cu_audit,
            "actual_tv": actual_tv_audit,
            "p_te_class_0": float(a_audit),
            "delta_1": delta_1_tv,
            "delta_1_A": delta_1_A_tv,
            "delta_1_B": delta_1_B_tv,
            "epsilon_2": epsilon_2_tv,
            "sqrt_epsilon_2": float(np.sqrt(epsilon_2_tv)),
            "identifiability_err": identifiability_err_tv,
            "effective_n_te": float((weights_te_audit.sum() ** 2) / (weights_te_audit ** 2).sum()),
        })

    pass_any = pass_cka or pass_transfer or pass_linear
    verdict = "PASS" if pass_any else "HONEST_NEGATIVE"

    result = {
        "verdict": verdict,
        "method": "F0",
        "ablation_note": "u tensor zeroed out before model input -- F-minus-u causal ablation",
        **meta,
        "n_head_train": int(len(train_idx)),
        "n_head_test": int(len(test_idx)),
        "ivae": {
            "latent_dim": z_dim,
            "aux_dim": int(cfg["ivae_aux_dim"]),
            "enc_hidden": int(cfg["ivae_enc_hidden"]),
            "dec_hidden": int(cfg["ivae_dec_hidden"]),
            "prior_hidden": int(cfg["ivae_prior_hidden"]),
            "log_var_clip": [float(cfg["ivae_log_var_min"]), float(cfg["ivae_log_var_max"])],
            "lr": float(cfg["ivae_lr"]),
            "epochs": int(cfg["ivae_epochs"]),
            "batch_size": int(cfg["ivae_batch_size"]),
            "n_seeds": n_seeds,
            "best_cell": best_cell,
            "best_final_loss": float(best_final_loss),
            "collapse_warning": collapse_warning,
            "mean_kl_per_dim": float(mean_kl_per_dim),
            "collapse_kl_threshold": float(cfg["ivae_collapse_kl_threshold"]),
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
            "delta_1_A_l2": delta_1_A_l2,
            "delta_1_B_l2": delta_1_B_l2,
            "epsilon_2": epsilon_2,
            "alpha_1_none": alpha_1_none,
            "k_star": k_star,
            "k_star_below_M_R": pass_linear,
            "sigma_min_full_median": sigma_min_full_median,
            "sigma_min_full_p25": sigma_min_full_p25,
            "sigma_min_full_p75": sigma_min_full_p75,
            "sigma_min_c_other_median": sigma_min_other_median,
            "sigma_min_c_other_p25": sigma_min_other_p25,
            "sigma_min_c_other_p75": sigma_min_other_p75,
            "n_jacob_points": n_jacob,
            "sigma_min_threshold": SIGMA_MIN_THRESHOLD,
            "pass_richness_full": pass_richness_full,
            "pass_richness_c_other": pass_richness_other,
            "tv_cells": tv_cells,
            "delta_1_note": (
                "TRACK A AUDIT 2026-06-06: F0 trained on zeroed u; δ_1 measured against "
                "un-ablated u_true (rebuilt via _build_aux_u). δ_1 = max(OLS_A, OLS_B) on holdout, "
                "ε_2 = mean ||z_A^μ - z_B^μ||² on holdout, α_1 = σ_min of LSQ z_A→z_B (none-aligned), "
                "K* = α_1^{-1} √(1+‖Σ‖_op/q) (δ_1 + √ε_2) per ATTACK #3 cancellation. "
                "Constants from paper App. B.5 L356 (cross-setup transfer caveat)."
            ),
            "c1_pass": c1_pass,
            "pass_linear": pass_linear,
            "pass_cka": pass_cka,
            "pass_transfer": pass_transfer,
            "pass_any": pass_any,
            "thresholds": dict(cfg["pass_gate"]),
        },
        "deviation_from_spec": (
            "F0 = F-minus-u causal ablation. Pipeline IDENTICAL to F (same data scaffold, "
            "SSL/FT epochs, architecture, sweep, eval). The ONLY change: auxiliary tensor u is "
            "replaced with np.zeros_like(u) immediately after construction (preserving shape "
            "and dtype) so the iVAE prior p(z|u) and encoder q(z|x,u) receive zero auxiliary "
            "signal. Predicted HONEST_NEGATIVE. If F0 PASSes, paper's claim that u is the "
            "load-bearing mechanism is wrong. Inherits F's post-hoc deviation: iVAE on "
            "z_a_pre/z_b_pre (32-d \u00a75.7 baseline latents at lambda_C=1.0) instead of raw "
            "x_A, x_B inputs."
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
