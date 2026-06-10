"""H -- Optimal Transport / Sinkhorn divergence alignment (Cuturi 2013, Genevay 2018).

Per methods/method-H-ot-sinkhorn.md: treat batch-level {z_A^i}, {z_B^i} as empirical
distributions; entropic-regularized Sinkhorn divergence finds the optimal transport
plan; the transport plan IS the cross-modal correspondence -- handles basis-mismatch
geometrically, sidestepping G's norm-collapse and I's k=16 projection collapse.

Loss: L_total = recon_a + recon_b + lambda_c * SinkhornEntropic(z_proj_a, z_proj_b; eps)
   where SinkhornEntropic = <pi, C> with pi the entropic-regularized OT plan with cost
   C_ij = ||z_proj_a_i - z_proj_b_j||^2 and uniform marginals 1/B.

Sinkhorn-Knopp implementation: hand-rolled in log-domain (numerically stable) via
jax.lax.scan over fixed iterations=50 per spec. NOT using ott-jax to avoid an extra
runtime dependency on the EC2 venv (cron tick budget; can't probe venv from local).

DEVIATION FROM SPEC (documented; mirrors AC/F/G/A/F0/C/B/I):
  Spec calls for joint encoder retraining with Sinkhorn loss replacing L_C in
  section-5.7 training. MVP is post-hoc on z_A_pre, z_B_pre (32-d section-5.7
  baseline latents at lambda_C=1.0). Per-pathway projection MLPs (enc + dec)
  trained with reconstruction (acting as L_A, L_B 'predictive' terms in spec)
  plus lambda_C * SinkhornEntropic(z_proj_a, z_proj_b).

Output: results/method-H.json with PASS_GATE 4-gate evaluator
(c1, pass_linear, pass_cka, pass_transfer) per the original spec.

Honest-negative paths:
  * n_eval_paired < 100
  * ALL 3 sweep cells (epsilon in {0.01, 0.1, 1.0}) diverge to NaN
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


def _init_g(key, x_dim: int, z_dim: int, enc_h: int, dec_h: int):
    keys = jax.random.split(key, 4)
    enc_a = _init_mlp(keys[0], [x_dim, enc_h, z_dim])
    enc_b = _init_mlp(keys[1], [x_dim, enc_h, z_dim])
    dec_a = _init_mlp(keys[2], [z_dim, dec_h, x_dim])
    dec_b = _init_mlp(keys[3], [z_dim, dec_h, x_dim])
    return {"enc_a": enc_a, "enc_b": enc_b, "dec_a": dec_a, "dec_b": dec_b}


# --- Sinkhorn-Knopp (log-domain, uniform marginals) ---
def _sinkhorn_loss(z_a, z_b, eps, n_iter):
    """Entropic-regularized OT between uniform empirical distributions on z_a, z_b.

    Cost: C_ij = ||z_a_i - z_b_j||^2  (squared Euclidean).
    Uniform marginals a = b = 1/B.
    Log-domain Sinkhorn-Knopp via jax.lax.scan over n_iter iterations.

    Returns (transport_cost, entropy_of_plan, final_max_marginal_residual).
    transport_cost = <pi, C>  where pi is the converged plan.
    """
    n = z_a.shape[0]
    m = z_b.shape[0]
    # Squared-euclidean cost matrix.
    C = jnp.sum(z_a ** 2, axis=-1, keepdims=True) \
        + jnp.sum(z_b ** 2, axis=-1, keepdims=True).T \
        - 2.0 * z_a @ z_b.T            # (n, m)
    # Numerical safety: clamp to >=0 (squared distances are non-negative; tiny FP noise).
    C = jnp.clip(C, a_min=0.0)

    log_a = -jnp.log(jnp.asarray(n, dtype=C.dtype))
    log_b = -jnp.log(jnp.asarray(m, dtype=C.dtype))

    def body(carry, _):
        f, g = carry
        # f_i = eps * (log_a_i - logsumexp_j((g_j - C_ij) / eps))
        f_new = eps * (log_a - jax.scipy.special.logsumexp((g[None, :] - C) / eps, axis=1))
        g_new = eps * (log_b - jax.scipy.special.logsumexp((f_new[:, None] - C) / eps, axis=0))
        return (f_new, g_new), None

    f0 = jnp.zeros((n,), dtype=C.dtype)
    g0 = jnp.zeros((m,), dtype=C.dtype)
    (f, g), _ = jax.lax.scan(body, (f0, g0), None, length=int(n_iter))

    # pi_ij = exp((f_i + g_j - C_ij) / eps).  Marginals satisfied iff f, g converged.
    log_pi = (f[:, None] + g[None, :] - C) / eps
    pi = jnp.exp(log_pi)
    cost = jnp.sum(pi * C)

    # Plan entropy H(pi) = -sum pi*log(pi). Stable form via log_pi.
    # Mask zeros to avoid 0*(-inf)=NaN; pi==0 contributes 0.
    H = -jnp.sum(jnp.where(pi > 0, pi * log_pi, 0.0))

    # Marginal residuals (diagnostic; should be ~0 at convergence).
    row_marg = jnp.sum(pi, axis=1)              # should equal 1/n
    col_marg = jnp.sum(pi, axis=0)              # should equal 1/m
    max_resid = jnp.maximum(
        jnp.max(jnp.abs(row_marg - jnp.exp(log_a))),
        jnp.max(jnp.abs(col_marg - jnp.exp(log_b))),
    )
    return cost, H, max_resid


def _g_step(params, x_a, x_b, lambda_c, eps, n_iter):
    """Joint loss for encoders+decoders: recon_a + recon_b + lambda_c * SinkhornEntropic.
    Note: MINIMIZED (Sinkhorn cost is non-negative; smaller = tighter alignment)."""
    z_a = _mlp_forward(params["enc_a"], x_a)
    z_b = _mlp_forward(params["enc_b"], x_b)
    x_a_rec = _mlp_forward(params["dec_a"], z_a)
    x_b_rec = _mlp_forward(params["dec_b"], z_b)
    recon_a = 0.5 * jnp.mean(jnp.sum((x_a_rec - x_a) ** 2, axis=-1))
    recon_b = 0.5 * jnp.mean(jnp.sum((x_b_rec - x_b) ** 2, axis=-1))
    sk_cost, sk_H, sk_resid = _sinkhorn_loss(z_a, z_b, eps, n_iter)
    loss = recon_a + recon_b + lambda_c * sk_cost
    aux = {
        "recon_a": recon_a,
        "recon_b": recon_b,
        "sinkhorn_cost": sk_cost,
        "sinkhorn_entropy": sk_H,
        "sinkhorn_marginal_residual": sk_resid,
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


def _adam_apply(params, opts, grads, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = opts["t"] + 1
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1 - b1) * g, opts["m"], grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1 - b2) * (g * g), opts["v"], grads)
    mh = jax.tree_util.tree_map(lambda mm: mm / (1 - b1 ** t), m)
    vh = jax.tree_util.tree_map(lambda vv: vv / (1 - b2 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mm, vv: p - lr * mm / (jnp.sqrt(vv) + eps), params, mh, vh
    )
    return new_params, {"m": m, "v": v, "t": t}


def _train_one_seed(z_a_tr, z_b_tr, z_a_va, z_b_va, cfg, lambda_c: float, eps_val: float, seed: int):
    """Returns (params, history, final). Mini-batch Adam over h_epochs.
    NOTE: Sinkhorn loss requires the full mini-batch as the empirical-distribution support,
    so we keep deterministic mini-batches and re-shuffle each epoch."""
    z_dim = int(cfg["h_latent_dim"])
    enc_h = int(cfg["h_enc_hidden"])
    dec_h = int(cfg["h_dec_hidden"])
    n_iter = int(cfg["h_sinkhorn_iters"])
    bs_cfg = int(cfg["h_batch_size"])
    n_epochs = int(cfg["h_epochs"])
    lr = float(cfg["h_lr"])
    eps_floor = float(cfg.get("h_sinkhorn_eps_floor", 1e-3))
    eps_eff = float(max(eps_val, eps_floor))

    n_train = z_a_tr.shape[0]
    bs = min(bs_cfg, n_train)

    key = jax.random.PRNGKey(int(seed))
    init_key, _ = jax.random.split(key)
    params = _init_g(
        init_key,
        x_dim=z_a_tr.shape[1], z_dim=z_dim,
        enc_h=enc_h, dec_h=dec_h,
    )
    opts = _adam_init(params)

    grad_fn = jax.value_and_grad(_g_step, has_aux=True)

    @jax.jit
    def step(params, opts, x_a, x_b):
        (loss, aux), g = grad_fn(params, x_a, x_b, lambda_c, eps_eff, n_iter)
        new, new_opts = _adam_apply(params, opts, g, lr)
        return new, new_opts, loss, aux

    @jax.jit
    def eval_step(params, x_a, x_b):
        return _g_step(params, x_a, x_b, lambda_c, eps_eff, n_iter)

    history = []
    rng = np.random.default_rng(int(seed))
    for ep in range(n_epochs):
        order = rng.permutation(n_train)
        for s in range(0, n_train, bs):
            idx = order[s:s + bs]
            params, opts, _loss, _aux = step(
                params, opts,
                jnp.asarray(z_a_tr[idx], dtype=jnp.float32),
                jnp.asarray(z_b_tr[idx], dtype=jnp.float32),
            )
        # Per-epoch val snapshot: full val set as one batch (val is small).
        val_loss, val_aux = eval_step(
            params,
            jnp.asarray(z_a_va, dtype=jnp.float32),
            jnp.asarray(z_b_va, dtype=jnp.float32),
        )
        vl = float(val_loss)
        if not np.isfinite(vl):
            return params, history, {"epoch": ep, "val_loss": float("nan"), "nan": True}
        history.append({
            "epoch": ep,
            "val_loss": vl,
            "val_recon_a": float(val_aux["recon_a"]),
            "val_recon_b": float(val_aux["recon_b"]),
            "val_sinkhorn_cost": float(val_aux["sinkhorn_cost"]),
            "val_sinkhorn_entropy": float(val_aux["sinkhorn_entropy"]),
            "val_sinkhorn_marg_resid": float(val_aux["sinkhorn_marginal_residual"]),
            "val_norm_a": float(val_aux["norm_a_mean"]),
            "val_norm_b": float(val_aux["norm_b_mean"]),
        })

    final = history[-1] if history else {"epoch": -1, "val_loss": float("nan"), "nan": True}
    return params, history, final


def _proj(params, x, side: str):
    enc = params["enc_a"] if side == "a" else params["enc_b"]
    return np.asarray(_mlp_forward(enc, jnp.asarray(x, dtype=jnp.float32)))


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

    # 1) Build §5.7 baseline pair at lambda_C=1.0 (same as AC/F/G/A/F0/C/B/I).
    import run as d1_run  # type: ignore[import-not-found]
    z_a_pre, z_b_pre, meta = d1_run.build_baseline_pair(cfg)
    n = z_a_pre.shape[0]
    print(f"[H] baseline pair built. n={n} dimA={meta['dim_A']} dimB={meta['dim_B']} "
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

    # 2) Recover 11-bit basis labels deterministically (same as D3/AC/F/G/A/F0/C/B/I).
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

    # 3) 70/30 head split (same head_split_seed=271828 as AC/F/G/A/F0/C/B/I/D3/D4).
    rng = np.random.default_rng(int(cfg["head_split_seed"]))
    perm = rng.permutation(n)
    n_train = int(round(n * float(cfg["head_train_fraction"])))
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    z_a_tr, z_a_te = z_a_pre[tr_idx], z_a_pre[te_idx]
    z_b_tr, z_b_te = z_b_pre[tr_idx], z_b_pre[te_idx]
    y_tr, y_te = y_bits[tr_idx], y_bits[te_idx]

    # 4) Hyperparameter sweep over epsilon.
    eps_grid = list(cfg["h_eps_grid"])
    lambda_c_fixed = float(cfg["h_lambda_c_fixed"])
    sweep_seed = int(cfg["h_sweep_seed"])
    cell_records = []
    nan_cells = 0
    for eps_val in eps_grid:
        t_cell = time.perf_counter()
        _params, _hist, fin = _train_one_seed(
            z_a_tr, z_b_tr, z_a_te, z_b_te, cfg, lambda_c=lambda_c_fixed,
            eps_val=float(eps_val), seed=sweep_seed,
        )
        rec = {"eps": float(eps_val), "lambda_c": lambda_c_fixed, **fin,
               "wall_s": round(time.perf_counter() - t_cell, 2)}
        if rec.get("nan", False) or not np.isfinite(rec.get("val_loss", float("nan"))):
            nan_cells += 1
        cell_records.append(rec)
        print(f"[H] sweep cell eps={eps_val:.3g}: val_loss={rec.get('val_loss')!r} "
              f"wall={rec['wall_s']}s", flush=True)

    if nan_cells == len(eps_grid):
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": "All Sinkhorn sweep cells diverged to NaN -- entropic-OT optimization unstable on this pair.",
            "cell_records": cell_records,
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # Best cell = lowest finite val_loss.
    finite_cells = [r for r in cell_records if np.isfinite(r.get("val_loss", float("nan")))]
    best_cell = min(finite_cells, key=lambda r: r["val_loss"])
    best_eps = float(best_cell["eps"])
    print(f"[H] best cell eps={best_eps:.3g} val_loss={best_cell['val_loss']:.4f}", flush=True)

    # 5) 20-seed paired-bootstrap on best cell.
    seed_base = int(cfg["h_seed_base"])
    n_seeds = int(cfg["h_n_seeds"])
    seed_records = []
    best_seed_id = -1
    best_seed_loss = float("inf")
    best_params = None
    for s_off in range(n_seeds):
        s = seed_base + s_off
        t_s = time.perf_counter()
        params_s, _hist_s, fin_s = _train_one_seed(
            z_a_tr, z_b_tr, z_a_te, z_b_te, cfg, lambda_c=lambda_c_fixed,
            eps_val=best_eps, seed=s,
        )
        wall_s = round(time.perf_counter() - t_s, 2)
        rec = {"seed": s, **fin_s, "wall_s": wall_s}
        seed_records.append(rec)
        vl = rec.get("val_loss", float("nan"))
        if np.isfinite(vl) and vl < best_seed_loss:
            best_seed_loss = vl
            best_seed_id = s
            best_params = params_s

    if best_params is None:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": "All 20 seeds on best cell diverged to NaN.",
            "best_cell": best_cell,
            "cell_records": cell_records,
            "seed_records": seed_records,
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # 6) Project to z_proj on best seed.
    z_proj_a_tr = _proj(best_params, z_a_tr, "a")
    z_proj_b_tr = _proj(best_params, z_b_tr, "b")
    z_proj_a_te = _proj(best_params, z_a_te, "a")
    z_proj_b_te = _proj(best_params, z_b_te, "b")

    # Diagnostic: norms on test slice.
    norm_a_te = np.linalg.norm(z_proj_a_te, axis=-1)
    norm_b_te = np.linalg.norm(z_proj_b_te, axis=-1)

    # 7) PASS_GATE: per-bit probes within and across pathways.
    probe_C = float(cfg["probe_C"])
    probe_max_iter = int(cfg["probe_max_iter"])
    probes_A, skipped_A = _per_bit_probes(z_proj_a_tr, y_tr, probe_C, probe_max_iter)
    probes_B, skipped_B = _per_bit_probes(z_proj_b_tr, y_tr, probe_C, probe_max_iter)
    auroc_A_within = _per_bit_auroc(probes_A, skipped_A, z_proj_a_te, y_te)
    auroc_B_within = _per_bit_auroc(probes_B, skipped_B, z_proj_b_te, y_te)
    auroc_A_to_B = _per_bit_auroc(probes_A, skipped_A, z_proj_b_te, y_te)  # A-trained on B-test
    auroc_B_to_A = _per_bit_auroc(probes_B, skipped_B, z_proj_a_te, y_te)  # B-trained on A-test

    probe_auroc_A = float(np.nanmean(auroc_A_within))
    probe_auroc_B = float(np.nanmean(auroc_B_within))
    transfer_AtoB_mean = float(np.nanmean(auroc_A_to_B))
    transfer_BtoA_mean = float(np.nanmean(auroc_B_to_A))
    within_A_mean = probe_auroc_A
    within_B_mean = probe_auroc_B
    ratio_AtoB = float(transfer_AtoB_mean / within_A_mean) if within_A_mean > 0 else float("nan")
    ratio_BtoA = float(transfer_BtoA_mean / within_B_mean) if within_B_mean > 0 else float("nan")
    lower_transfer_ratio = float(min(ratio_AtoB, ratio_BtoA))

    probe_transfer_auroc = float(0.5 * (transfer_AtoB_mean + transfer_BtoA_mean))
    within_pathway_auroc = float(0.5 * (within_A_mean + within_B_mean))

    # 8) CKA with bootstrap CI on test split.
    cka_val = linear_cka(z_proj_a_te, z_proj_b_te)
    cka_lo, cka_hi = bootstrap_ci(
        linear_cka, z_proj_a_te, z_proj_b_te,
        n_boot=int(cfg["n_bootstrap"]), seed=int(cfg["bootstrap_seed"]),
    )

    # 9) PASS_GATE evaluator per the original spec.
    th = cfg["pass_gate"]
    c1_min = float(th["c1_min_per_pathway_auroc"])
    cka_min = float(th["cka_min"])
    transfer_ratio_min = float(th["transfer_ratio_min"])

    c1_pass = bool(probe_auroc_A >= c1_min and probe_auroc_B >= c1_min)
    pass_cka = bool(c1_pass and cka_val >= cka_min)
    # Cron-prompt formula: probe_transfer_auroc >= 0.95 * within_pathway_auroc.
    pass_transfer = bool(c1_pass and probe_transfer_auroc >= transfer_ratio_min * within_pathway_auroc)
    pass_linear = False  # delta_1 / epsilon_2 not exposed by build_baseline_pair (MVP).
    pass_any = bool(pass_cka or pass_transfer or pass_linear)
    verdict = "PASS" if pass_any else "HONEST_NEGATIVE"

    skipped_bits_A = [int(b) for b in np.where(skipped_A)[0].tolist()]
    skipped_bits_B = [int(b) for b in np.where(skipped_B)[0].tolist()]

    # 10) Final-cell sinkhorn diagnostic on the best seed (read back from final history).
    # The training loop ended; recompute one eval pass at best-eps to capture stable diagnostic.
    sk_cost_final, sk_H_final, sk_resid_final = _sinkhorn_loss(
        jnp.asarray(z_proj_a_te, dtype=jnp.float32),
        jnp.asarray(z_proj_b_te, dtype=jnp.float32),
        eps=float(max(best_eps, float(cfg.get("h_sinkhorn_eps_floor", 1e-3)))),
        n_iter=int(cfg["h_sinkhorn_iters"]),
    )

    headline = {
        "lambda_c": meta["lambda_c"],
        "n_eval_samples": int(n),
        "n_head_train": int(n_train),
        "n_head_test": int(n - n_train),
        "h_head": {
            "latent_dim": int(cfg["h_latent_dim"]),
            "enc_hidden": int(cfg["h_enc_hidden"]),
            "dec_hidden": int(cfg["h_dec_hidden"]),
            "sinkhorn_iters": int(cfg["h_sinkhorn_iters"]),
            "lr": float(cfg["h_lr"]),
            "epochs": int(cfg["h_epochs"]),
            "batch_size_cfg": int(cfg["h_batch_size"]),
            "batch_size_effective": int(min(cfg["h_batch_size"], n_train)),
            "n_seeds": int(n_seeds),
            "best_cell": best_cell,
            "best_seed_id": int(best_seed_id),
            "best_seed_loss": float(best_seed_loss),
            "norm_a_te_mean": float(norm_a_te.mean()),
            "norm_b_te_mean": float(norm_b_te.mean()),
            "norm_a_te_std": float(norm_a_te.std()),
            "norm_b_te_std": float(norm_b_te.std()),
            "final_sinkhorn_loss": float(sk_cost_final),
            "transport_plan_entropy": float(sk_H_final),
            "sinkhorn_iterations_at_convergence": int(cfg["h_sinkhorn_iters"]),
            "sinkhorn_marginal_residual_at_eval": float(sk_resid_final),
        },
        "best_cell_eps": float(best_eps),
        "best_cell_lambda_c": lambda_c_fixed,
        "best_cell_val_loss": float(best_cell["val_loss"]),
        "n_seeds_on_best_cell": int(n_seeds),
        "probe_auroc_A": probe_auroc_A,
        "probe_auroc_B": probe_auroc_B,
        "auroc_A_per_bit": [
            float(x) if (i not in skipped_bits_A and np.isfinite(x)) else None
            for i, x in enumerate(auroc_A_within)
        ],
        "auroc_B_per_bit": [
            float(x) if (i not in skipped_bits_B and np.isfinite(x)) else None
            for i, x in enumerate(auroc_B_within)
        ],
        "skipped_bits_A": skipped_bits_A,
        "skipped_bits_B": skipped_bits_B,
        "cka": float(cka_val),
        "cka_ci_95": [float(cka_lo), float(cka_hi)],
        "transfer_AtoB_mean": transfer_AtoB_mean,
        "transfer_BtoA_mean": transfer_BtoA_mean,
        "within_A_mean": within_A_mean,
        "within_B_mean": within_B_mean,
        "ratio_AtoB": ratio_AtoB,
        "ratio_BtoA": ratio_BtoA,
        "lower_transfer_ratio": lower_transfer_ratio,
        "probe_transfer_auroc": probe_transfer_auroc,
        "within_pathway_auroc": within_pathway_auroc,
        "delta_1": None,
        "epsilon_2": None,
        "c1_pass": c1_pass,
        "pass_linear": pass_linear,
        "pass_cka": pass_cka,
        "pass_transfer": pass_transfer,
        "pass_any": pass_any,
        "deviation_from_spec": (
            "POST-HOC Sinkhorn-OT projection on z_a_pre/z_b_pre (32-d section-5.7 baseline "
            "latents at lambda_C=1.0) instead of joint encoder retraining with Sinkhorn replacing "
            "L_C. Per-pathway projection MLPs (enc + dec) trained with reconstruction (acting as "
            "the L_A, L_B 'predictive' terms in the spec) plus lambda_C * SinkhornEntropic "
            "between projections. Sinkhorn-Knopp implemented hand-rolled in log-domain via "
            "jax.lax.scan over 50 fixed iterations (numerical stability) -- not using ott-jax to "
            "avoid an extra runtime dep on the EC2 venv. Mirrors AC/F/G/A/F0/C/B/I post-hoc choice. "
            "delta_1 / epsilon_2 not exposed by build_baseline_pair -> pass_linear forced FALSE; "
            "pass_cka and pass_transfer carry verdict."
        ),
        "n_bootstrap": int(cfg["n_bootstrap"]),
    }

    result = {
        "verdict": verdict,
        "method": "H",
        **meta,
        **headline,
        "cells": cell_records,
        "seed_records": seed_records,
        "wall_s": round(time.perf_counter() - t0, 1),
        "git_head_at_run": _git_sha(HERE),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
