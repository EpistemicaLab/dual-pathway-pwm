"""D1 -- Centered Kernel Alignment (CKA) diagnostic.

Reads hyperparameters from config.json in the same dir. Runs the §5.7 dual-pathway
baseline at lambda_C=0.0 (pre-alignment condition: no L_C term), extracts paired
(zA, zB) latents on a held-out eval set, and computes linear + RBF CKA with 95%
bootstrap CIs. Also caches dimensions, sample size, and git HEAD SHA.

CKA is basis-invariant. Together with D3 (linear-probe-transfer), it disambiguates:
  - high CKA AND failing transfer  -> nonlinear-parametrization mechanism
  - low CKA                        -> information-mismatch mechanism
  - high CKA AND passing transfer  -> basis-ambiguity mechanism (suggests CCA bug
                                      in §5.7 evaluation rather than a real failure)

This is a diagnostic, not a PASS gate -- numbers feed the `decide` phase's tree.
The only honest-negative outcome is n_eval_paired < 100 (insufficient samples).
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


HERE = Path(__file__).resolve().parent
E1_DIR = HERE.parents[1] / "E1-numerical-tightness"
W4_DIR = HERE.parents[1] / "W4-learned-B"
W5_DIR = HERE.parents[1] / "W5-C13-LC-ablation"
for p in (E1_DIR, W4_DIR, W5_DIR):
    sys.path.insert(0, str(p))


def _git_sha(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception as e:  # noqa: BLE001
        return f"<unavailable: {e}>"


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered linear-kernel CKA (Kornblith+ ICML 2019, eq. 4)."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    num = float(np.linalg.norm(X.T @ Y, ord="fro") ** 2)
    den = float(np.linalg.norm(X.T @ X, ord="fro") * np.linalg.norm(Y.T @ Y, ord="fro"))
    if den <= 0.0:
        return float("nan")
    return num / den


def _gram_rbf(M: np.ndarray, sigma: float) -> np.ndarray:
    d2 = ((M[:, None, :] - M[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * sigma * sigma))


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma_x: float | None = None,
            sigma_y: float | None = None) -> float:
    """RBF-kernel CKA with median-heuristic bandwidth (per matrix)."""
    if sigma_x is None:
        d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        d_off = d[d > 0]
        sigma_x = float(np.median(np.sqrt(d_off))) if d_off.size else 1.0
    if sigma_y is None:
        d = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        d_off = d[d > 0]
        sigma_y = float(np.median(np.sqrt(d_off))) if d_off.size else 1.0
    Kx = _gram_rbf(X, sigma_x)
    Ky = _gram_rbf(Y, sigma_y)
    n = Kx.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kx_c = H @ Kx @ H
    Ky_c = H @ Ky @ H
    num = float((Kx_c * Ky_c).sum())
    den = float(np.sqrt(max((Kx_c * Kx_c).sum(), 1e-30)
                        * max((Ky_c * Ky_c).sum(), 1e-30)))
    return num / den


def bootstrap_ci(X: np.ndarray, Y: np.ndarray, fn, n_boot: int, seed: int,
                 alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = fn(X[idx], Y[idx])
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def build_baseline_pair(cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train the §5.7 baseline at lambda_C=0.0 and return paired (zA, zB) eval matrices.

    Mirrors the in-process pipeline in E1's measure_bound_real_lc.main(): SSL warm-start
    on B, init A as a measurement-MLP, joint-train at lambda_C=0.0 (no L_C term -- this
    is the pre-alignment baseline), then evaluate on the held-out grid produced by
    _eval_set(seed_base=50000) and apply each encoder.
    """
    import jax  # noqa: F401  (forces JAX init for device discovery in logs)
    import jax.numpy as jnp
    import measure_bound_real_lc as e1

    e1.NORMALIZE = False  # raw L2 (matches §5.7 baseline)
    if cfg["smoke"]:
        c_grid, k_grid = (0.05, 0.10), (1.0, 2.0)
        ssl_ep, ft_ep = 5, 3
        nspc, n_eval, bs = 4, 12, 8
    else:
        c_grid, k_grid = (0.05, 0.10, 0.20), (1.0, 2.0, 5.0)
        ssl_ep = int(cfg["ssl_epochs"])
        ft_ep = int(cfg["ft_epochs"])
        nspc = int(cfg["n_seeds_per_config"])
        n_eval = int(cfg["n_eval_seeds"])
        bs = 32

    print(f"[D1] device={jnp.zeros(1).device}  ssl_ep={ssl_ep} ft_ep={ft_ep} "
          f"nspc={nspc} n_eval={n_eval} lambda_c=0.0", flush=True)

    samples = e1.build_multi_class_training_set(
        c_grid, k_grid, e1.TRAIN_CLASSES, e1.SIGMA,
        n_seeds_per_config=nspc, seed_offset=10_000)
    va, vb, _ = e1.render_two_view_dataset(samples)
    ssl_enc, _ = e1.train_ssl_encoder(va, vb, n_epochs=ssl_ep, batch_size=bs, lr=1e-3, seed=42)

    imgs = np.stack([e1.render_single_view(x, xd) for (x, xd, _m) in samples])
    raw_feats = np.stack(
        [e1.trajectory_features(x, xd) for (x, xd, _m) in samples]
    ).astype(np.float32)
    feat_mu = raw_feats.mean(axis=0)
    feat_sd = raw_feats.std(axis=0) + 1e-6
    feats = (raw_feats - feat_mu) / feat_sd
    t_bce = np.stack([m for (_x, _xd, m) in samples])
    t_lc = np.stack([e1.lasso_target_from_trajectory(x, xd, e1.SIGMA)
                     for (x, xd, _m) in samples])

    probe_a0 = e1.init_probe(__import__("jax").random.PRNGKey(0),
                             latent_dim=e1.LATENT_DIM, out_dim=11)
    probe_b0 = e1.init_probe(__import__("jax").random.PRNGKey(1),
                             latent_dim=e1.LATENT_DIM, out_dim=11)
    enc_a0 = e1.init_mlp(__import__("jax").random.PRNGKey(7), in_dim=feats.shape[1])

    print("[D1] joint-training pre-alignment baseline (lambda_C=0.0)…", flush=True)
    enc_a, enc_b, hist = e1.joint_dual_finetune_with_lc(
        enc_a0, ssl_enc, probe_a0, probe_b0,
        feats, imgs, t_bce, t_lc, lambda_c=float(cfg["lambda_c"]),
        n_epochs=ft_ep, batch_size=bs, lr=float(cfg["ft_lr"]))

    # Held-out eval pass — same as measure_one(), seed_base=50_000
    eval_imgs, eval_feats, eval_cls = e1._eval_set(c_grid, k_grid, n_eval, seed_base=50_000)
    eval_feats = (eval_feats - feat_mu) / feat_sd
    z_a = np.asarray(e1.mlp_forward(enc_a, jnp.asarray(eval_feats)))
    z_b = np.asarray(e1.encoder_forward(enc_b, jnp.asarray(eval_imgs, dtype=jnp.float32)))

    meta = {
        "n_train_samples": int(imgs.shape[0]),
        "n_eval_samples": int(z_a.shape[0]),
        "dim_A": int(z_a.shape[1]),
        "dim_B": int(z_b.shape[1]),
        "ssl_epochs": ssl_ep,
        "ft_epochs": ft_ep,
        "ft_lr": float(cfg["ft_lr"]),
        "lambda_c": float(cfg["lambda_c"]),
        "final_train_lc_loss": float(hist[-1]["lc"]),
        "final_train_bce_loss": float(hist[-1]["bce"]),
        "final_train_mse_loss": float(hist[-1]["mse"]),
        "n_class_id": int((eval_cls != e1.ALL_CLASSES.index("pendulum")).sum()),
        "n_class_ood": int((eval_cls == e1.ALL_CLASSES.index("pendulum")).sum()),
    }
    return z_a, z_b, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    z_a, z_b, meta = build_baseline_pair(cfg)
    n = z_a.shape[0]

    if n < 100:
        result = {
            "verdict": "HONEST_NEGATIVE",
            "reason": f"n_eval_samples={n} < 100; CKA estimate is uninformative.",
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    print(f"[D1] computing CKA on n={n}, dimA={meta['dim_A']}, dimB={meta['dim_B']}…",
          flush=True)
    lin = linear_cka(z_a, z_b)
    rbf = rbf_cka(z_a, z_b)
    n_boot = int(cfg.get("n_bootstrap", 1000))
    seed = int(cfg.get("bootstrap_seed", 12345))
    lin_lo, lin_hi = bootstrap_ci(z_a, z_b, linear_cka, n_boot=n_boot, seed=seed)
    rbf_lo, rbf_hi = bootstrap_ci(z_a, z_b, rbf_cka, n_boot=n_boot, seed=seed + 1)

    # Spec interpretation table -> decision-tree input
    if lin >= 0.90:
        interp = "strong_shared_structure_basis_aligned"
    elif lin >= 0.70:
        interp = "shared_structure_not_basis_aligned__nonlinear_parametrization"
    elif lin >= 0.40:
        interp = "partial_overlap__mixed"
    else:
        interp = "information_mismatch"

    result = {
        "verdict": "diagnostic_complete",
        **meta,
        "linear_cka": lin,
        "linear_cka_ci_95": [lin_lo, lin_hi],
        "rbf_cka": rbf,
        "rbf_cka_ci_95": [rbf_lo, rbf_hi],
        "n_bootstrap": n_boot,
        "interpretation": interp,
        "wall_s": round(time.perf_counter() - t0, 1),
        "git_head_at_run": _git_sha(HERE),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
