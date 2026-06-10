"""D4 -- Translator-Network Capacity probe.

Trains translator MLPs T: z_A -> z_B at depths {1, 2, 4, 8} with hidden=64. The
minimal depth at which val_mse_rel < 0.10 quantifies how structurally different
the two latents are:

  min_capacity = 1   -> basis-ambiguity (linear fix is enough)
  min_capacity = 2   -> mild nonlinear-parametrization (iVAE class likely sufficient)
  min_capacity = 4   -> heavy nonlinear; iVAE may not suffice, consider shared-bottleneck
  min_capacity = 8 or none-converging -> information-mismatch (no map exists)

Both directions A->B and B->A are trained; asymmetry is informative per spec.

Reuses D1's build_baseline_pair() to keep the (zA, zB) substrate byte-equivalent
across diagnostics. Uses pure JAX (no optax) with hand-rolled Adam, matching the
repo's existing pattern (W4-learned-B/w4_2_ssl_probe.py:_adam_apply).

This is a diagnostic, not a PASS gate -- output feeds the decide phase's tree.

Honest-negative outcomes:
  * n_eval_paired < 100: insufficient samples (matches D1/D2/D3 floor).
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

# JAX is required on this machine (verified working in D1/D2/D3 launches via
# $REPO/.venv). Importing at module scope to avoid the lazy-import
# pattern; this also forces JAX device discovery to happen once.
import jax
import jax.numpy as jnp
import jax.nn as jnn


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


# ---------------------------------------------------------------------------
# JAX MLP translator + Adam (pure JAX, no optax/flax dep)
# ---------------------------------------------------------------------------

def _init_mlp(key, dim_in: int, dim_out: int, depth: int, hidden: int):
    """Return a list[dict{W,b}] of length depth.

    depth=1 -> [Linear(dim_in -> dim_out)].
    depth=d>=2 -> [Linear(dim_in,hidden), ..., Linear(hidden,dim_out)] with
                  ReLU between every consecutive pair (applied in forward, not stored).
    He-init for ReLU layers; Glorot-like for the final linear.
    """
    keys = jax.random.split(key, depth)
    layers = []
    cur = dim_in
    for i in range(depth - 1):
        # He-normal scale = sqrt(2/fan_in) for ReLU
        W = jax.random.normal(keys[i], (cur, hidden)) * jnp.sqrt(2.0 / cur)
        b = jnp.zeros((hidden,))
        layers.append({"W": W, "b": b})
        cur = hidden
    # Final linear (no ReLU after) -- Glorot scale = sqrt(1/fan_in)
    W = jax.random.normal(keys[-1], (cur, dim_out)) * jnp.sqrt(1.0 / cur)
    b = jnp.zeros((dim_out,))
    layers.append({"W": W, "b": b})
    return layers


def _count_params(layers) -> int:
    n = 0
    for layer in layers:
        n += int(layer["W"].size + layer["b"].size)
    return n


def _forward(layers, x):
    """Apply MLP. ReLU between every consecutive pair (none after last)."""
    h = x
    L = len(layers)
    for i, layer in enumerate(layers):
        h = h @ layer["W"] + layer["b"]
        if i < L - 1:
            h = jnn.relu(h)
    return h


def _mse_loss(layers, x, y_unit):
    """Per-row mean squared error against unit-normalized target."""
    pred = _forward(layers, x)
    return jnp.mean(jnp.sum((pred - y_unit) ** 2, axis=1))


def _val_mse_rel(layers, x, y_unit):
    """val_mse_rel = mean per-row ||T(x) - y_unit||^2 / mean per-row ||y_unit||^2.

    Since y_unit is row-L2-normalized, the denominator is exactly 1.0; we still
    compute it to be explicit and to handle the rare zero-row case.
    """
    pred = _forward(layers, x)
    num = jnp.mean(jnp.sum((pred - y_unit) ** 2, axis=1))
    den = jnp.mean(jnp.sum(y_unit ** 2, axis=1))
    return num / (den + 1e-12)


def _adam_init(params):
    """Return Adam state matching params pytree."""
    return {
        "m": jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params),
        "v": jax.tree_util.tree_map(lambda p: jnp.zeros_like(p), params),
        "t": 0,
    }


def _adam_apply(params, opt_state, grads, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    """Apply Adam given pre-computed grads. Mirrors W4-learned-B/w4_2_ssl_probe.py."""
    t = opt_state["t"] + 1
    m = jax.tree_util.tree_map(lambda mm, g: b1 * mm + (1 - b1) * g, opt_state["m"], grads)
    v = jax.tree_util.tree_map(lambda vv, g: b2 * vv + (1 - b2) * (g * g), opt_state["v"], grads)
    m_hat = jax.tree_util.tree_map(lambda mm: mm / (1 - b1 ** t), m)
    v_hat = jax.tree_util.tree_map(lambda vv: vv / (1 - b2 ** t), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps),
        params, m_hat, v_hat,
    )
    return new_params, {"m": m, "v": v, "t": t}


def _train_one_translator(
    z_in_train: np.ndarray, z_out_train_unit: np.ndarray,
    z_in_val: np.ndarray, z_out_val_unit: np.ndarray,
    depth: int, hidden: int, lr: float, n_steps: int, patience: int,
    seed: int,
) -> dict:
    """Train one translator depth/direction/seed via full-batch Adam with early stop on val.

    Returns: {final_val_mse_rel, params, wall_s, train_mse_history_compact}.
    """
    key = jax.random.PRNGKey(int(seed))
    layers = _init_mlp(key, dim_in=z_in_train.shape[1], dim_out=z_out_train_unit.shape[1],
                       depth=int(depth), hidden=int(hidden))
    n_params = _count_params(layers)

    x_tr = jnp.asarray(z_in_train, dtype=jnp.float32)
    y_tr = jnp.asarray(z_out_train_unit, dtype=jnp.float32)
    x_va = jnp.asarray(z_in_val, dtype=jnp.float32)
    y_va = jnp.asarray(z_out_val_unit, dtype=jnp.float32)

    opt_state = _adam_init(layers)
    grad_fn = jax.value_and_grad(_mse_loss)

    @jax.jit
    def step(layers, opt_state, x, y, lr):
        loss, grads = grad_fn(layers, x, y)
        new_layers, new_opt = _adam_apply(layers, opt_state, grads, lr=lr)
        return new_layers, new_opt, loss

    @jax.jit
    def eval_val(layers, x, y):
        return _val_mse_rel(layers, x, y)

    best_val = float(eval_val(layers, x_va, y_va))
    best_step = 0
    no_improve = 0
    train_curve = []
    t0 = time.perf_counter()
    for s in range(int(n_steps)):
        layers, opt_state, train_loss = step(layers, opt_state, x_tr, y_tr, jnp.float32(lr))
        if (s + 1) % 25 == 0 or s == 0:
            cur_val = float(eval_val(layers, x_va, y_va))
            train_curve.append({"step": s + 1, "train_mse": float(train_loss), "val_mse_rel": cur_val})
            if cur_val < best_val - 1e-6:
                best_val = cur_val
                best_step = s + 1
                no_improve = 0
            else:
                no_improve += 25
            if no_improve >= int(patience):
                break

    final_val = float(eval_val(layers, x_va, y_va))
    final_val_used = float(min(final_val, best_val))  # report best-seen on val

    return {
        "final_val_mse_rel": final_val_used,
        "best_step": int(best_step),
        "params": int(n_params),
        "wall_s": round(time.perf_counter() - t0, 2),
        "train_curve": train_curve,
    }


# ---------------------------------------------------------------------------
# Per-depth aggregation across seeds + diagnosis
# ---------------------------------------------------------------------------

def _depth_summary(per_seed_results: list, pass_threshold: float) -> dict:
    vals = [r["final_val_mse_rel"] for r in per_seed_results]
    params = per_seed_results[0]["params"]
    wall_s = float(np.sum([r["wall_s"] for r in per_seed_results]))
    val_mse_rel_mean = float(np.mean(vals))
    val_mse_rel_std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {
        "val_mse_rel_mean": val_mse_rel_mean,
        "val_mse_rel_std": val_mse_rel_std,
        "val_mse_rel_min": float(np.min(vals)),
        "val_mse_rel_max": float(np.max(vals)),
        "pass": bool(val_mse_rel_mean < float(pass_threshold)),
        "params": int(params),
        "wall_s": wall_s,
        "n_seeds": int(len(vals)),
        "per_seed_val_mse_rel": [float(v) for v in vals],
    }


def _min_capacity(per_depth: list, pass_threshold: float) -> int | None:
    """Return smallest depth with mean val_mse_rel < threshold, or None."""
    for entry in per_depth:
        if entry["val_mse_rel_mean"] < float(pass_threshold):
            return int(entry["depth"])
    return None


def _diagnosis(min_cap_AtoB, min_cap_BtoA) -> tuple[str, list, bool]:
    """Per spec table:
        min_depth=1     -> basis-ambiguity        (top_3 = [A, AC, G])
        min_depth=2     -> mild-nonlinear         (top_3 = [F, G, I])
        min_depth=4     -> heavy-nonlinear        (top_3 = [F, G, AC])
        min_depth=8 or None -> information-mismatch (top_3 = [F, C, AC])

    Asymmetry: if directions disagree by >=2 levels (e.g. one is 2, other is 8 or None),
    flag asymmetric=True and route by the WORSE direction (more conservative).
    """
    def _level(mc):
        if mc is None:
            return 4
        return {1: 0, 2: 1, 4: 2, 8: 3}.get(int(mc), 3)

    la = _level(min_cap_AtoB)
    lb = _level(min_cap_BtoA)
    asymmetric = abs(la - lb) >= 2
    worse_level = max(la, lb)

    table = [
        ("basis-ambiguity", ["A", "AC", "G"]),
        ("mild-nonlinear", ["F", "G", "I"]),
        ("heavy-nonlinear", ["F", "G", "AC"]),
        ("information-mismatch", ["F", "C", "AC"]),
        ("information-mismatch", ["F", "C", "AC"]),  # level 4 (None)
    ]
    diagnosis, top_3 = table[worse_level]
    return diagnosis, top_3, asymmetric


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
            "reason": f"n_eval_samples={n} < 100; translator-capacity estimate is uninformative.",
            **meta,
            "wall_s": round(time.perf_counter() - t0, 1),
            "git_head_at_run": _git_sha(HERE),
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_path.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    # 80/20 train/val split.
    test_frac = float(cfg.get("test_fraction", 0.20))
    split_rng = np.random.default_rng(int(cfg.get("split_seed", 271828)))
    perm = split_rng.permutation(n)
    n_val = max(1, int(round(n * test_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    z_a_tr, z_a_va = z_a[train_idx], z_a[val_idx]
    z_b_tr, z_b_va = z_b[train_idx], z_b[val_idx]

    # Cosine-normalized targets per spec ("z_B / ||z_B||" removes scale freedom).
    def _row_unit(M: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        # Avoid division by zero (extremely unlikely on trained encoders but be safe)
        norms = np.where(norms > 1e-12, norms, 1.0)
        return (M / norms).astype(np.float32)

    z_a_tr_u, z_a_va_u = _row_unit(z_a_tr), _row_unit(z_a_va)
    z_b_tr_u, z_b_va_u = _row_unit(z_b_tr), _row_unit(z_b_va)

    print(
        f"[D4] computing translator-capacity on n={n} (train={len(train_idx)}, val={n_val}); "
        f"dimA={meta['dim_A']}, dimB={meta['dim_B']}",
        flush=True,
    )

    depths = [int(d) for d in cfg.get("depths", [1, 2, 4, 8])]
    hidden = int(cfg.get("hidden", 64))
    lr = float(cfg.get("translator_lr", 1e-3))
    n_steps = int(cfg.get("translator_steps", 600))
    patience = int(cfg.get("early_stop_patience", 80))
    n_seeds = int(cfg.get("translator_n_seeds", 5))
    seed_base = int(cfg.get("translator_seed_base", 4242))
    pass_threshold = float(cfg.get("pass_threshold_val_mse_rel", 0.10))

    directions = {}
    for direction_name, (z_in_tr, z_out_tr_u, z_in_va, z_out_va_u) in {
        "A_to_B": (z_a_tr, z_b_tr_u, z_a_va, z_b_va_u),
        "B_to_A": (z_b_tr, z_a_tr_u, z_b_va, z_a_va_u),
    }.items():
        per_depth = []
        for d in depths:
            seed_results = []
            for s in range(n_seeds):
                seed = seed_base + 1000 * d + s + (0 if direction_name == "A_to_B" else 99)
                r = _train_one_translator(
                    z_in_train=z_in_tr.astype(np.float32),
                    z_out_train_unit=z_out_tr_u,
                    z_in_val=z_in_va.astype(np.float32),
                    z_out_val_unit=z_out_va_u,
                    depth=d, hidden=hidden, lr=lr,
                    n_steps=n_steps, patience=patience, seed=seed,
                )
                seed_results.append(r)
            summary = _depth_summary(seed_results, pass_threshold=pass_threshold)
            summary["depth"] = int(d)
            summary["per_seed_records"] = seed_results
            per_depth.append(summary)
            print(
                f"[D4] {direction_name} depth={d}: "
                f"val_mse_rel mean={summary['val_mse_rel_mean']:.4f} "
                f"std={summary['val_mse_rel_std']:.4f} "
                f"params={summary['params']} pass={summary['pass']} wall_s={summary['wall_s']:.1f}",
                flush=True,
            )
        directions[direction_name] = per_depth

    min_cap_AtoB = _min_capacity(directions["A_to_B"], pass_threshold=pass_threshold)
    min_cap_BtoA = _min_capacity(directions["B_to_A"], pass_threshold=pass_threshold)
    diagnosis, top_3, asymmetric = _diagnosis(min_cap_AtoB, min_cap_BtoA)

    result = {
        "verdict": "diagnostic_complete",
        **meta,
        "n_train_samples_translator": int(len(train_idx)),
        "n_val_samples_translator": int(n_val),
        "depths": depths,
        "hidden": hidden,
        "translator_lr": lr,
        "translator_steps": n_steps,
        "translator_n_seeds": n_seeds,
        "early_stop_patience": patience,
        "pass_threshold_val_mse_rel": pass_threshold,
        "directions": directions,
        "min_capacity_A_to_B": min_cap_AtoB,
        "min_capacity_B_to_A": min_cap_BtoA,
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
