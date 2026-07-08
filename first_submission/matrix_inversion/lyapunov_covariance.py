#!/usr/bin/env python3
"""
Deterministic Lyapunov (covariance) dynamics for thermodynamic computing / Mpemba pre-thermalization.

This script:
  - Builds SPD matrices J from either a fixed-spectrum ensemble or a Wishart ensemble.
  - Constructs the equilibrium covariance Sigma_eq = kBT * J^{-1}.
  - Constructs an "optimized" initialization Sigma0 that pre-thermalizes the k slowest modes.
  - Measures the absolute Frobenius error E(t) = ||Sigma(t) - Sigma_eq||_F via a closed-form
    eigenbasis expression (no ODE integration required).
  - Computes the first-passage time t_eps: first time E(t) <= epsilon (bisection).
  - Plots measured speedup S_eps = t_eps(k=0) / t_eps(k=k_speed) vs dimension d.
  - Overlays a theoretical prediction based on eigenvalues including the epsilon-dependent
    log correction:
        S_eps^th ≈ (alpha_k/alpha_1) * log(kBT/(eps*alpha_1)) / log(kBT/(eps*alpha_k))
    where alpha_1 is the smallest eigenvalue and alpha_k is the k-th smallest eigenvalue
    (k = number of pre-thermalized modes; the slowest remaining mode is alpha_k in
    1-indexed notation).

Usage
-----
    python lyapunov_covariance.py [options]

Run ``python lyapunov_covariance.py --help`` for a full list of options.

Output
------
All numerical data underlying the figures is saved as compressed NumPy archives
(.npz) plus a params.json file in a subdirectory of ``data/`` whose name encodes
the full set of run parameters, making results reproducible.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams.update(
    {
        "text.usetex": False,       # use matplotlib's mathtext
        "mathtext.fontset": "stix", # best-looking math font
        "font.family": "STIXGeneral",
        "axes.labelsize": 25,
        "font.size": 25,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }
)


# ---------------------------------------------------------------------------
# Matrix ensembles
# ---------------------------------------------------------------------------

def make_J_fixed(d: int, alpha_min: float, alpha_max: float, rng: np.random.Generator) -> np.ndarray:
    """
    Fixed-spectrum SPD ensemble.

    Eigenvalues are linearly spaced::

        spectrum = linspace(alpha_min, alpha_max * d, d)

    Eigenvectors are Haar-random (via QR of a Gaussian matrix).

    Parameters
    ----------
    d : int
        Matrix dimension.
    alpha_min : float
        Smallest eigenvalue (> 0).
    alpha_max : float
        Spacing parameter; largest eigenvalue is ``alpha_max * d`` (> 0).
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    J : ndarray of shape (d, d)
        Symmetric positive-definite matrix.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    if alpha_min <= 0 or alpha_max <= 0:
        raise ValueError("alpha_min and alpha_max must be > 0")

    spectrum = np.linspace(alpha_min, alpha_max * d, d)

    # Haar-random orthogonal matrix via QR decomposition.
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # Fix sign ambiguity so the distribution is exactly Haar.
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs

    return Q @ np.diag(spectrum) @ Q.T


def make_J_wishart(d: int, m: int, rng: np.random.Generator, ridge: float) -> np.ndarray:
    """
    Wishart SPD ensemble.

    ``J = (X^T X) / m + ridge * I``, with ``X ~ N(0,1)^{m x d}``.

    Parameters
    ----------
    d : int
        Matrix dimension.
    m : int
        Number of samples (must be >= d for a well-conditioned matrix before ridge).
    rng : np.random.Generator
        NumPy random generator.
    ridge : float
        Ridge regularisation (>= 0).

    Returns
    -------
    J : ndarray of shape (d, d)
        Symmetric positive-definite matrix.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    if m < d:
        raise ValueError("m must be >= d for a well-conditioned Wishart SPD (before ridge).")
    if ridge < 0:
        raise ValueError("ridge must be >= 0")

    X = rng.normal(size=(m, d))
    J = (X.T @ X) / float(m)
    if ridge > 0:
        J = J + ridge * np.eye(d)
    return J


# ---------------------------------------------------------------------------
# Equilibrium covariance and initial condition
# ---------------------------------------------------------------------------

def sigma_eq(J: np.ndarray, kBT: float) -> np.ndarray:
    """Return the equilibrium covariance ``Sigma_eq = kBT * J^{-1}``."""
    d = J.shape[0]
    return kBT * np.linalg.solve(J, np.eye(d))


def sigma0_optimized(J: np.ndarray, k: int, kBT: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-thermalize the k slowest modes (k smallest eigenvalues) of J.

    In the eigenbasis of J the initial covariance is::

        Sigma0 = sum_{i=1}^k (kBT / alpha_i) u_i u_i^T

    so that these modes already sit at their equilibrium values while all
    remaining modes start at zero.

    Parameters
    ----------
    J : ndarray of shape (d, d)
        Symmetric positive-definite matrix.
    k : int
        Number of slow modes to pre-thermalize (0 <= k <= d).
    kBT : float
        Thermal energy scale.

    Returns
    -------
    Sigma0 : ndarray of shape (d, d)
        Initial covariance matrix.
    evals_k : ndarray of shape (k,)
        The k smallest eigenvalues of J (ascending order).
    """
    d = J.shape[0]
    if k < 0 or k > d:
        raise ValueError("k must satisfy 0 <= k <= d")
    if k == 0:
        return np.zeros((d, d)), np.array([], dtype=float)

    evals, evecs = np.linalg.eigh(J)
    evals_k = evals[:k]
    evecs_k = evecs[:, :k]

    Sigma0 = evecs_k @ np.diag(kBT / evals_k) @ evecs_k.T
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)  # enforce exact symmetry
    return Sigma0, evals_k


# ---------------------------------------------------------------------------
# RK4 integrator (kept for validation; production code uses the closed form)
# ---------------------------------------------------------------------------

def _lyapunov_rhs(S: np.ndarray, J: np.ndarray, mu: float, kBT: float, I: np.ndarray) -> np.ndarray:
    """Return ``dS/dt = -mu (J S + S J) + 2 mu kBT I``."""
    return -mu * (J @ S + S @ J) + 2.0 * mu * kBT * I


def _choose_dt(J: np.ndarray, mu: float, safety: float = 0.05) -> float:
    """Heuristic RK4 step size: ``safety / (2 mu alpha_max)``."""
    alpha_max = float(np.linalg.eigvalsh(J).max())
    return safety / (2.0 * mu * alpha_max)


def integrate_lyapunov_rk4(
    J: np.ndarray,
    Sigma0: np.ndarray,
    mu: float,
    kBT: float,
    tmax: float,
    dt: float,
    Sigma_eq_mat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate the Lyapunov ODE with explicit RK4 and record absolute Frobenius error.

    This function is provided for validation against the closed-form
    ``error_curve_from_eigs``. The two should agree to within numerical
    precision for well-chosen ``dt``.

    Parameters
    ----------
    J : ndarray of shape (d, d)
        SPD coupling matrix.
    Sigma0 : ndarray of shape (d, d)
        Initial covariance.
    mu : float
        Relaxation rate.
    kBT : float
        Thermal energy scale.
    tmax : float
        Integration end time.
    dt : float
        Time step (must be positive).
    Sigma_eq_mat : ndarray of shape (d, d)
        Equilibrium covariance, used to compute the error at each step.

    Returns
    -------
    t : ndarray of shape (n_steps + 1,)
    E : ndarray of shape (n_steps + 1,)
        Absolute Frobenius error at each time point.
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    n_steps = int(np.ceil(tmax / dt))
    d = J.shape[0]
    I = np.eye(d)

    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    S = 0.5 * (Sigma0 + Sigma0.T)

    err = np.empty(n_steps + 1, dtype=float)
    err[0] = np.linalg.norm(S - Sigma_eq_mat, ord="fro")

    for n in range(n_steps):
        k1 = _lyapunov_rhs(S, J, mu, kBT, I)
        k2 = _lyapunov_rhs(S + 0.5 * dt * k1, J, mu, kBT, I)
        k3 = _lyapunov_rhs(S + 0.5 * dt * k2, J, mu, kBT, I)
        k4 = _lyapunov_rhs(S + dt * k3, J, mu, kBT, I)
        S = S + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        S = 0.5 * (S + S.T)
        err[n + 1] = np.linalg.norm(S - Sigma_eq_mat, ord="fro")

    return t, err


# ---------------------------------------------------------------------------
# Closed-form error in the eigenbasis of J
# ---------------------------------------------------------------------------

def E_of_t_from_eigs(evals: np.ndarray, k_thermalized: int, mu: float, kBT: float, t: float) -> float:
    """
    Closed-form absolute Frobenius error ``||Sigma(t) - Sigma_eq||_F``.

    For the Lyapunov dynamics with symmetric SPD J the covariance in the
    eigenbasis evolves element-wise. With the initializations used here
    (k=0: Sigma0=0; k>0: pre-thermalize the k slowest modes) the error is::

        E(t) = kBT * sqrt( sum_{i > k} lambda_i^{-2} * exp(-4 mu lambda_i t) )

    This avoids integrating a d×d matrix ODE and is orders of magnitude faster.

    Parameters
    ----------
    evals : ndarray
        Full sorted eigenvalue array of J (ascending).
    k_thermalized : int
        Number of slow modes already at equilibrium.
    mu, kBT, t : float
        Physical parameters and evaluation time.

    Returns
    -------
    float
        Absolute Frobenius error at time t.
    """
    if k_thermalized < 0 or k_thermalized > evals.size:
        raise ValueError("k_thermalized out of range")
    tail = evals[k_thermalized:]
    if tail.size == 0:
        return 0.0
    s = np.sum((1.0 / (tail * tail)) * np.exp(-4.0 * mu * tail * t))
    return float(kBT * np.sqrt(s))


def error_curve_from_eigs(
    evals: np.ndarray,
    k_thermalized: int,
    mu: float,
    kBT: float,
    tmax: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(t, E(t))`` on a uniform grid using the closed-form expression.

    Parameters
    ----------
    evals : ndarray
        Full sorted eigenvalue array of J (ascending).
    k_thermalized : int
        Number of slow modes already at equilibrium.
    mu, kBT, tmax : float
        Physical parameters and end time.
    n_points : int
        Number of time points (>= 2).

    Returns
    -------
    t : ndarray of shape (n_points,)
    E : ndarray of shape (n_points,)
    """
    t = np.linspace(0.0, float(tmax), int(max(2, n_points)))
    tail = evals[k_thermalized:]
    if tail.size == 0:
        return t, np.zeros_like(t)
    expo = np.exp(-4.0 * mu * tail[:, None] * t[None, :])
    s = np.sum((1.0 / (tail * tail))[:, None] * expo, axis=0)
    E = kBT * np.sqrt(s)
    return t, E


# ---------------------------------------------------------------------------
# First-passage time
# ---------------------------------------------------------------------------

def first_passage_time_from_eigs(
    evals: np.ndarray,
    k_thermalized: int,
    epsilon: float,
    mu: float,
    kBT: float,
    tmax: float,
    rtol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """
    Smallest t in [0, tmax] with E(t) <= epsilon, found by monotone bisection.

    Returns ``np.inf`` if there is no crossing within ``[0, tmax]``.

    Parameters
    ----------
    evals : ndarray
        Full sorted eigenvalue array of J (ascending).
    k_thermalized : int
        Number of slow modes already at equilibrium.
    epsilon : float
        Target error threshold (> 0).
    mu, kBT, tmax : float
        Physical parameters and search horizon.
    rtol : float
        Relative tolerance for the bisection interval width.
    max_iter : int
        Maximum number of bisection iterations.

    Returns
    -------
    float
        First-passage time, or ``np.inf`` if not reached within tmax.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    E0 = E_of_t_from_eigs(evals, k_thermalized, mu, kBT, 0.0)
    if E0 <= epsilon:
        return 0.0
    Emax = E_of_t_from_eigs(evals, k_thermalized, mu, kBT, float(tmax))
    if Emax > epsilon:
        return float("inf")

    lo, hi = 0.0, float(tmax)
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        if E_of_t_from_eigs(evals, k_thermalized, mu, kBT, mid) <= epsilon:
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= rtol * max(1.0, hi):
            break
    return float(hi)


# ---------------------------------------------------------------------------
# Seeding helpers
# ---------------------------------------------------------------------------

def _seed_for_trial(*, ensemble: str, d: int, seed: int, tr: int) -> int:
    """Deterministic per-trial seed, consistent across the whole script."""
    ens = ensemble.lower()
    if ens == "fixed":
        return int(seed) + 17 * int(d) + 123 + int(tr)
    if ens == "wishart":
        return int(seed) + 1000 * int(d) + int(tr)
    raise ValueError("ensemble must be 'fixed' or 'wishart'")


# ---------------------------------------------------------------------------
# Building and caching disorder realizations
# ---------------------------------------------------------------------------

def build_J_trials(
    *,
    ensemble: str,
    d: int,
    seed: int,
    alpha_min: float,
    alpha_max: float,
    fixed_trials: int,
    wishart_m_factor: float,
    wishart_ridge: float,
    wishart_trials: int,
) -> list[np.ndarray]:
    """
    Build all disorder realizations for a given (ensemble, d).

    Uses ``_seed_for_trial`` so that results are exactly reproducible and
    consistent with the per-trial eigenvalue cache used in ``main``.

    Returns
    -------
    list of ndarray
        One SPD matrix per trial.
    """
    ens = ensemble.lower()
    if ens not in ("fixed", "wishart"):
        raise ValueError("ensemble must be 'fixed' or 'wishart'")

    Js: list[np.ndarray] = []
    if ens == "fixed":
        for tr in range(int(max(1, fixed_trials))):
            rng_tr = np.random.default_rng(_seed_for_trial(ensemble="fixed", d=d, seed=seed, tr=tr))
            Js.append(make_J_fixed(int(d), alpha_min=alpha_min, alpha_max=alpha_max, rng=rng_tr))
    else:
        m = int(np.ceil(wishart_m_factor * int(d)))
        for tr in range(int(max(1, wishart_trials))):
            rng_tr = np.random.default_rng(_seed_for_trial(ensemble="wishart", d=d, seed=seed, tr=tr))
            Js.append(make_J_wishart(int(d), m=m, rng=rng_tr, ridge=wishart_ridge))
    return Js


def eigs_trials_from_Js(Js: list[np.ndarray]) -> list[np.ndarray]:
    """Return sorted eigenvalue arrays for each matrix in ``Js``."""
    return [np.linalg.eigvalsh(J) for J in Js]


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Lyapunov/covariance Mpemba speedup (deterministic).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dimensions
    ap.add_argument("--d_min", type=int, default=30, help="Smallest dimension to sweep.")
    ap.add_argument("--d_max", type=int, default=35, help="Largest dimension to sweep.")
    ap.add_argument("--d_step", type=int, default=5, help="Step size for dimension sweep.")

    # Physics
    ap.add_argument("--mu", type=float, default=1.0, help="Relaxation rate mu.")
    ap.add_argument("--kBT", type=float, default=1.0, help="Thermal energy kBT.")

    # Error thresholds (can be set independently for the two ensembles)
    ap.add_argument(
        "--epsilon_fixed_list",
        type=str,
        default="1e-2, 1e-3, 1e-4",
        help="Comma-separated epsilon thresholds for the fixed-spectrum inset.",
    )
    ap.add_argument(
        "--epsilon_wishart_list",
        type=str,
        default="1e-1, 1e-2, 1e-3",
        help="Comma-separated epsilon thresholds for the Wishart inset.",
    )

    # Mpemba parameter
    ap.add_argument("--k_speed", type=int, default=10, help="Number of slow modes pre-thermalized.")

    # Trace plot
    ap.add_argument(
        "--k_list",
        type=str,
        default="0,1,5,10",
        help="Comma-separated k values (pre-thermalized modes) to show in E(t) traces.",
    )
    ap.add_argument("--d_trace", type=int, default=30, help="Dimension used for E(t) trace plots.")

    # Fixed-spectrum parameters
    ap.add_argument("--alpha_min", type=float, default=0.5, help="Smallest eigenvalue.")
    ap.add_argument(
        "--alpha_max",
        type=float,
        default=0.5,
        help="Spacing parameter; spectrum = linspace(alpha_min, alpha_max * d, d).",
    )
    ap.add_argument(
        "--fixed_trials",
        type=int,
        default=1,
        help="Number of Haar-random eigenvector draws to average for the fixed-spectrum ensemble.",
    )

    # Wishart parameters
    ap.add_argument("--wishart_m_factor", type=float, default=2, help="Sample count m = ceil(m_factor * d).")
    ap.add_argument("--wishart_ridge", type=float, default=0.0, help="Ridge regularisation added to Wishart matrix.")
    ap.add_argument("--wishart_trials", type=int, default=1, help="Number of Wishart realizations to average.")

    # Integration / time horizon
    ap.add_argument("--tmax_fixed", type=float, default=15.0, help="Time horizon for fixed-spectrum runs.")
    ap.add_argument("--tmax_wishart", type=float, default=70.0, help="Time horizon for Wishart runs.")
    ap.add_argument(
        "--dt",
        type=float,
        default=-1.0,
        help="Time-grid spacing for output curves. If <= 0, use 2500 points over tmax.",
    )
    ap.add_argument("--dt_safety", type=float, default=0.05, help="Safety factor for heuristic dt (unused by closed form).")

    # RNG
    ap.add_argument("--seed", type=int, default=0, help="Base random seed.")

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Output folder: encode all run parameters so results are reproducible.
    # ------------------------------------------------------------------
    def _clean(x: str) -> str:
        """Return a filesystem-safe token."""
        return (
            str(x)
            .replace(" ", "")
            .replace(",", "-")
            .replace("/", "_")
            .replace("\\", "_")
            .replace("{", "")
            .replace("}", "")
        )

    param_tokens = [
        f"seed={args.seed}",
        f"mu={args.mu}",
        f"kBT={args.kBT}",
        f"dmin={args.d_min}",
        f"dmax={args.d_max}",
        f"dstep={args.d_step}",
        f"dtrace={args.d_trace}",
        f"klist={_clean(args.k_list)}",
        f"kspeed={args.k_speed}",
        f"epsF={_clean(args.epsilon_fixed_list)}",
        f"epsW={_clean(args.epsilon_wishart_list)}",
        f"amin={args.alpha_min}",
        f"amax={args.alpha_max}",
        f"mfac={args.wishart_m_factor}",
        f"ridge={args.wishart_ridge}",
        f"tr_wish={args.wishart_trials}",
        f"tmaxF={args.tmax_fixed}",
        f"tmaxW={args.tmax_wishart}",
        f"dt={args.dt}",
        f"dtsafe={args.dt_safety}",
    ]
    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    out_dir = os.path.join(out_base, "__".join(_clean(t) for t in param_tokens))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # Parse list arguments
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    eps_fixed_list = [float(x) for x in args.epsilon_fixed_list.split(",") if x.strip()]
    eps_wishart_list = [float(x) for x in args.epsilon_wishart_list.split(",") if x.strip()]
    d_list = np.arange(args.d_min, args.d_max + 1, args.d_step, dtype=int)

    # ------------------------------------------------------------------
    # Cache disorder realizations / spectra.
    # Building J and diagonalising it is the expensive step; caching
    # ensures we do it only once per (ensemble, d) across all k and eps.
    # ------------------------------------------------------------------
    evals_cache: dict[tuple[str, int], list[np.ndarray]] = {}

    def _get_evals_trials(ensemble: str, d: int) -> list[np.ndarray]:
        key = (ensemble.lower(), int(d))
        if key not in evals_cache:
            Js = build_J_trials(
                ensemble=ensemble,
                d=int(d),
                seed=int(args.seed),
                alpha_min=float(args.alpha_min),
                alpha_max=float(args.alpha_max),
                fixed_trials=int(args.fixed_trials),
                wishart_m_factor=float(args.wishart_m_factor),
                wishart_ridge=float(args.wishart_ridge),
                wishart_trials=int(args.wishart_trials),
            )
            evals_cache[key] = eigs_trials_from_Js(Js)
        return evals_cache[key]

    def _mean_teps(evals_trials: list[np.ndarray], k_th: int, eps: float, tmax: float) -> float:
        """Average first-passage time over disorder trials (np.inf if any trial misses)."""
        t_list = [
            first_passage_time_from_eigs(e, int(k_th), float(eps), float(args.mu), float(args.kBT), float(tmax))
            for e in evals_trials
        ]
        return float(np.mean(np.asarray(t_list, dtype=float)))

    # ------------------------------------------------------------------
    # Save the representative matrices used for the d_trace plots.
    # These are trial-0 realisations from the cache so they match the curves.
    # ------------------------------------------------------------------
    evals_trials_fixed_trace = _get_evals_trials("fixed", int(args.d_trace))
    evals_trials_wish_trace = _get_evals_trials("wishart", int(args.d_trace))

    # Reconstruct trial-0 matrices for archiving (cheap: one diagonalisation each).
    J_fix_save = build_J_trials(
        ensemble="fixed",
        d=int(args.d_trace),
        seed=int(args.seed),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        fixed_trials=1,
        wishart_m_factor=float(args.wishart_m_factor),
        wishart_ridge=float(args.wishart_ridge),
        wishart_trials=1,
    )[0]
    m_trace = int(np.ceil(args.wishart_m_factor * args.d_trace))
    J_w_save = build_J_trials(
        ensemble="wishart",
        d=int(args.d_trace),
        seed=int(args.seed),
        alpha_min=float(args.alpha_min),
        alpha_max=float(args.alpha_max),
        fixed_trials=1,
        wishart_m_factor=float(args.wishart_m_factor),
        wishart_ridge=float(args.wishart_ridge),
        wishart_trials=1,
    )[0]

    np.savez_compressed(
        os.path.join(out_dir, "trace_matrices.npz"),
        d_trace=args.d_trace,
        m_trace=m_trace,
        J_fixed=J_fix_save,
        J_wishart=J_w_save,
        lam_fixed=np.linalg.eigvalsh(J_fix_save),
        lam_wishart=np.linalg.eigvalsh(J_w_save),
    )

    # Shared time-grid resolution
    n_points_fixed = int(np.ceil(float(args.tmax_fixed) / float(args.dt))) + 1 if args.dt > 0 else 2500
    n_points_wish = int(np.ceil(float(args.tmax_wishart) / float(args.dt))) + 1 if args.dt > 0 else 2500

    # ================================================================
    # Figure 1: fixed-spectrum — E(t) traces (main) + speedup vs d (inset)
    # ================================================================
    fig_fix, ax_fix = plt.subplots(figsize=(8, 4.5))

    fig1_traces: dict[int, dict] = {}
    t_f = np.linspace(0.0, float(args.tmax_fixed), n_points_fixed)

    for k_th in k_list:
        E_trials = [
            error_curve_from_eigs(
                evals=evals,
                k_thermalized=int(k_th),
                mu=float(args.mu),
                kBT=float(args.kBT),
                tmax=float(args.tmax_fixed),
                n_points=len(t_f),
            )[1]
            for evals in evals_trials_fixed_trace
        ]
        E_stack = np.vstack(E_trials)
        E_med = np.median(E_stack, axis=0)
        E_min = np.min(E_stack, axis=0)
        E_max = np.max(E_stack, axis=0)

        fig1_traces[int(k_th)] = {
            "t": t_f,
            "t_scaled": t_f * (args.mu * args.alpha_min),
            "E": E_med,
            "E_min": E_min,
            "E_max": E_max,
        }
        (ln,) = ax_fix.semilogy(t_f * (args.mu * args.alpha_min), E_med, linestyle="-", label=rf"$k={k_th}$")
        ax_fix.fill_between(
            t_f * (args.mu * args.alpha_min),
            E_min,
            E_max,
            alpha=0.18,
            linewidth=0.0,
            color=ln.get_color(),
        )

    ax_fix.set_xlabel(r"time $t \, [\mu^{-1}\lambda_1^{-1}]$")
    ax_fix.set_ylabel(r"$E(t)=\|\Sigma(t)-\Sigma_{\rm eq}\|_F$")
    ax_fix.grid(True, alpha=0.3)
    ax_fix.legend(loc="upper left", frameon=False, ncol=2)
    ax_fix.set_ylim([1e-10, 1e5])

    # Inset: speedup vs d
    axins_fix = ax_fix.inset_axes([0.58, 0.52, 0.38, 0.43])

    linestyles = ["solid", "dashed", "dotted"]
    fig1_inset_tk0, fig1_inset_tk, fig1_inset_S = [], [], []
    for i, eps in enumerate(eps_fixed_list):
        t_k0 = np.array(
            [_mean_teps(_get_evals_trials("fixed", int(d)), 0, eps, args.tmax_fixed) for d in d_list],
            dtype=float,
        )
        t_k = np.array(
            [_mean_teps(_get_evals_trials("fixed", int(d)), args.k_speed, eps, args.tmax_fixed) for d in d_list],
            dtype=float,
        )
        S = np.divide(
            t_k0, t_k,
            out=np.full_like(t_k0, np.nan),
            where=np.isfinite(t_k0) & np.isfinite(t_k) & (t_k > 0),
        )
        fig1_inset_tk0.append(t_k0)
        fig1_inset_tk.append(t_k)
        fig1_inset_S.append(S)

        mask = np.isfinite(S)
        exp_label = int(np.floor(np.log10(eps) + 1e-12))
        axins_fix.plot(
            d_list[mask], S[mask],
            color="#d62728",
            marker="o",
            linewidth=1.2,
            label=rf"$\epsilon=10^{{{exp_label}}}$",
            linestyle=linestyles[i % len(linestyles)],
        )

    # Theoretical spectral ratio R = <lambda_{k+1} / lambda_1>
    R_fix = np.array(
        [
            np.nanmean(
                [
                    (ev[args.k_speed] / ev[0])
                    if (len(ev) > args.k_speed and ev[0] > 0)
                    else np.nan
                    for ev in _get_evals_trials("fixed", int(d))
                ]
            )
            for d in d_list
        ],
        dtype=float,
    )
    axins_fix.plot(d_list, R_fix, linewidth=1.0, color="black", label=rf"$\lambda_{{{args.k_speed + 1}}}/\lambda_1$")

    axins_fix.grid(True, alpha=0.25)
    axins_fix.tick_params(labelsize=8)
    axins_fix.set_xlabel("$d$")
    axins_fix.set_ylabel(r"$\mathcal{S}_\epsilon$")
    axins_fix.legend(loc="center right", bbox_to_anchor=(-0.2, 0.5), frameon=False)
    axins_fix.set_title(rf"Speedup, $k={args.k_speed}$", fontsize=15)

    # Save Figure 1 data
    np.savez_compressed(
        os.path.join(out_dir, "figure1_fixed_data.npz"),
        **{f"trace_k{kk}_t": vv["t"] for kk, vv in fig1_traces.items()},
        **{f"trace_k{kk}_t_scaled": vv["t_scaled"] for kk, vv in fig1_traces.items()},
        **{f"trace_k{kk}_E": vv["E"] for kk, vv in fig1_traces.items()},
        **{f"trace_k{kk}_E_min": vv["E_min"] for kk, vv in fig1_traces.items()},
        **{f"trace_k{kk}_E_max": vv["E_max"] for kk, vv in fig1_traces.items()},
        d_list=d_list,
        eps_list=np.array(eps_fixed_list, dtype=float),
        k_speed=args.k_speed,
        t_eps_k0=np.array(fig1_inset_tk0, dtype=float),
        t_eps_k=np.array(fig1_inset_tk, dtype=float),
        S_eps=np.array(fig1_inset_S, dtype=float),
        R_theory=R_fix,
    )

    plt.title("Haar")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Haar.pdf"))
    plt.show()

    # ================================================================
    # Figure 2: Wishart — E(t) traces (main) + speedup vs d (inset)
    # ================================================================
    fig_w, ax_w = plt.subplots(figsize=(8, 4.5))

    fig2_traces: dict[int, dict] = {}
    lam1_w_trace = float(np.median([float(e[0]) for e in evals_trials_wish_trace]))
    t_wt = np.linspace(0.0, float(args.tmax_wishart), n_points_wish)

    for k_th in k_list:
        E_trials = [
            error_curve_from_eigs(
                evals=evals,
                k_thermalized=int(k_th),
                mu=float(args.mu),
                kBT=float(args.kBT),
                tmax=float(args.tmax_wishart),
                n_points=len(t_wt),
            )[1]
            for evals in evals_trials_wish_trace
        ]
        E_stack = np.vstack(E_trials)
        E_med = np.median(E_stack, axis=0)
        E_min = np.min(E_stack, axis=0)
        E_max = np.max(E_stack, axis=0)

        fig2_traces[int(k_th)] = {
            "t": t_wt,
            "t_scaled": t_wt * (args.mu * lam1_w_trace),
            "E": E_med,
            "E_min": E_min,
            "E_max": E_max,
        }
        (ln,) = ax_w.semilogy(t_wt * (args.mu * lam1_w_trace), E_med, linestyle="-", label=rf"$k={k_th}$")
        ax_w.fill_between(
            t_wt * (args.mu * lam1_w_trace),
            E_min,
            E_max,
            alpha=0.18,
            linewidth=0.0,
            color=ln.get_color(),
        )

    ax_w.set_xlabel(r"time $t \,[\mu^{-1}\lambda_1^{-1}]$")
    ax_w.set_ylabel(r"$E(t)=\|\Sigma(t)-\Sigma_{\rm eq}\|_F$")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="upper left", frameon=False, ncol=2)
    ax_w.set_ylim([1e-10, 1e5])

    # Inset: speedup vs d
    axins_w = ax_w.inset_axes([0.58, 0.52, 0.38, 0.43])

    fig2_inset_tk0, fig2_inset_tk, fig2_inset_S = [], [], []
    for i, eps in enumerate(eps_wishart_list):
        t_k0 = np.array(
            [_mean_teps(_get_evals_trials("wishart", int(d)), 0, eps, args.tmax_wishart) for d in d_list],
            dtype=float,
        )
        t_k = np.array(
            [_mean_teps(_get_evals_trials("wishart", int(d)), args.k_speed, eps, args.tmax_wishart) for d in d_list],
            dtype=float,
        )
        S = np.divide(
            t_k0, t_k,
            out=np.full_like(t_k0, np.nan),
            where=np.isfinite(t_k0) & np.isfinite(t_k) & (t_k > 0),
        )
        fig2_inset_tk0.append(t_k0)
        fig2_inset_tk.append(t_k)
        fig2_inset_S.append(S)

        mask = np.isfinite(S)
        exp_label = int(np.floor(np.log10(eps) + 1e-12))
        axins_w.plot(
            d_list[mask], S[mask],
            color="#d62728",
            marker="o",
            linewidth=1.2,
            label=rf"$\epsilon=10^{{{exp_label}}}$",
            linestyle=linestyles[i % len(linestyles)],
        )

    # Theoretical spectral ratio R = <lambda_{k+1} / lambda_1>
    R_w = np.array(
        [
            np.nanmean(
                [
                    (e[args.k_speed] / e[0])
                    if (len(e) > args.k_speed and e[0] > 0)
                    else np.nan
                    for e in _get_evals_trials("wishart", int(d))
                ]
            )
            for d in d_list
        ],
        dtype=float,
    )
    axins_w.plot(d_list, R_w, linewidth=1.0, color="black", label=rf"$\lambda_{{{args.k_speed + 1}}}/\lambda_1$")

    axins_w.grid(True, alpha=0.25)
    axins_w.tick_params(labelsize=8)
    axins_w.set_xlabel("$d$")
    axins_w.set_ylabel(r"$\mathcal{S}_\epsilon$")
    axins_w.legend(loc="center right", bbox_to_anchor=(-0.2, 0.5), frameon=False)
    axins_w.set_title(rf"Speedup, $k={args.k_speed}$", fontsize=15)

    # Save Figure 2 data
    np.savez_compressed(
        os.path.join(out_dir, "figure2_wishart_data.npz"),
        **{f"trace_k{kk}_t": vv["t"] for kk, vv in fig2_traces.items()},
        **{f"trace_k{kk}_t_scaled": vv["t_scaled"] for kk, vv in fig2_traces.items()},
        **{f"trace_k{kk}_E": vv["E"] for kk, vv in fig2_traces.items()},
        **{f"trace_k{kk}_E_min": vv["E_min"] for kk, vv in fig2_traces.items()},
        **{f"trace_k{kk}_E_max": vv["E_max"] for kk, vv in fig2_traces.items()},
        d_list=d_list,
        eps_list=np.array(eps_wishart_list, dtype=float),
        k_speed=args.k_speed,
        m_trace=m_trace,
        lam1_trace=lam1_w_trace,
        t_eps_k0=np.array(fig2_inset_tk0, dtype=float),
        t_eps_k=np.array(fig2_inset_tk, dtype=float),
        S_eps=np.array(fig2_inset_S, dtype=float),
        R_theory=R_w,
    )

    plt.title("Wishart")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Wishart.pdf"))
    plt.show()


if __name__ == "__main__":
    main()
