
#!/usr/bin/env python3
"""
Deterministic Lyapunov (covariance) dynamics for thermodynamic computing / Mpemba pre-thermalization.

This script:
  - Builds SPD matrices J from either a fixed-spectrum ensemble or a Wishart ensemble.
  - Constructs the equilibrium covariance Sigma_eq = kBT * J^{-1}.
  - Constructs an "optimized" initialization Sigma0 that pre-thermalizes the k slowest modes.
  - Integrates the Lyapunov ODE with explicit RK4:
        dSigma/dt = -mu (J Sigma + Sigma J) + 2 mu kBT I
  - Measures the absolute Frobenius error E(t) = ||Sigma(t) - Sigma_eq||_F
  - Computes the first-passage time t_eps: first time E(t) <= epsilon (with linear interpolation).
  - Plots measured speedup S_eps = t_eps(k=0) / t_eps(k=k_speed) vs dimension d.
  - Overlays a theoretical prediction based on eigenvalues including the epsilon-dependent log correction:
        S_eps^th ≈ (alpha_k/alpha_1) * log(kBT/(eps*alpha_1)) / log(kBT/(eps*alpha_k))
    where alpha_1 is the smallest eigenvalue and alpha_k is the k-th smallest eigenvalue
    (k = number of pre-thermalized modes; the slowest remaining mode is alpha_k in 1-indexed notation).
"""

from __future__ import annotations

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({
    "text.usetex": False,          # use matplotlib's mathtext
    "mathtext.fontset": "stix",     # best-looking math font
    "font.family": "STIXGeneral",
    "axes.labelsize": 25,
    "font.size": 25,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
})

# ---------------------------
# Matrix ensembles
# ---------------------------

def make_J_fixed(d: int, alpha_min: float, alpha_max: float, rng: np.random.Generator) -> np.ndarray:
    """
    Fixed-spectrum SPD ensemble:
      - eigenvalues are linearly spaced with fixed spacing as d changes:
            spectrum = linspace(alpha_min, alpha_max * d, d)
      - eigenvectors are Haar-random (via QR of a Gaussian matrix).
    """
    if d <= 0:
        raise ValueError("d must be positive")
    if alpha_min <= 0 or alpha_max <= 0:
        raise ValueError("alpha_min and alpha_max must be > 0")

    spectrum = np.linspace(alpha_min, alpha_max * d, d)

    # Haar-random orthogonal matrix via QR
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # Fix sign ambiguity for deterministic distribution
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs

    J = Q @ np.diag(spectrum) @ Q.T
    return J


def make_J_wishart(d: int, m: int, rng: np.random.Generator, ridge: float) -> np.ndarray:
    """
    Wishart SPD ensemble:
      J = (X^T X)/m + ridge * I, with X ~ N(0,1)^{m x d}.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    if m < d:
        # You can allow m<d but spectrum will include many ~0 modes; we disallow by default.
        raise ValueError("m must be >= d for a well-conditioned Wishart SPD (before ridge).")
    if ridge < 0:
        raise ValueError("ridge must be >= 0")

    X = rng.normal(size=(m, d))
    J = (X.T @ X) / float(m)
    if ridge > 0:
        J = J + ridge * np.eye(d)
    return J


# ---------------------------
# Lyapunov dynamics + helpers
# ---------------------------

def sigma_eq(J: np.ndarray, kBT: float) -> np.ndarray:
    """Sigma_eq = kBT * J^{-1}."""
    d = J.shape[0]
    return kBT * np.linalg.solve(J, np.eye(d))


def sigma0_optimized(J: np.ndarray, k: int, kBT: float, *, use_lanczos: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Pre-thermalize the k slowest modes (k smallest eigenvalues) in the eigenbasis of J:
        Sigma0 = sum_{i=1}^k (kBT/alpha_i) u_i u_i^T
    Returns (Sigma0, evals_k) where evals_k are the k smallest eigenvalues (ascending).
    """
    d = J.shape[0]
    if k < 0 or k > d:
        raise ValueError("k must satisfy 0 <= k <= d")
    if k == 0:
        return np.zeros((d, d)), np.array([], dtype=float)

    # For simplicity/robustness, default to exact diagonalization (fast enough up to ~few 100s).
    # If you want Lanczos later, you can add a scipy.sparse.linalg.eigsh branch.
    evals, evecs = np.linalg.eigh(J)
    evals = evals[:k]
    evecs = evecs[:, :k]

    Sigma0 = evecs @ np.diag(kBT / evals) @ evecs.T
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)
    return Sigma0, evals


def lyapunov_rhs(S: np.ndarray, J: np.ndarray, mu: float, kBT: float, I: np.ndarray) -> np.ndarray:
    """dS/dt = -mu (J S + S J) + 2 mu kBT I"""
    return -mu * (J @ S + S @ J) + 2.0 * mu * kBT * I


def choose_dt(J: np.ndarray, mu: float, safety: float = 0.05) -> float:
    """
    Heuristic RK4 stepsize based on fastest decay rate ~ 2 mu alpha_max.
    dt ≈ safety / (2 mu alpha_max).
    """
    alpha_max = float(np.linalg.eigvalsh(J).max())
    return safety / (2.0 * mu * alpha_max)


# ---------------------------
# Fast closed-form evaluation in the eigenbasis of J
# ---------------------------

def E_of_t_from_eigs(evals: np.ndarray, k_thermalized: int, mu: float, kBT: float, t: float) -> float:
    """Closed-form absolute Frobenius error ||Sigma(t) - Sigma_eq||_F.

    For the Lyapunov dynamics with symmetric SPD J, the covariance in the eigenbasis
    evolves elementwise. For the initializations used in this script (k=0: Sigma0=0,
    and k>0: pre-thermalize the k slowest modes), the Frobenius error is

        E(t) = kBT * sqrt( sum_{i>k} lambda_i^{-2} * exp(-4 mu lambda_i t) ).

    This avoids integrating a d×d matrix ODE (RK4) and is typically orders of magnitude faster.
    """
    if k_thermalized < 0 or k_thermalized > evals.size:
        raise ValueError("k_thermalized out of range")
    tail = evals[k_thermalized:]
    if tail.size == 0:
        return 0.0
    s = np.sum((1.0 / (tail * tail)) * np.exp(-4.0 * mu * tail * t))
    return float(kBT * np.sqrt(s))


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
    """Smallest t in [0, tmax] with E(t) <= epsilon by monotone bisection.

    Returns np.inf if there is no crossing within [0, tmax].
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
        Emid = E_of_t_from_eigs(evals, k_thermalized, mu, kBT, mid)
        if Emid <= epsilon:
            hi = mid
        else:
            lo = mid
        if (hi - lo) <= rtol * max(1.0, hi):
            break
    return float(hi)


def error_curve_from_eigs(
    evals: np.ndarray,
    k_thermalized: int,
    mu: float,
    kBT: float,
    tmax: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, E(t)) on a uniform grid using the closed form."""
    t = np.linspace(0.0, float(tmax), int(max(2, n_points)))
    tail = evals[k_thermalized:]
    if tail.size == 0:
        return t, np.zeros_like(t)
    expo = np.exp(-4.0 * mu * tail[:, None] * t[None, :])
    s = np.sum((1.0 / (tail * tail))[:, None] * expo, axis=0)
    E = kBT * np.sqrt(s)
    return t, E


def _seed_for_trial(*, ensemble: str, d: int, seed: int, tr: int) -> int:
    """Keep seeding consistent with the original script."""
    ens = ensemble.lower()
    if ens == "fixed":
        return int(seed) + 17 * int(d) + 123 + int(tr)
    if ens == "wishart":
        return int(seed) + 1000 * int(d) + int(tr)
    raise ValueError("ensemble must be 'fixed' or 'wishart'")


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
    """Build all disorder realizations for a given (ensemble, d) once."""
    ens = ensemble.lower()
    if ens not in ("fixed", "wishart"):
        raise ValueError("ensemble must be 'fixed' or 'wishart'")
    Js: list[np.ndarray] = []
    if ens == "fixed":
        n_trials = int(max(1, fixed_trials))
        for tr in range(n_trials):
            rng_tr = np.random.default_rng(_seed_for_trial(ensemble="fixed", d=d, seed=seed, tr=tr))
            Js.append(make_J_fixed(int(d), alpha_min=alpha_min, alpha_max=alpha_max, rng=rng_tr))
    else:
        n_trials = int(max(1, wishart_trials))
        m = int(np.ceil(wishart_m_factor * int(d)))
        for tr in range(n_trials):
            rng_tr = np.random.default_rng(_seed_for_trial(ensemble="wishart", d=d, seed=seed, tr=tr))
            Js.append(make_J_wishart(int(d), m=m, rng=rng_tr, ridge=wishart_ridge))
    return Js


def eigs_trials_from_Js(Js: list[np.ndarray]) -> list[np.ndarray]:
    """Eigenvalues for each trial (cached at the call-site in main)."""
    return [np.linalg.eigvalsh(J) for J in Js]



def integrate_lyapunov_rk4(
    J: np.ndarray,
    Sigma0: np.ndarray,
    mu: float,
    kBT: float,
    tmax: float,
    dt: float,
    Sigma_eq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Integrate Lyapunov ODE using explicit RK4 and stream absolute Frobenius error:
        E(t) = ||Sigma(t) - Sigma_eq||_F
    Returns (t, E).
    """
    if dt <= 0:
        raise ValueError("dt must be positive")
    n_steps = int(np.ceil(tmax / dt))
    d = J.shape[0]
    I = np.eye(d)

    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    S = 0.5 * (Sigma0 + Sigma0.T)

    err = np.empty(n_steps + 1, dtype=float)
    err[0] = np.linalg.norm(S - Sigma_eq, ord="fro")

    for n in range(n_steps):
        k1 = lyapunov_rhs(S, J, mu, kBT, I)
        k2 = lyapunov_rhs(S + 0.5 * dt * k1, J, mu, kBT, I)
        k3 = lyapunov_rhs(S + 0.5 * dt * k2, J, mu, kBT, I)
        k4 = lyapunov_rhs(S + dt * k3, J, mu, kBT, I)
        S = S + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        S = 0.5 * (S + S.T)

        err[n + 1] = np.linalg.norm(S - Sigma_eq, ord="fro")

    return t, err


def first_passage_time(t: np.ndarray, E: np.ndarray, epsilon: float) -> float:
    """
    First time E(t) <= epsilon, with linear interpolation between steps.
    Returns np.inf if never crosses.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if E[0] <= epsilon:
        return float(t[0])

    for n in range(len(t) - 1):
        if E[n] > epsilon and E[n + 1] <= epsilon:
            # linear interpolation
            t0, t1 = float(t[n]), float(t[n + 1])
            e0, e1 = float(E[n]), float(E[n + 1])
            if e1 == e0:
                return t1
            return t0 + (epsilon - e0) * (t1 - t0) / (e1 - e0)
    return float("inf")


# ---------------------------
# Theoretical prediction (log-corrected)
# ---------------------------

def teps_vs_d_curve(
    d_list: np.ndarray,
    ensemble: str,
    k_thermalized: int,
    *,
    epsilon: float,
    mu: float,
    kBT: float,
    alpha_min: float,
    alpha_max: float,
    wishart_m_factor: float,
    wishart_ridge: float,
    wishart_trials: int,
    fixed_trials: int,
    seed: int,
    tmax: float,
    dt: float,
    dt_safety: float,
) -> np.ndarray:
    """
    Returns array t_eps(d) for chosen ensemble and number of thermalized modes k_thermalized.
    """
    ens = ensemble.lower()
    if ens not in ("fixed", "wishart"):
        raise ValueError("ensemble must be 'fixed' or 'wishart'")

    t_eps_list = []
    for d in d_list:
        d = int(d)
        rng = np.random.default_rng(int(seed) + 17 * d + 123)

        if ens == "fixed":
            # average t_eps over multiple random eigenvector realizations
            t_trials = []
            for tr in range(fixed_trials):
                rng_tr = np.random.default_rng(int(seed) + 17 * d + 123 + tr)
                J = make_J_fixed(d, alpha_min=alpha_min, alpha_max=alpha_max, rng=rng_tr)
                t_trials.append(_single_teps(J, k_thermalized, epsilon, mu, kBT, tmax, dt, dt_safety))
            t_eps_list.append(float(np.mean(t_trials)))
            continue
        else:
            # average t_eps over multiple realizations
            m = int(np.ceil(wishart_m_factor * d))
            t_trials = []
            for tr in range(wishart_trials):
                rng_tr = np.random.default_rng(int(seed) + 1000 * d + tr)
                J = make_J_wishart(d, m=m, rng=rng_tr, ridge=wishart_ridge)
                t_trials.append(_single_teps(J, k_thermalized, epsilon, mu, kBT, tmax, dt, dt_safety))
            t_eps_list.append(float(np.mean(t_trials)))
            continue

        t_eps_list.append(_single_teps(J, k_thermalized, epsilon, mu, kBT, tmax, dt, dt_safety))

    return np.array(t_eps_list, dtype=float)


def _single_teps(
    J: np.ndarray,
    k_thermalized: int,
    epsilon: float,
    mu: float,
    kBT: float,
    tmax: float,
    dt: float,
    dt_safety: float,
) -> float:
    """Fast first-passage time using eigenvalues (no RK4)."""
    evals = np.linalg.eigvalsh(J)
    return first_passage_time_from_eigs(evals, k_thermalized, epsilon, mu, kBT, tmax)





def error_curve_for_instance(
    J: np.ndarray,
    k_thermalized: int,
    *,
    mu: float,
    kBT: float,
    tmax: float,
    dt: float,
    dt_safety: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, E(t)) for one matrix instance and one k.

    Uses the closed-form eigenbasis expression (no RK4). The time grid is:
      - if dt > 0: uniform with step dt
      - else: uses a default resolution of 2500 points.
    """
    evals = np.linalg.eigvalsh(J)
    if dt > 0:
        n_points = int(np.ceil(float(tmax) / float(dt))) + 1
    else:
        n_points = 2500
    return error_curve_from_eigs(evals, int(k_thermalized), float(mu), float(kBT), float(tmax), n_points)


def error_curve_stats_for_ensemble(
    *,
    ensemble: str,
    d: int,
    k_thermalized: int,
    mu: float,
    kBT: float,
    tmax: float,
    dt: float,
    dt_safety: float,
    seed: int,
    # fixed params
    alpha_min: float,
    alpha_max: float,
    fixed_trials: int,
    # wishart params
    wishart_m_factor: float,
    wishart_ridge: float,
    wishart_trials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return median error curve with a min/max band across disorder realizations.

    Returns (t, E_med, E_min, E_max, lam1_trials).
    The time grid is shared across trials. If dt <= 0, we pick a conservative
    dt as the minimum of the per-trial heuristic time steps.

    Notes:
      - For 'fixed': average over `fixed_trials` Haar-random eigenvector draws.
      - For 'wishart': average over `wishart_trials` independent Wishart matrices.
    """
    ens = ensemble.lower()
    if ens not in ("fixed", "wishart"):
        raise ValueError("ensemble must be 'fixed' or 'wishart'")
    if d <= 0:
        raise ValueError("d must be positive")

    # --- Build J realizations (deterministic seeding, consistent with teps_vs_d_curve) ---
    Js: list[np.ndarray] = []
    if ens == "fixed":
        n_trials = int(max(1, fixed_trials))
        for tr in range(n_trials):
            rng_tr = np.random.default_rng(int(seed) + 17 * int(d) + 123 + tr)
            Js.append(make_J_fixed(int(d), alpha_min=alpha_min, alpha_max=alpha_max, rng=rng_tr))
    else:
        n_trials = int(max(1, wishart_trials))
        m = int(np.ceil(wishart_m_factor * int(d)))
        for tr in range(n_trials):
            rng_tr = np.random.default_rng(int(seed) + 1000 * int(d) + tr)
            Js.append(make_J_wishart(int(d), m=m, rng=rng_tr, ridge=wishart_ridge))

    # --- Shared dt ---
    if dt > 0:
        dt_use = float(dt)
    else:
        # Conservative choice: use the smallest suggested dt across trials.
        dt_use = min(choose_dt(J, mu=mu, safety=dt_safety) for J in Js)

    # --- Evaluate each trial on the shared grid (fast closed form) ---
    E_trials = []
    lam1_trials = []
    t_ref = None
    for J in Js:
        evals = np.linalg.eigvalsh(J)

        if dt_use > 0:
            n_points = int(np.ceil(float(tmax) / float(dt_use))) + 1
        else:
            n_points = 2500

        t_i, E_i = error_curve_from_eigs(
            evals=evals,
            k_thermalized=int(k_thermalized),
            mu=float(mu),
            kBT=float(kBT),
            tmax=float(tmax),
            n_points=int(n_points),
        )

        if t_ref is None:
            t_ref = t_i
        else:
            if len(t_i) != len(t_ref) or np.max(np.abs(t_i - t_ref)) > 1e-12:
                E_i = np.interp(t_ref, t_i, E_i)

        E_trials.append(E_i)
        lam1_trials.append(float(evals[0]))

    t_ref = np.asarray(t_ref, dtype=float)
    E_stack = np.vstack(E_trials)  # (n_trials, nT)
    E_med = np.median(E_stack, axis=0)
    E_min = np.min(E_stack, axis=0)
    E_max = np.max(E_stack, axis=0)
    return t_ref, E_med, E_min, E_max, np.asarray(lam1_trials, dtype=float)
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Lyapunov/covariance Mpemba speedup (deterministic).")

    # dimensions
    ap.add_argument("--d_min", type=int, default=30)
    ap.add_argument("--d_max", type=int, default=35)
    ap.add_argument("--d_step", type=int, default=5)

    # physics params
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--kBT", type=float, default=1.0)

    # error thresholds (absolute Frobenius), can be set independently
    ap.add_argument("--epsilon_fixed_list", type=str, default="1e-2, 1e-3, 1e-4",
                    help="Comma-separated epsilons for fixed inset, e.g. 1e-6,1e-8")
    ap.add_argument("--epsilon_wishart_list", type=str, default="1e-1, 1e-2, 1e-3",
                    help="Comma-separated epsilons for Wishart inset, e.g. 1e-6,1e-8")

    # Mpemba parameter
    ap.add_argument("--k_speed", type=int, default=10, help="Number of slow modes pre-thermalized.")

    # error-curve trace plot
    ap.add_argument("--k_list", type=str, default="0,1,5,10",
                    help="Comma-separated list of k values (number of slow modes pre-thermalized) to plot E(t) for.")
    ap.add_argument("--d_trace", type=int, default=30,
                    help="Dimension used for the time-dependent error trace plot.")

    # fixed-spectrum params
    ap.add_argument("--alpha_min", type=float, default=0.5)
    ap.add_argument("--alpha_max", type=float, default=0.5,
                    help="Fixed-spectrum spacing parameter (spectrum is linspace(alpha_min, alpha_max*d, d)).")
    ap.add_argument("--fixed_trials", type=int, default=1, help="Realizations (random eigenvectors) to average for fixed-spectrum ensemble.")

    # Wishart params
    ap.add_argument("--wishart_m_factor", type=float, default=2, help="m = ceil(m_factor * d)")
    ap.add_argument("--wishart_ridge", type=float, default=0.0)
    ap.add_argument("--wishart_trials", type=int, default=1, help="Realizations to average for measured t_eps.")

    # integration params
    ap.add_argument("--tmax_fixed", type=float, default=15.0, help="Max integration time for fixed-spectrum runs")
    ap.add_argument("--tmax_wishart", type=float, default=70.0, help="Max integration time for Wishart runs")
    ap.add_argument("--dt", type=float, default=-1.0,
                    help="Time step. If <=0, use heuristic dt based on alpha_max.")
    ap.add_argument("--dt_safety", type=float, default=0.05)

    # RNG
    ap.add_argument("--seed", type=int, default=0)

    return ap.parse_args()



def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Output folder: save all data underlying the two figures.
    # Folder name encodes all run parameters to make results reproducible.
    # ------------------------------------------------------------------
    def _clean(x: str) -> str:
        """Filesystem-safe token."""
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
        f"tr_wish={args.wishart_trials}",        f"tmaxF={args.tmax_fixed}",
        f"tmaxW={args.tmax_wishart}",
        f"dt={args.dt}",
        f"dtsafe={args.dt_safety}",
    ]
    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    out_dir = os.path.join(out_base, "__".join(_clean(t) for t in param_tokens))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    # Parse lists
    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    eps_fixed_list = [float(x) for x in args.epsilon_fixed_list.split(",") if x.strip()]
    eps_wishart_list = [float(x) for x in args.epsilon_wishart_list.split(",") if x.strip()]

    d_list = np.arange(args.d_min, args.d_max + 1, args.d_step, dtype=int)

    # ------------------------------------------------------------------
    # Cache disorder realizations / spectra so we reuse them across:
    #   - multiple k in the trace plots
    #   - multiple eps in the inset (speedup) plots
    # This avoids re-building matrices and re-diagonalizing them repeatedly.
    # ------------------------------------------------------------------
    evals_cache: dict[tuple[str, int], list[np.ndarray]] = {}

    def _get_evals_trials(ensemble: str, d: int) -> list[np.ndarray]:
        key = (ensemble.lower(), int(d))
        if key in evals_cache:
            return evals_cache[key]
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

    def _mean_teps_from_cached_eigs(evals_trials: list[np.ndarray], k_th: int, eps: float, tmax: float) -> float:
        t_list = [
            first_passage_time_from_eigs(e, int(k_th), float(eps), float(args.mu), float(args.kBT), float(tmax))
            for e in evals_trials
        ]
        # If any trial never reaches eps within tmax, keep it as inf (np.mean will yield inf).
        return float(np.mean(np.asarray(t_list, dtype=float)))

    # === Build one representative instance for saving (tr=0), but plot traces as disorder-averaged stats ===
    rng = np.random.default_rng(args.seed)
    J_fix = make_J_fixed(d=args.d_trace, alpha_min=args.alpha_min, alpha_max=args.alpha_max, rng=rng)
    m_trace = int(np.ceil(args.wishart_m_factor * args.d_trace))
    J_w = make_J_wishart(d=args.d_trace, m=m_trace, rng=rng, ridge=args.wishart_ridge)

    # Save the representative matrices (and their spectra) used for the trace plots
    np.savez_compressed(
        os.path.join(out_dir, "trace_matrices.npz"),
        d_trace=args.d_trace,
        m_trace=m_trace,
        J_fixed=J_fix,
        J_wishart=J_w,
        lam_fixed=np.linalg.eigvalsh(J_fix),
        lam_wishart=np.linalg.eigvalsh(J_w),
    )

    # ================================================================
    # Figure 1: fixed — main: E(t) traces; inset: speedup vs d for eps list
    # ================================================================
    fig_fix, ax_fix = plt.subplots(figsize=(8, 4.5))

    # Collect data for saving
    fig1_traces = {}
    fig1_inset = {"d_list": d_list.copy(), "k_speed": args.k_speed, "eps_list": np.array(eps_fixed_list, dtype=float)}

    # Reuse the same disorder realizations for all k in the trace plot
    evals_trials_fixed_trace = _get_evals_trials("fixed", int(args.d_trace))

    # Shared time grid for trace plot
    if args.dt > 0:
        n_points_trace = int(np.ceil(float(args.tmax_fixed) / float(args.dt))) + 1
    else:
        n_points_trace = 2500
    t_f = np.linspace(0.0, float(args.tmax_fixed), int(max(2, n_points_trace)))

    for k_th in k_list:
        # Compute E(t) for each trial using cached eigenvalues
        E_trials = []
        for evals in evals_trials_fixed_trace:
            _, E_i = error_curve_from_eigs(
                evals=evals,
                k_thermalized=int(k_th),
                mu=float(args.mu),
                kBT=float(args.kBT),
                tmax=float(args.tmax_fixed),
                n_points=int(len(t_f)),
            )
            E_trials.append(E_i)

        E_stack = np.vstack(E_trials)
        E_med = np.median(E_stack, axis=0)
        E_min = np.min(E_stack, axis=0)
        E_max = np.max(E_stack, axis=0)

        fig1_traces[int(k_th)] = {
            "t": t_f,
            "t_scaled": t_f * (args.mu * args.alpha_min),
            "E": E_med,          # backward-compat: store median under 'E'
            "E_min": E_min,
            "E_max": E_max,
        }
        (ln,) = ax_fix.semilogy(t_f * (args.mu * args.alpha_min), E_med, linestyle="-", label=fr"$k={k_th}$")
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
    # ax_fix.set_title(fr"Fixed-spectrum error traces at $d={args.d_trace}$")
    ax_fix.grid(True, alpha=0.3)
    ax_fix.legend(loc="upper left", frameon=False, ncol=2)
    ax_fix.set_ylim([1e-10, 1e+5])
    
    # Inset: speedup vs d
    axins_fix = ax_fix.inset_axes([0.58, 0.52, 0.38, 0.43])  # [x0,y0,w,h] in axes fraction
    # axins_fix.set_title("speedup")

    inset_linestyle_list = ["solid", "dashed", "dotted"]
    inset_linestyle_counter = 0
    fig1_inset_S = []
    fig1_inset_tk0 = []
    fig1_inset_tk = []
    for eps in eps_fixed_list:
        # Reuse cached spectra for each d and trial (no re-sampling / re-diagonalizing).
        t_fix_k0 = np.array([
            _mean_teps_from_cached_eigs(_get_evals_trials("fixed", int(d)), 0, float(eps), float(args.tmax_fixed))
            for d in d_list
        ], dtype=float)

        t_fix_k = np.array([
            _mean_teps_from_cached_eigs(_get_evals_trials("fixed", int(d)), int(args.k_speed), float(eps), float(args.tmax_fixed))
            for d in d_list
        ], dtype=float)

        S_fix = np.divide(
            t_fix_k0, t_fix_k,
            out=np.full_like(t_fix_k0, np.nan, dtype=float),
            where=np.isfinite(t_fix_k0) & np.isfinite(t_fix_k) & (t_fix_k > 0),
        )

        fig1_inset_tk0.append(t_fix_k0)
        fig1_inset_tk.append(t_fix_k)
        fig1_inset_S.append(S_fix)

        mask = np.isfinite(S_fix)
        exp_label = int(np.floor(np.log10(eps) + 1e-12))
        (ln,) = axins_fix.plot(
            d_list[mask], S_fix[mask],
            color="#d62728",
            marker="o",
            linewidth=1.2,
            label=rf"$\epsilon=10^{{{exp_label}}}$",
            linestyle=inset_linestyle_list[inset_linestyle_counter],
        )
        inset_linestyle_counter += 1
    
    # --- Reconstruct R_fix (spectral ratio) from cached eigenvalues ---
    # R_fix(d) := <lambda_{k+1} / lambda_1> averaged over the same cached trials.
    # (k = args.k_speed; label uses k+1 because eigenvalues are 1-indexed in the paper.)
    R_fix = np.array([
    np.nanmean([
        (ev[int(args.k_speed)] / ev[0]) if (len(ev) > int(args.k_speed) and ev[0] > 0) else np.nan
        for ev in _get_evals_trials("fixed", int(d))
    ])
    for d in d_list
    ], dtype=float)
    
    axins_fix.plot(d_list, R_fix, linewidth=1.0, color='black', label=rf'$\lambda_{{{args.k_speed+1}}}/\lambda_1$')

    # Save Figure 1 data
    fig1_inset["t_eps_k0"] = np.array(fig1_inset_tk0, dtype=float)
    fig1_inset["t_eps_k"] = np.array(fig1_inset_tk, dtype=float)
    fig1_inset["S_eps"] = np.array(fig1_inset_S, dtype=float)
    fig1_inset["R_theory"] = R_fix
    np.savez_compressed(os.path.join(out_dir, "figure1_fixed_data.npz"),
                        traces=np.array([0], dtype=int),  # placeholder for backward compat
                        **{f"trace_k{kk}_t": vv["t"] for kk, vv in fig1_traces.items()},
                        **{f"trace_k{kk}_t_scaled": vv["t_scaled"] for kk, vv in fig1_traces.items()},
        **{f"trace_k{kk}_E": vv["E"] for kk, vv in fig1_traces.items()},
        **{f"trace_k{kk}_E_min": vv["E_min"] for kk, vv in fig1_traces.items()},
        **{f"trace_k{kk}_E_max": vv["E_max"] for kk, vv in fig1_traces.items()},
                        d_list=fig1_inset["d_list"],
                        eps_list=fig1_inset["eps_list"],
                        k_speed=fig1_inset["k_speed"],
                        t_eps_k0=fig1_inset["t_eps_k0"],
                        t_eps_k=fig1_inset["t_eps_k"],
                        S_eps=fig1_inset["S_eps"],
                        R_theory=fig1_inset["R_theory"],
                        )
    axins_fix.grid(True, alpha=0.25)
    # axins_fix.tick_params(labelsize=8)
    axins_fix.set_xlabel("$d$")
    axins_fix.set_ylabel(r"$\mathcal{S}_\epsilon$")
    axins_fix.legend(loc="center right",bbox_to_anchor=(-0.2, 0.5),frameon=False)
    axins_fix.set_title(rf"Speedup, $k={args.k_speed}$", fontsize=15)
    
    plt.title("Haar")
    plt.tight_layout()
    plt.savefig(out_dir+"/Haar.pdf")
    plt.show()

    # ================================================================
    # Figure 2: Wishart — main: E(t) traces; inset: speedup vs d for eps list
    # ================================================================
    fig_w, ax_w = plt.subplots(figsize=(8, 4.5))

    # Collect data for saving
    fig2_traces = {}
    fig2_inset = {"d_list": d_list.copy(), "k_speed": args.k_speed, "eps_list": np.array(eps_wishart_list, dtype=float), "m_trace": m_trace}
    # Reuse the same disorder realizations for all k in the Wishart trace plot
    evals_trials_wish_trace = _get_evals_trials("wishart", int(args.d_trace))
    lam1_w_trace = float(np.median([float(e[0]) for e in evals_trials_wish_trace]))

    # Shared time grid for trace plot
    if args.dt > 0:
        n_points_trace = int(np.ceil(float(args.tmax_wishart) / float(args.dt))) + 1
    else:
        n_points_trace = 2500
    t_wt = np.linspace(0.0, float(args.tmax_wishart), int(max(2, n_points_trace)))

    for k_th in k_list:
        E_trials = []
        for evals in evals_trials_wish_trace:
            _, E_i = error_curve_from_eigs(
                evals=evals,
                k_thermalized=int(k_th),
                mu=float(args.mu),
                kBT=float(args.kBT),
                tmax=float(args.tmax_wishart),
                n_points=int(len(t_wt)),
            )
            E_trials.append(E_i)

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
        (ln,) = ax_w.semilogy(t_wt * (args.mu * lam1_w_trace), E_med, linestyle="-", label=fr"$k={k_th}$")
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
    # ax_w.set_title(fr"Wishart error traces at $d={args.d_trace}$")
    ax_w.grid(True, alpha=0.3)
    ax_w.legend(loc="upper left", frameon=False, ncol=2)
    ax_w.set_ylim([1e-10, 1e+5])
    axins_w = ax_w.inset_axes([0.58, 0.52, 0.38, 0.43])
    # axins_w.set_title("speedup", fontsize=9)

    inset_linestyle_list = ["solid", "dashed", "dotted"]
    inset_linestyle_counter = 0
    fig2_inset_tk0 = []
    fig2_inset_tk = []
    fig2_inset_S = []
    for eps in eps_wishart_list:
        t_w_k0 = np.array([
            _mean_teps_from_cached_eigs(_get_evals_trials("wishart", int(d)), 0, float(eps), float(args.tmax_wishart))
            for d in d_list
        ], dtype=float)

        t_w_k = np.array([
            _mean_teps_from_cached_eigs(_get_evals_trials("wishart", int(d)), int(args.k_speed), float(eps), float(args.tmax_wishart))
            for d in d_list
        ], dtype=float)

        S_w = np.divide(
            t_w_k0, t_w_k,
            out=np.full_like(t_w_k0, np.nan, dtype=float),
            where=np.isfinite(t_w_k0) & np.isfinite(t_w_k) & (t_w_k > 0),
        )

        fig2_inset_tk0.append(t_w_k0)
        fig2_inset_tk.append(t_w_k)
        fig2_inset_S.append(S_w)

        mask = np.isfinite(S_w)
        exp_label = int(np.floor(np.log10(eps) + 1e-12))
        (ln,) = axins_w.plot(
            d_list[mask], S_w[mask],
            color="#d62728",
            marker="o",
            linewidth=1.2,
            label=rf"$\epsilon=10^{{{exp_label}}}$",
            linestyle=inset_linestyle_list[inset_linestyle_counter],
        )
        inset_linestyle_counter += 1
    # Reconstruct theory ratio R_w(d) = <lambda_{k+1} / lambda_1> across cached Wishart trials
    R_w = np.array([
        np.nanmean([
            (e[int(args.k_speed)] / e[0]) if (len(e) > int(args.k_speed) and e[0] > 0) else np.nan
            for e in _get_evals_trials("wishart", int(d))
        ])
        for d in d_list
    ], dtype=float)

    axins_w.plot(d_list, R_w, linewidth=1.0, color='black', label=rf'$\lambda_{{{args.k_speed+1}}}/\lambda_1$')

    # Save Figure 2 data
    fig2_inset["t_eps_k0"] = np.array(fig2_inset_tk0, dtype=float)
    fig2_inset["t_eps_k"] = np.array(fig2_inset_tk, dtype=float)
    fig2_inset["S_eps"] = np.array(fig2_inset_S, dtype=float)
    fig2_inset["R_theory"] = R_w
    np.savez_compressed(
        os.path.join(out_dir, "figure2_wishart_data.npz"),
        traces=np.array([0], dtype=int),  # placeholder for backward compat
        **{f"trace_k{kk}_t": vv["t"] for kk, vv in fig2_traces.items()},
        **{f"trace_k{kk}_t_scaled": vv["t_scaled"] for kk, vv in fig2_traces.items()},
        **{f"trace_k{kk}_E": vv["E"] for kk, vv in fig2_traces.items()},
        **{f"trace_k{kk}_E_min": vv["E_min"] for kk, vv in fig2_traces.items()},
        **{f"trace_k{kk}_E_max": vv["E_max"] for kk, vv in fig2_traces.items()},
        d_list=fig2_inset["d_list"],
        eps_list=fig2_inset["eps_list"],
        k_speed=fig2_inset["k_speed"],
        m_trace=fig2_inset["m_trace"],
        lam1_trace=lam1_w_trace,
        t_eps_k0=fig2_inset["t_eps_k0"],
        t_eps_k=fig2_inset["t_eps_k"],
        S_eps=fig2_inset["S_eps"],
        R_theory=fig2_inset["R_theory"],
    )
    axins_w.grid(True, alpha=0.25)
    axins_w.tick_params(labelsize=8)
    axins_w.set_xlabel("$d$")
    axins_w.set_ylabel(r"$\mathcal{S}_\epsilon$")
    axins_w.legend(loc="center right",bbox_to_anchor=(-0.2, 0.5),frameon=False)
    axins_w.set_title(rf"Speedup, $k={args.k_speed}$", fontsize=15)
    plt.title("Wishart")
    plt.tight_layout()
    plt.savefig(out_dir+"/Wishart.pdf")
    plt.show()


if __name__ == "__main__":
    main()