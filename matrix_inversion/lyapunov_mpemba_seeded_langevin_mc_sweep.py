#!/usr/bin/env python3
"""
Simulate Lyapunov covariance dynamics for thermodynamic matrix inversion:

    Σ̇ = -μ (J Σ + Σ J) + 2 μ kB T I     (Eq. 4)

Compare standard initialization Σ0 = 0 vs optimized initialization that
pre-thermalizes k slow modes:

    Σ0^(opt)(k) = sum_{i=1..k} (kB T / α_i) u_i u_i^T     (Eq. 10 generalized)

We plot the relative Frobenius error || Σ(t) - kB T J^{-1} ||_F\|\Sigma_{\rm eq}\|_F vs t on a semilogy plot.

Two matrix ensembles:
  1) Fixed spectrum with Haar-random eigenvectors
  2) Wishart SPD (positive definite) matrices

References: notes PDF. (Eq. 4, Eq. 10) :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib as mpl
mpl.rcParams["text.usetex"] = True



# -----------------------------
# Matrix ensembles
# -----------------------------
def haar_random_orthogonal(d: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Haar-random orthogonal matrix Q ∈ O(d) via QR with sign fix."""
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # Make Q uniform on O(d): fix signs using diag(R)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    Q = Q * signs
    return Q


def make_J_fixed_spectrum(d: int, spectrum: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """J = Q diag(spectrum) Q^T with Haar-random eigenvectors."""
    if spectrum.shape != (d,):
        raise ValueError("spectrum must have shape (d,)")
    Q = haar_random_orthogonal(d, rng)
    return Q @ np.diag(spectrum) @ Q.T


def make_J_wishart(d: int, m: int, rng: np.random.Generator, ridge: float = 1e-6) -> np.ndarray:
    """
    Wishart SPD: J = (1/m) X^T X + ridge I, with X ∈ R^{m×d} i.i.d. N(0,1).
    For m >= d this is SPD with probability 1; ridge improves conditioning.
    NOTE 1: ridge does not change alpha_1-alpha_2 separation.
    NOTE 2: increasing m increases the alpha_1-alpha_2 separation.
    """
    if m < d:
        raise ValueError("For SPD Wishart, use m >= d (e.g. m=2d).")
    X = rng.normal(size=(m, d))
    J = (X.T @ X) / float(m)
    J = J + ridge * np.eye(d)
    return J

def plot_spectra(J_fixed: np.ndarray, J_wishart: np.ndarray):
    lam_fixed = np.linalg.eigvalsh(J_fixed)
    lam_wish  = np.linalg.eigvalsh(J_wishart)
    print("lam_fixed: ", lam_fixed)
    print("lam_wish: ", lam_wish)

    plt.figure(figsize=(6, 4.5))
    plt.semilogy(lam_fixed, marker='o', linestyle='None', label='fixed spectrum + Haar eigenvectors')
    plt.semilogy(lam_wish,  marker='x', linestyle='None', label='Wishart SPD')
    plt.xlabel("eigenvalue index")
    plt.ylabel("eigenvalue (log scale)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

def wishart_edge_spacing_vs_d(
    d_list,
    m_factor: float,
    ridge: float,
    n_trials: int,
    seed: int = 0,
    loglog: bool = False,
):
    master_seed = int(seed)

    means, stds = [], []
    for d in d_list:
        m = int(np.ceil(m_factor * d))
        spacings = []

        for _ in range(n_trials):
            rng_trial = np.random.default_rng(master_seed + 1000*d + _)
            J = make_J_wishart(d, m=m, rng=rng_trial, ridge=ridge)
            lam = np.linalg.eigvalsh(J)  # ascending
            spacings.append(lam[1] - lam[0])

        spacings = np.array(spacings, dtype=float)
        means.append(spacings.mean())
        stds.append(spacings.std(ddof=1) if n_trials > 1 else 0.0)

    means = np.array(means)
    stds  = np.array(stds)

    plt.figure(figsize=(6, 4.5))
    plt.errorbar(d_list, means, yerr=stds, marker='o', linestyle='-')
    plt.xlabel("dimension $d$")
    plt.ylabel(r"edge spacing $\lambda_2 - \lambda_1$")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    if loglog:
        plt.xscale("log")
        plt.yscale("log")

    plt.tight_layout()
    plt.show()



def smallest_k_eigs_lanczos(J: np.ndarray, k: int, tol: float = 1e-10, maxiter: int | None = None):
    """
    Compute k smallest eigenpairs of symmetric positive definite J using Lanczos/ARPACK.

    Returns:
        evals: (k,) smallest eigenvalues (ascending)
        evecs: (d,k) corresponding eigenvectors
        elapsed: time in seconds
        info: dict with method metadata
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    d = J.shape[0]
    k_eff = min(k, d-1)  # eigsh requires k < d for dense arrays

    t0 = time.perf_counter()
    try:
        from scipy.sparse.linalg import eigsh

        # which='SA' => smallest algebraic
        v0 = np.ones(d, dtype=float)
        v0 /= np.linalg.norm(v0)
        evals, evecs = eigsh(J, k=k_eff, which="SA", v0=v0, tol=tol, maxiter=maxiter)
        # eigsh doesn't guarantee sorted order
        idx = np.argsort(evals)
        evals = evals[idx]
        evecs = evecs[:, idx]

        elapsed = time.perf_counter() - t0
        return evals, evecs, elapsed, {"method": "scipy.sparse.linalg.eigsh (ARPACK/Lanczos)", "k_eff": k_eff}

    except Exception as e:
        # Fallback (still returns something) but note it's full diagonalization
        evals_all, evecs_all = np.linalg.eigh(J)
        evals = evals_all[:k]
        evecs = evecs_all[:, :k]
        elapsed = time.perf_counter() - t0
        return evals, evecs, elapsed, {"method": f"fallback numpy.linalg.eigh (NOT Lanczos): {type(e).__name__}: {e}", "k_eff": k}


# -----------------------------
# Optimized initialization
# -----------------------------
# def sigma0_optimized(J: np.ndarray, k: int, kBT: float) -> np.ndarray:
#     """
#     Generalized Eq. (10): pre-thermalize the first k eigenmodes (smallest α_i).
#         Σ0 = sum_{i=1..k} (kBT/α_i) u_i u_i^T
#     """
#     if k <= 0:
#         return np.zeros_like(J)

#     # eigh returns eigenvalues ascending
#     evals, evecs = np.linalg.eigh(J)
#     k = min(k, J.shape[0])
#     U_k = evecs[:, :k]
#     a_k = evals[:k]
#     if np.any(a_k <= 0):
#         raise ValueError("J must be positive definite (all eigenvalues > 0).")

#     # Σ0 = U_k diag(kBT/a_k) U_k^T
#     return U_k @ np.diag(kBT / a_k) @ U_k.T

def sigma0_optimized(J: np.ndarray, k: int, kBT: float, use_lanczos: bool = True):
    """
    Optimized init (generalized Eq. 10):
        Σ0 = sum_{i=1..k} (kBT/α_i) u_i u_i^T

    If use_lanczos=True, compute (α_i, u_i) via Lanczos (eigsh).
    Returns (Sigma0, timing_info).
    """
    d = J.shape[0]
    if k <= 0:
        return np.zeros((d, d)), {"method": "none", "elapsed": 0.0, "k_eff": 0}

    if use_lanczos and k < d:
        evals, evecs, elapsed, info = smallest_k_eigs_lanczos(J, k=k)
        info["elapsed"] = elapsed
    else:
        t0 = time.perf_counter()
        evals_all, evecs_all = np.linalg.eigh(J)
        evals = evals_all[:k]
        evecs = evecs_all[:, :k]
        elapsed = time.perf_counter() - t0
        info = {"method": "numpy.linalg.eigh", "elapsed": elapsed, "k_eff": k}

    if np.any(evals <= 0):
        raise ValueError("J must be positive definite (all eigenvalues > 0).")

    # Σ0 = U_k diag(kBT/α_k) U_k^T
    Sigma0 = evecs @ np.diag(kBT / evals) @ evecs.T
    Sigma0 = 0.5 * (Sigma0 + Sigma0.T)
    # print("eigenvalues in optimized state ", evals)

    return Sigma0, info


# -----------------------------
# Lyapunov ODE integration
# -----------------------------
def lyapunov_rhs(Sigma: np.ndarray, J: np.ndarray, mu: float, kBT: float, I: np.ndarray) -> np.ndarray:
    """RHS of Eq. (4): Σ̇ = -μ (JΣ + ΣJ) + 2 μ kBT I.

    NOTE: I is precomputed (np.eye(d)) to avoid allocating it at every call.
    """
    return -mu * (J @ Sigma + Sigma @ J) + 2.0 * mu * kBT * I



def integrate_lyapunov(J, Sigma0, mu, kBT, tmax, dt, Sigma_eq):
    if dt <= 0:
        raise ValueError("dt must be positive")
    n_steps = int(np.ceil(tmax / dt))
    d = J.shape[0]
    I = np.eye(d)

    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    S = 0.5 * (Sigma0 + Sigma0.T)
    den = np.linalg.norm(Sigma_eq, ord="fro")
    if den == 0:
        raise ValueError("Sigma_eq Frobenius norm is zero")

    err = np.empty(n_steps + 1, dtype=float)
    err[0] = np.linalg.norm(S - Sigma_eq, ord="fro") #/ den

    for n in range(n_steps):
        k1 = lyapunov_rhs(S,                 J, mu, kBT, I)
        k2 = lyapunov_rhs(S + 0.5*dt*k1,     J, mu, kBT, I)
        k3 = lyapunov_rhs(S + 0.5*dt*k2,     J, mu, kBT, I)
        k4 = lyapunov_rhs(S + dt*k3,         J, mu, kBT, I)

        S = S + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        S = 0.5 * (S + S.T)  # keep symmetry

        err[n + 1] = np.linalg.norm(S - Sigma_eq, ord="fro")# / den

    return t, err


# -----------------------------
# Langevin trajectory unraveling (Ornstein–Uhlenbeck)
# -----------------------------
def simulate_langevin_cov_error(
    J: np.ndarray,
    mu: float,
    kBT: float,
    tmax: float,
    dt: float,
    Sigma_eq: np.ndarray,
    n_traj: int,
    rng: np.random.Generator,
    init: str = "zero",
    k: int = 0,
    eigvals: np.ndarray | None = None,
    eigvecs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate n_traj Langevin trajectories for the OU process:

        dx = -mu * J x dt + sqrt(2 mu kBT) dW

    and estimate the covariance Σ_hat(t) = (1/n) X(t)^T X(t) (mean is zero by symmetry).

    We return the *relative* Frobenius error:
        ||Σ_hat(t) - Σ_eq||_F / ||Σ_eq||_F

    init:
      - "zero": x(0)=0 for all trajectories
      - "sampled_opt": sample x(0) in the span of the k slowest eigenmodes:
            x(0) = sum_{i=1..k} sqrt(kBT/α_i) u_i z_i,   z_i ~ N(0,1)
        (so that E[x(0)x(0)^T] = Σ0^(opt)(k))
    """
    d = J.shape[0]
    n_steps = int(np.ceil(tmax / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    # Initial conditions: X has shape (n_traj, d) with each row a trajectory state
    if init == "zero":
        X = np.zeros((n_traj, d), dtype=float)
    elif init == "sampled_opt":
        if k <= 0:
            raise ValueError("sampled_opt requires k>0")
        if eigvals is None or eigvecs is None:
            raise ValueError("sampled_opt requires (eigvals, eigvecs) for the k slow modes")
        if eigvecs.shape != (d, k) or eigvals.shape != (k,):
            raise ValueError("eigvecs must be (d,k) and eigvals must be (k,)")

        Z = rng.normal(size=(n_traj, k))
        scales = np.sqrt(kBT / eigvals)  # (k,)
        X = (Z * scales[None, :]) @ eigvecs.T
    else:
        raise ValueError(f"unknown init='{init}'")

    # Precompute constants
    den = np.linalg.norm(Sigma_eq, ord="fro")
    if den == 0:
        raise ValueError("Sigma_eq Frobenius norm is zero; cannot compute relative error.")

    noise_scale = np.sqrt(2.0 * mu * kBT * dt)

    # For row-wise states, deterministic drift is: dX/dt = -(mu) X J
    err = np.empty(n_steps + 1, dtype=float)
    Sigma_hat = (X.T @ X) / float(n_traj)
    err[0] = np.linalg.norm(Sigma_hat - Sigma_eq, ord="fro") #/ den

    for n in range(n_steps):
        # Euler–Maruyama update
        X = X - (mu * dt) * (X @ J) + noise_scale * rng.normal(size=X.shape)

        # covariance estimate
        Sigma_hat = (X.T @ X) / float(n_traj)
        err[n + 1] = np.linalg.norm(Sigma_hat - Sigma_eq, ord="fro") #/ den

    return t, err



# -----------------------------
# Main experiment
# -----------------------------


# -----------------------------
# Monte-Carlo (trajectory) convergence utilities
# -----------------------------
def first_passage_time(t: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """Return the first time y(t) <= threshold using linear interpolation, or NaN if never."""
    if y[0] <= threshold:
        return float(t[0])
    for i in range(1, len(t)):
        if y[i] <= threshold:
            # interpolate between i-1 and i
            t0, t1 = t[i-1], t[i]
            y0, y1 = y[i-1], y[i]
            if y1 == y0:
                return float(t1)
            frac = (threshold - y0) / (y1 - y0)
            return float(t0 + frac * (t1 - t0))
    return float("nan")


def mc_error_curve(
    J: np.ndarray,
    mu: float,
    kBT: float,
    tmax: float,
    dt: float,
    Sigma_eq: np.ndarray,
    n_traj: int,
    rng: np.random.Generator,
    init: str,
    k_opt: int = 0,
    eigvals: np.ndarray | None = None,
    eigvecs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Wrapper returning (t, rel_err(t)) for the trajectory estimator."""
    return simulate_langevin_cov_error(
        J=J,
        mu=mu,
        kBT=kBT,
        tmax=tmax,
        dt=dt,
        Sigma_eq=Sigma_eq,
        n_traj=n_traj,
        rng=rng,
        init=init,
        k=k_opt,
        eigvals=eigvals,
        eigvecs=eigvecs,
    )


def required_n_traj_until(
    t_det: np.ndarray,
    err_det: np.ndarray,
    J: np.ndarray,
    mu: float,
    kBT: float,
    dt: float,
    Sigma_eq: np.ndarray,
    eps: float = 1e-2,
    tol: float = 0.2,
    n_traj_start: int = 500,
    n_traj_max: int = 200000,
    reps: int = 3,
    base_seed: int = 0,
    init: str = "zero",
    k_opt: int = 0,
    eigvals: np.ndarray | None = None,
    eigvecs: np.ndarray | None = None,
) -> int:
    """
    Find the smallest n_traj such that the Monte-Carlo estimated relative error curve
    matches the deterministic curve up to the time when the deterministic curve first
    reaches `eps`.

    Accuracy criterion:
      max_t<=t_hit_det  |err_mc(t) - err_det(t)| / err_det(t)  <= tol

    We test `reps` independent Monte-Carlo runs and require the worst-case to satisfy the criterion.

    Notes:
      - This measures 'accuracy of the unraveling' in reproducing the deterministic relaxation curve
        down to error level eps.
      - If the deterministic curve never reaches eps, we use tmax.
    """
    t_hit = first_passage_time(t_det, err_det, eps)
    if not np.isfinite(t_hit):
        t_hit = float(t_det[-1])

    # only compare where deterministic curve is >= eps (avoid comparing in the noise floor)
    compare_mask = t_det <= t_hit
    t_cmp = t_det[compare_mask]
    det_cmp = err_det[compare_mask]
    # guard against tiny det values
    det_cmp = np.maximum(det_cmp, 1e-300)

    n_traj = int(n_traj_start)
    while n_traj <= n_traj_max:
        worst_rel = 0.0
        for r in range(reps):
            rng_r = np.random.default_rng(base_seed + 1000003 * r + 9176 * n_traj)
            t_mc, err_mc = mc_error_curve(
                J=J,
                mu=mu,
                kBT=kBT,
                tmax=float(t_det[-1]),
                dt=dt,
                Sigma_eq=Sigma_eq,
                n_traj=n_traj,
                rng=rng_r,
                init=init,
                k_opt=k_opt,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )
            # Interpolate MC curve onto deterministic time grid (up to t_hit)
            mc_on_det = np.interp(t_cmp, t_mc, err_mc)
            rel = np.max(np.abs(mc_on_det - det_cmp) / det_cmp)
            worst_rel = max(worst_rel, float(rel))
            if worst_rel > tol:
                break

        if worst_rel <= tol:
            return n_traj
        n_traj *= 2

    return int(n_traj_max)


def run_one_ensemble(
    J: np.ndarray,
    mu: float,
    kBT: float,
    tmax: float,
    dt: float,
    ks: list[int],
    use_lanczos: bool = True,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, dict]]:
    """
    Returns:
      results: k -> (t, err(t))
      timings: k -> dict(method, elapsed, ...)
    """
    d = J.shape[0]
    I = np.eye(d)
    Sigma_eq = kBT * np.linalg.solve(J, I)

    results = {}
    timings = {}

    all_ks = [0] + ks
    for k in all_ks:
        if k == 0:
            Sigma0 = np.zeros((d, d))
            timings[k] = {"method": "standard (Sigma0=0)", "elapsed": 0.0, "k_eff": 0}
        else:
            Sigma0, info = sigma0_optimized(J, k=k, kBT=kBT, use_lanczos=use_lanczos)
            timings[k] = info

        t, err = integrate_lyapunov(J, Sigma0, mu, kBT, tmax, dt, Sigma_eq)
        results[k] = (t, err)

    return results, timings



def default_fixed_spectrum(d: int, alpha_min: float = 1.0, alpha_max: float = 10.0) -> np.ndarray:
    # """A simple fixed spectrum (log-spaced) with controlled condition number."""
    # return np.logspace(np.log10(alpha_min), np.log10(alpha_max), d)
    #NOTE: I am calling the function with an alpha_max scaled with d
    return np.linspace(alpha_min, alpha_max, d)  


def choose_dt(J: np.ndarray, mu: float, safety: float = 0.05) -> float:
    """
    Heuristic RK4 stepsize based on fastest decay rate ~ 2 μ α_max.
    We use dt ≈ safety / (2 μ α_max).
    """
    alpha_max = np.linalg.eigvalsh(J).max()
    return safety / (2.0 * mu * alpha_max) 

def first_passage_time(t: np.ndarray, err: np.ndarray, threshold: float) -> float:
    """
    Return the earliest time when err(t) <= threshold.
    Uses linear interpolation between the last point above threshold
    and the first point below threshold.

    Returns np.nan if the threshold is never reached.
    """
    if err[0] <= threshold:
        return float(t[0])

    idx = np.where(err <= threshold)[0]
    if len(idx) == 0:
        return float("nan")

    j = int(idx[0])  # first index below threshold
    i = j - 1        # previous index above threshold

    # Linear interpolation: err(t) ~ err_i + (err_j-err_i)*alpha
    ti, tj = float(t[i]), float(t[j])
    ei, ej = float(err[i]), float(err[j])

    # Avoid divide-by-zero if flat (rare)
    if ej == ei:
        return tj

    alpha = (threshold - ei) / (ej - ei)  # alpha in (0,1]
    return ti + alpha * (tj - ti)


def extract_threshold_times(results: dict[int, tuple[np.ndarray, np.ndarray]], threshold: float) -> dict[int, float]:
    """
    results: dict k -> (t, err)
    returns: dict k -> t_hit
    """
    out = {}
    for k, (t, err) in results.items():
        out[k] = first_passage_time(t, err, threshold)
    return out

def teps_vs_d_curve(
    d_list,
    ensemble: str,              # "fixed" or "wishart"
    k: int,
    epsilon: float,
    mu: float,
    kBT: float,
    alpha_min: float,
    alpha_max: float,
    wishart_m_factor: float,
    wishart_ridge: float,
    tmax: float,
    dt: float | None,
    seed: int,
    wishart_trials: int = 5,
    normalize_error: bool = False,   # optional: divide Frobenius by sqrt(d)
):
    """
    Returns arrays (d_list, t_eps(d)).

    For Wishart we average t_eps over 'wishart_trials' random draws.
    """
    master_seed = int(seed)

    t_eps = []

    for d in d_list:
        print("d = ", d)
        rng_d = np.random.default_rng(master_seed + 1000*d + (0 if ensemble == "fixed" else 12345))
        if ensemble == "fixed":
            # Your current choice was linear spectrum; choose one of the two:

            # (A) endpoints fixed (spacing changes with d):
            spectrum = np.linspace(alpha_min, alpha_max*d, d)
            print("DEBUG d", d, "alpha_min", alpha_min, "alpha_max", alpha_max)
            print("DEBUG spec[0:12] =", spectrum[:12])
            print("DEBUG spec[k], spec[k+1] =", spectrum[k-1], spectrum[k])

            # (B) spacing fixed (endpoints grow with d) — NOT a fixed spectrum:
            # spectrum = np.linspace(alpha_min, alpha_max * d, d)

            J = make_J_fixed_spectrum(d, spectrum, rng_d)

            # choose dt if not provided
            dt_use = dt if dt is not None else choose_dt(J, mu, safety=0.05)

            # run only for k of interest
            res, _ = run_one_ensemble(J, mu, kBT, tmax, dt_use, ks=[k], use_lanczos=True)
            t, err = res[k]

            if normalize_error:
                err = err / np.sqrt(d)

            t_hit = first_passage_time(t, err, epsilon)
            t_eps.append(t_hit)

        elif ensemble == "wishart":
            # average over trials (Wishart randomness)
            hits = []
            for trial in range(wishart_trials):
                m = int(np.ceil(wishart_m_factor * d))
                rng_trial = np.random.default_rng(master_seed + 1000*d + trial + 777)
                J = make_J_wishart(d, m=m, rng=rng_trial, ridge=wishart_ridge)

                dt_use = dt if dt is not None else choose_dt(J, mu, safety=0.05)

                res, _ = run_one_ensemble(J, mu, kBT, tmax, dt_use, ks=[k], use_lanczos=True)
                t, err = res[k]

                if normalize_error:
                    err = err / np.sqrt(d)

                hits.append(first_passage_time(t, err, epsilon))

            # mean over finite draws (ignore NaNs if threshold not reached)
            hits = np.array(hits, dtype=float)
            t_eps.append(np.nanmean(hits))

        else:
            raise ValueError("ensemble must be 'fixed' or 'wishart'")

    return np.array(d_list, dtype=int), np.array(t_eps, dtype=float)

import numpy as np

def alpha_ratio_fixed(d_list, alpha_min, alpha_max, k, *, epsilon=None, kBT=1.0):
    """
    Fixed-spectrum predicted speedup.

    If epsilon is None:
        returns ak/a1
    Else (absolute Frobenius first-passage log-corrected prediction):
        returns (ak/a1) * log(kBT/(eps*a1)) / log(kBT/(eps*ak))

    IMPORTANT: must match your teps_vs_d_curve convention:
        spectrum = linspace(alpha_min, alpha_max * d, d)
    """
    if k < 1:
        raise ValueError("k must be >= 1 (k=1 corresponds to alpha_1).")
    if epsilon is not None and epsilon <= 0:
        raise ValueError("epsilon must be > 0 when provided.")
    if kBT <= 0:
        raise ValueError("kBT must be > 0.")

    out = []
    for d in d_list:
        spectrum = np.linspace(alpha_min, alpha_max * d, d)
        a1 = float(spectrum[0])
        ak = float(spectrum[k - 1])

        r = ak / a1
        if epsilon is not None:
            x1 = kBT / (epsilon * a1)
            xk = kBT / (epsilon * ak)
            if x1 <= 1.0 or xk <= 1.0:
                raise ValueError(
                    f"log arguments must exceed 1 for positive predicted times: "
                    f"kBT/(eps*a1)={x1:.3g}, kBT/(eps*ak)={xk:.3g}. "
                    "Use smaller epsilon or larger kBT."
                )
            r *= np.log(x1) / np.log(xk)

        out.append(r)

    return np.array(out, dtype=float)


def alpha_ratio_wishart(d_list, m_factor, ridge, k, *, n_trials=5, seed=0, epsilon=None, kBT=1.0):
    """
    Wishart predicted speedup (trial-averaged).

    If epsilon is None:
        returns mean_trial( ak/a1 )
    Else (absolute Frobenius first-passage log-corrected prediction):
        returns mean_trial( (ak/a1) * log(kBT/(eps*a1)) / log(kBT/(eps*ak)) )
    """
    if k < 1:
        raise ValueError("k must be >= 1 (k=1 corresponds to alpha_1).")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")
    if epsilon is not None and epsilon <= 0:
        raise ValueError("epsilon must be > 0 when provided.")
    if kBT <= 0:
        raise ValueError("kBT must be > 0.")

    master_seed = int(seed)
    out = []

    for d in d_list:
        m = int(np.ceil(m_factor * d))
        trial_vals = []
        for t in range(n_trials):
            rng_trial = np.random.default_rng(master_seed + 1000 * d + t)
            J = make_J_wishart(d, m=m, rng=rng_trial, ridge=ridge)
            lam = np.linalg.eigvalsh(J)

            a1 = float(lam[0])
            ak = float(lam[k - 1])

            r = ak / a1
            if epsilon is not None:
                x1 = kBT / (epsilon * a1)
                xk = kBT / (epsilon * ak)
                if x1 <= 1.0 or xk <= 1.0:
                    raise ValueError(
                        f"log arguments must exceed 1 for positive predicted times: "
                        f"kBT/(eps*a1)={x1:.3g}, kBT/(eps*ak)={xk:.3g}. "
                        "Use smaller epsilon or larger kBT."
                    )
                r *= np.log(x1) / np.log(xk)

            trial_vals.append(r)

        out.append(float(np.mean(trial_vals)))

    return np.array(out, dtype=float)




def mc_sweep_required_ntraj(args):
    """
    Sweep over d and find the smallest n_traj such that the trajectory-based (unraveled)
    covariance relaxation curve matches the deterministic Lyapunov curve down to eps.

    We compute requirements for:
      - standard init: x(0)=0  ("zero")
      - sampled Mpemba-opt init ("sampled_opt") using k_opt = args.mc_k_opt

    The accuracy criterion is set by args.mc_tol (default 0.2 = 20% relative error w.r.t. det curve)
    up to the deterministic first-passage time to eps.
    """
    d_list = args.mc_d_list
    k_opt = int(args.mc_k_opt)
    eps = float(args.mc_eps)
    tol = float(args.mc_tol)

    req_fixed_zero = []
    req_fixed_opt  = []
    req_wish_zero  = []
    req_wish_opt   = []

    for d in d_list:
        print(f"[MC sweep] d={d}")

        # Deterministic dt choice: keep comparable across runs for a given d and ensemble
        # (If args.dt provided, use it for both.)
        # Build fixed-spectrum matrix
        rng_fix = np.random.default_rng(args.seed + 1000 * d + 11)
        spectrum = default_fixed_spectrum(d, alpha_min=args.alpha_min, alpha_max=args.alpha_max * d)
        J_fixed = make_J_fixed_spectrum(d, spectrum, rng_fix)

        # Build Wishart matrix
        rng_w = np.random.default_rng(args.seed + 2000 * d + 37)
        m = int(np.ceil(args.wishart_m_factor * d))
        J_wish = make_J_wishart(d, m=m, rng=rng_w, ridge=args.wishart_ridge)

        if args.dt is None:
            dt_fixed = choose_dt(J_fixed, args.mu, safety=0.05)
            dt_wish  = choose_dt(J_wish,  args.mu, safety=0.05)
        else:
            dt_fixed = dt_wish = float(args.dt)

        # Equilibrium covariances
        I = np.eye(d)
        Sigma_eq_fixed = args.kBT * np.linalg.solve(J_fixed, I)
        Sigma_eq_wish  = args.kBT * np.linalg.solve(J_wish,  I)

        # Deterministic curves: standard (Sigma0=0)
        Sigma0 = np.zeros((d, d))
        t_fix0, err_fix0 = integrate_lyapunov(J_fixed, Sigma0, args.mu, args.kBT, args.tmax, dt_fixed, Sigma_eq_fixed)
        t_w0,   err_w0   = integrate_lyapunov(J_wish,  Sigma0, args.mu, args.kBT, args.tmax, dt_wish,  Sigma_eq_wish)

        # Deterministic curves: optimized covariance Sigma0_opt(k)
        if k_opt > 0:
            Sigma0_fix_opt, _ = sigma0_optimized(J_fixed, k=k_opt, kBT=args.kBT, use_lanczos=True)
            Sigma0_w_opt,   _ = sigma0_optimized(J_wish,  k=k_opt, kBT=args.kBT, use_lanczos=True)
            t_fixk, err_fixk = integrate_lyapunov(J_fixed, Sigma0_fix_opt, args.mu, args.kBT, args.tmax, dt_fixed, Sigma_eq_fixed)
            t_wk,   err_wk   = integrate_lyapunov(J_wish,  Sigma0_w_opt,   args.mu, args.kBT, args.tmax, dt_wish,  Sigma_eq_wish)

            # eigenpairs for sampled_opt init (k slow modes)
            evals_fix, evecs_fix, _, _ = smallest_k_eigs_lanczos(J_fixed, k_opt)
            evals_w,   evecs_w,   _, _ = smallest_k_eigs_lanczos(J_wish,  k_opt)
        else:
            t_fixk = err_fixk = t_wk = err_wk = None
            evals_fix = evecs_fix = evals_w = evecs_w = None

        # Find required n_traj for each case
        if args.mc_ensemble in ("fixed", "both"):
            n0 = required_n_traj_until(
                t_det=t_fix0, err_det=err_fix0,
                J=J_fixed, mu=args.mu, kBT=args.kBT, dt=dt_fixed,
                Sigma_eq=Sigma_eq_fixed,
                eps=eps, tol=tol,
                n_traj_start=args.mc_n_traj_start, n_traj_max=args.mc_n_traj_max,
                reps=args.mc_reps, base_seed=args.seed + 3000 * d + 101,
                init="zero",
            )
            req_fixed_zero.append(n0)
            if k_opt > 0:
                nk = required_n_traj_until(
                    t_det=t_fixk, err_det=err_fixk,
                    J=J_fixed, mu=args.mu, kBT=args.kBT, dt=dt_fixed,
                    Sigma_eq=Sigma_eq_fixed,
                    eps=eps, tol=tol,
                    n_traj_start=args.mc_n_traj_start, n_traj_max=args.mc_n_traj_max,
                    reps=args.mc_reps, base_seed=args.seed + 3000 * d + 303,
                    init="sampled_opt", k_opt=k_opt, eigvals=evals_fix, eigvecs=evecs_fix,
                )
                req_fixed_opt.append(nk)
            else:
                req_fixed_opt.append(np.nan)

        if args.mc_ensemble in ("wishart", "both"):
            n0 = required_n_traj_until(
                t_det=t_w0, err_det=err_w0,
                J=J_wish, mu=args.mu, kBT=args.kBT, dt=dt_wish,
                Sigma_eq=Sigma_eq_wish,
                eps=eps, tol=tol,
                n_traj_start=args.mc_n_traj_start, n_traj_max=args.mc_n_traj_max,
                reps=args.mc_reps, base_seed=args.seed + 4000 * d + 101,
                init="zero",
            )
            req_wish_zero.append(n0)
            if k_opt > 0:
                nk = required_n_traj_until(
                    t_det=t_wk, err_det=err_wk,
                    J=J_wish, mu=args.mu, kBT=args.kBT, dt=dt_wish,
                    Sigma_eq=Sigma_eq_wish,
                    eps=eps, tol=tol,
                    n_traj_start=args.mc_n_traj_start, n_traj_max=args.mc_n_traj_max,
                    reps=args.mc_reps, base_seed=args.seed + 4000 * d + 303,
                    init="sampled_opt", k_opt=k_opt, eigvals=evals_w, eigvecs=evecs_w,
                )
                req_wish_opt.append(nk)
            else:
                req_wish_opt.append(np.nan)

    # Plot
    plt.figure(figsize=(7, 4.5))
    d_arr = np.array(d_list, dtype=int)

    if args.mc_ensemble in ("fixed", "both"):
        plt.plot(d_arr, req_fixed_zero, marker="o", label="fixed: standard (x0=0)")
        plt.plot(d_arr, req_fixed_opt,  marker="o", label=f"fixed: sampled-opt (k={k_opt})")

    if args.mc_ensemble in ("wishart", "both"):
        plt.plot(d_arr, req_wish_zero, marker="x", label="Wishart: standard (x0=0)")
        plt.plot(d_arr, req_wish_opt,  marker="x", label=f"Wishart: sampled-opt (k={k_opt})")

    plt.yscale("log")
    plt.xlabel("dimension d")
    plt.ylabel(r"required $N_{\mathrm{traj}}$ (accuracy to $\epsilon$)")
    plt.title(fr"Required trajectories for accuracy down to $\epsilon={eps:g}$ (tol={tol:g})")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print table
    print("\nSummary (required N_traj):")
    for i, d in enumerate(d_list):
        parts = [f"d={d}"]
        if args.mc_ensemble in ("fixed", "both"):
            parts.append(f"fixed n0={req_fixed_zero[i]}")
            parts.append(f"fixed nk={req_fixed_opt[i]}")
        if args.mc_ensemble in ("wishart", "both"):
            parts.append(f"wish n0={req_wish_zero[i]}")
            parts.append(f"wish nk={req_wish_opt[i]}")
        print("  " + " | ".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=20, help="matrix dimension")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--mu", type=float, default=1.0, help="mobility μ")
    ap.add_argument("--kBT", type=float, default=1.0, help="kB*T")
    ap.add_argument("--tmax", type=float, default=10.0, help="max simulation time")
    ap.add_argument("--dt", type=float, default=0.00001, help="time step (auto if omitted)")
    ap.add_argument("--wishart_m_factor", type=float, default=5.0, help="m = factor*d for Wishart")
    ap.add_argument("--wishart_ridge", type=float, default=1e-6, help="ridge added to Wishart matrix")
    ap.add_argument("--alpha_min", type=float, default=.5, help="min eigenvalue for fixed-spectrum ensemble")
    ap.add_argument("--alpha_max", type=float, default=.5, help="max eigenvalue for fixed-spectrum ensemble. This gets multiplied by d")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 8, 10], help="k values to plot (optimized init)")
    ap.add_argument("--n_traj", type=int, default=10000, help="number of Langevin trajectories for unraveling")
    ap.add_argument("--no_langevin", dest="langevin", action="store_false", help="disable Langevin unraveling overlay")
    
    # Monte-Carlo convergence sweep (required N_traj vs d)
    ap.add_argument("--mc_sweep", action="store_true", help="sweep d and estimate required N_traj for accurate unraveling down to eps")
    ap.add_argument("--mc_d_list", type=int, nargs="+",
                    default=[12, 24],
                    help="dimensions to sweep for --mc_sweep")
    ap.add_argument("--mc_eps", type=float, default=1e-1, help="target error level eps for MC accuracy")
    ap.add_argument("--mc_tol", type=float, default=0.1, help="relative tolerance vs deterministic curve (e.g. 0.2 = 20%)")
    ap.add_argument("--mc_reps", type=int, default=1, help="number of independent MC repeats per N_traj when testing accuracy")
    ap.add_argument("--mc_n_traj_start", type=int, default=5, help="starting N_traj for doubling search")
    ap.add_argument("--mc_n_traj_max", type=int, default=1000, help="maximum N_traj to consider")
    ap.add_argument("--mc_k_opt", type=int, default=10, help="k used for sampled Mpemba-opt initial condition in MC sweep")
    ap.add_argument("--mc_ensemble", type=str, default="fixed", choices=["fixed", "wishart", "both"],
                    help="which ensemble(s) to sweep in --mc_sweep")

    ap.set_defaults(langevin=True)
    args = ap.parse_args()

    if args.mc_sweep:
        mc_sweep_required_ntraj(args)
        return


    d = args.d
    rng = np.random.default_rng(args.seed)

    # Build matrices
    #NOTE: I am multiplying alpha_max by d so that the spacing remains independent of the dimensionality
    spectrum = default_fixed_spectrum(d, alpha_min=args.alpha_min, alpha_max=args.alpha_max*d)
    J_fixed = make_J_fixed_spectrum(d, spectrum, rng)

    m = int(np.ceil(args.wishart_m_factor * d))
    J_wishart = make_J_wishart(d, m=m, rng=rng, ridge=args.wishart_ridge)

    
    # Choose dt if needed (separately for each J, then take smaller)
    if args.dt is None:
        dt_fixed = choose_dt(J_fixed, args.mu, safety=0.05)
        dt_wish = choose_dt(J_wishart, args.mu, safety=0.05)
        dt = min(dt_fixed, dt_wish)
        print(f"[info] auto dt = {dt:.3e} (min of fixed-spectrum and wishart heuristics)")
    else:
        dt = args.dt

    # Run
    res_fixed, timings_fixed = run_one_ensemble(J_fixed, args.mu, args.kBT, args.tmax, dt, ks=args.ks, use_lanczos=True)
    res_wish,  timings_wish  = run_one_ensemble(J_wishart, args.mu, args.kBT, args.tmax, dt, ks=args.ks, use_lanczos=True)

    print("\nTime to compute k slow modes (Lanczos):")
    print("Fixed spectrum + Haar:")
    for k in [0] + args.ks:
        info = timings_fixed[k]
        print(f"  k={k:>2d}: {info['elapsed']:.6g} s  [{info['method']}]")

    print("Wishart SPD:")
    for k in [0] + args.ks:
        info = timings_wish[k]
        print(f"  k={k:>2d}: {info['elapsed']:.6g} s  [{info['method']}]")


    # #COMPUTE TIME AT WHICH ERROR DROPS BELOW THRESHOLD
    # threshold_fixed = 1e-2
    # threshold_wishart = 1e-1

    # times_fixed = extract_threshold_times(res_fixed, threshold_fixed)
    # times_wish  = extract_threshold_times(res_wish, threshold_wishart)

    # print(f"\nFirst-passage times to error <= {threshold_fixed:g}")
    # print("Fixed spectrum + Haar eigenvectors:")
    # for k in sorted(times_fixed.keys()):
    #     print(f"  k={k:>2d}: t_hit = {times_fixed[k]:.6g}")

    # print(f"\nFirst-passage times to error <= {threshold_wishart:g}")
    # print("Wishart SPD:")
    # for k in sorted(times_wish.keys()):
    #     print(f"  k={k:>2d}: t_hit = {times_wish[k]:.6g}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    def plot_results(ax, results, title: str):
        # Standard
        t0, e0 = results[0]
        ax.semilogy(t0, e0, label=r"standard $k=0$ ($\Sigma_0=0$)")
        # Optimized
        for k in args.ks:
            t, e = results[k]
            ax.semilogy(t, e, label=rf"optimized $k={k}$")
        ax.set_title(title)
        ax.set_xlabel("time $t$")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)

    plot_results(axes[0], res_fixed, "Fixed spectrum + Haar-random eigenvectors")
    plot_results(axes[1], res_wish, f"Wishart SPD (m={m})")

    # Optional: Langevin-trajectory unraveling (OU process)
    if args.langevin:
        k_opt = max([0] + list(args.ks))
        # Precompute equilibrium covariances
        I = np.eye(d)
        Sigma_eq_fixed = args.kBT * np.linalg.solve(J_fixed, I)
        Sigma_eq_wish  = args.kBT * np.linalg.solve(J_wishart, I)

        # Eigenpairs for sampled Mpemba-optimized initial condition
        if k_opt > 0:
            evals_fix, evecs_fix, _, _ = smallest_k_eigs_lanczos(J_fixed, k_opt)
            evals_w,   evecs_w,   _, _ = smallest_k_eigs_lanczos(J_wishart, k_opt)
        else:
            evals_fix = evecs_fix = evals_w = evecs_w = None

        # Use independent RNG streams (reproducible, no dependence on control-flow elsewhere)
        rng_fix_std = np.random.default_rng(int(args.seed) + 1000003)
        rng_fix_opt = np.random.default_rng(int(args.seed) + 1000007)
        rng_w_std   = np.random.default_rng(int(args.seed) + 2000003)
        rng_w_opt   = np.random.default_rng(int(args.seed) + 2000007)

        # Standard init (x0=0)
        tL, eL = simulate_langevin_cov_error(
            J_fixed, args.mu, args.kBT, args.tmax, dt, Sigma_eq_fixed,
            n_traj=args.n_traj, rng=rng_fix_std, init="zero"
        )
        axes[0].semilogy(tL, eL, linestyle="--", linewidth=1.6,
                         label=rf"Langevin ($N={args.n_traj}$, $k=0$)")

        tL, eL = simulate_langevin_cov_error(
            J_wishart, args.mu, args.kBT, args.tmax, dt, Sigma_eq_wish,
            n_traj=args.n_traj, rng=rng_w_std, init="zero"
        )
        axes[1].semilogy(tL, eL, linestyle="--", linewidth=1.6,
                         label=rf"Langevin ($N={args.n_traj}$, $k=0$)")

        # Sampled Mpemba-optimized init in the slow subspace (k_opt)
        if k_opt > 0:
            tL, eL = simulate_langevin_cov_error(
                J_fixed, args.mu, args.kBT, args.tmax, dt, Sigma_eq_fixed,
                n_traj=args.n_traj, rng=rng_fix_opt, init="sampled_opt", k=k_opt,
                eigvals=evals_fix, eigvecs=evecs_fix
            )
            axes[0].semilogy(tL, eL, linestyle=":", linewidth=2.0,
                             label=rf"Langevin sampled-opt ($N={args.n_traj}$, $k={k_opt}$)")

            tL, eL = simulate_langevin_cov_error(
                J_wishart, args.mu, args.kBT, args.tmax, dt, Sigma_eq_wish,
                n_traj=args.n_traj, rng=rng_w_opt, init="sampled_opt", k=k_opt,
                eigvals=evals_w, eigvecs=evecs_w
            )
            axes[1].semilogy(tL, eL, linestyle=":", linewidth=2.0,
                             label=rf"Langevin sampled-opt ($N={args.n_traj}$, $k={k_opt}$)")


    axes[0].set_ylabel(r"relative Frobenius error $\|\Sigma(t)-\Sigma_{\rm eq}\|_F / \|\Sigma_{\rm eq}\|_F$")
    axes[1].legend(loc="best", fontsize=9)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
