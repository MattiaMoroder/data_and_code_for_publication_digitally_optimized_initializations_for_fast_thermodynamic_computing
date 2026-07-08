#!/usr/bin/env python3
"""
Crooks/BAR determinant protocol for SPD matrices with optional Mpemba-accelerated burn-in.

Implements the determinant-estimation idea from thermodynamic linear algebra using
Crooks' fluctuation theorem in its statistically efficient BAR form:
- Prepare equilibrium at A1.
- Drive A(t) from A1 -> A2 = a I over duration tau and measure forward work W_F.
- Prepare equilibrium at A2 = a I.
- Drive the reverse protocol A2 -> A1 and measure reverse work W_R.
- Solve the BAR equation for the free-energy difference ΔF.
- For quadratic H = 1/2 x^T A x (SPD),
      ΔF = (1/(2beta)) ln(det A2 / det A1),
  hence
      logdet(A1) = logdet(A2) - 2 beta ΔF.

Adds a convergence study similar to Fig. 6 in arXiv:2308.05660.

Mpemba-style speedup option:
- Use an optimized forward initialization that pre-thermalizes the slowest K modes of A1
  in the eigenbasis of A1.
- The forward burn-in time is then set by the next-slowest mode lambda_{K+1}
  instead of lambda_1.
- The reverse Crooks protocol still starts from equilibrium at A2, as required.

Designed to run fast for small matrices by default.

Reproducible: SeedSequence-spawned RNGs.

Dependencies:
  - numpy
  - matplotlib
  - tqdm (optional; if missing, the script will run without a progress bar)

EXAMPLE USAGE:
python3 crooks_det_wishart_fixed_mpemba.py --estimator crooks --a2 1.0 --n_traj 500 --n_traj_rev 500 --mpemba_K 2 --convergence --xaxis time --save_npz --show
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import os

_BURNIN_PRINTED = set()  # used to print burn-in checks once

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # fall back to no progress bar


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


def make_A1(ensemble: str, d: int, rng: np.random.Generator, *,
            wishart_m_factor: float, wishart_ridge: float,
            alpha_min: float, alpha_max: float) -> np.ndarray:
    """Factory for A1 matrices (matches lyapunov_covariance.py parameters)."""
    if ensemble == "wishart":
        m = int(math.ceil(wishart_m_factor * d))
        return make_J_wishart(d=d, m=m, rng=rng, ridge=wishart_ridge)
    if ensemble == "fixed":
        return make_J_fixed(d=d, alpha_min=alpha_min, alpha_max=alpha_max, rng=rng)
    raise ValueError(f"Unknown ensemble: {ensemble}")

@dataclass
class ProtocolParams:
    beta: float          # inverse temperature
    a2: float            # A2 = a2 * I
    tau: float           # driving duration
    dt: float            # integration timestep
    burnin_mode: str     # 'mult' or 'eps'
    burnin_mult: float   # used if burnin_mode=='mult'
    burnin_eps: float    # target relative covariance error if burnin_mode=='eps'
    seed: int            # base seed for trajectory noise
    print_burnin_check: bool = False  # print E(t_burn) vs burnin_eps once per run


def _check_spd(A: np.ndarray, name: str = "A") -> None:
    A = 0.5 * (A + A.T)
    w = np.linalg.eigvalsh(A)
    if np.min(w) <= 0:
        raise ValueError(f"{name} is not SPD (min eig = {np.min(w):.3e}).")


def exact_logdet(A: np.ndarray) -> float:
    sign, logdet = np.linalg.slogdet(A)
    if sign <= 0:
        raise ValueError("Matrix is not SPD (slogdet sign <= 0).")
    return float(logdet)


def logdet_errors(logdet_exact: float, logdet_est: float) -> Tuple[float, float]:
    """Return absolute and relative errors for log-determinants.

    abs_err = |L_est - L_exact|
    rel_err = abs_err / |L_exact|, with the convention:
      - 0 if abs_err == 0 and L_exact == 0
      - inf if abs_err > 0 and L_exact == 0
    """
    abs_err = abs(float(logdet_est) - float(logdet_exact))
    denom = abs(float(logdet_exact))
    if denom == 0.0:
        rel_err = 0.0 if abs_err == 0.0 else float("inf")
    else:
        rel_err = abs_err / denom
    return abs_err, rel_err


def _effective_mpemba_K(K: int, d: int, *, context: str = "") -> int:
    """Clip K to a benchmark-safe value in [0, d-1]."""
    if d <= 0:
        raise ValueError("d must be positive")
    K_in = int(K)
    K_eff = max(0, min(K_in, d - 1))
    if K_in >= d:
        msg = f"mpemba_K={K_in} >= d={d}; using K={K_eff} instead"
        if context:
            msg += f" [{context}]"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    return K_eff


def sample_mpemba_x0(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    beta: float,
    K: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample x0 ~ N(0, Sigma0) where Sigma0 is "Mpemba optimized":
      - for i < K (slowest modes): variance = 1/(beta*lambda_i) (equilibrium)
      - for i >= K: variance = 0 (start at 0 in those modes)

    This makes the burn-in dominated by mode K+1 (if K < d).
    """
    d = eigvals.shape[0]
    K = int(max(0, min(K, d)))
    if K == 0:
        return np.zeros(d, dtype=float)
    # diagonal stds in eigenbasis
    std = np.zeros(d, dtype=float)
    std[:K] = np.sqrt(1.0 / (beta * eigvals[:K]))
    z = rng.normal(size=d)
    y = std * z  # in eigenbasis
    return eigvecs @ y


def simulate_burnin_and_work(
    A1: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    params: ProtocolParams,
    rng: np.random.Generator,
    mpemba_K: int = 0,
    *,
    A_start: Optional[np.ndarray] = None,
    A_end: Optional[np.ndarray] = None,
) -> float:
    """
    One realization:
      1) (optional) Mpemba init x(0) with K slowest modes pre-thermalized
      2) thermalize at A_start for t_burn = burnin_mult / lambda_eff(A_start),
         where lambda_eff = lambda_{K+1} if mpemba_K>0 else lambda_1.
      3) drive linearly A(t) = (1-s)A_start + s A_end over tau
      4) compute work W = 1/2 ∫ x^T dA x, discretized as 1/2 Σ x_k^T (A_{k+1}-A_k) x_k

    Speed optimization:
      - In the protocols used here, A(t) is always a linear interpolation between A1 and a2*I.
        Since I commutes with A1, the dynamics is diagonal in the eigenbasis of A1.
      - We therefore evolve the eigen-coordinates y = Q^T x with elementwise updates (O(d) per step),
        instead of dense matvecs (O(d^2) per step).
      - Burn-in at fixed A_start is an Ornstein–Uhlenbeck process and is updated exactly in one shot.
    """
    d = A1.shape[0]
    beta = params.beta
    dt = params.dt
    K = int(max(0, min(mpemba_K, d)))

    # Default to the original forward protocol if not provided.
    if A_start is None:
        A_start = A1
    if A_end is None:
        A_end = params.a2 * np.eye(d)

    # --- identify whether we can use the diagonal-in-eigenbasis fast path ---

    def _is_scalar_identity(M: np.ndarray) -> Optional[float]:
        # Return scalar c if M ≈ c I, else None. Cheap check tailored to our use case.
        if M.shape[0] != M.shape[1]:
            return None
        # check off-diagonals
        off = M - np.diag(np.diag(M))
        if np.any(off != 0.0):
            # allow tiny numerical noise
            if float(np.max(np.abs(off))) > 1e-12:
                return None
        diag = np.diag(M)
        c = float(diag[0])
        if float(np.max(np.abs(diag - c))) > 1e-12:
            return None
        return c

    c_start = _is_scalar_identity(A_start)
    c_end = _is_scalar_identity(A_end)

    # Fast path requires: A_start and A_end are each either A1 or (scalar)*I.
    start_is_A1 = A_start is A1 or np.may_share_memory(A_start, A1) or np.allclose(A_start, A1, atol=1e-12, rtol=1e-12)
    end_is_A1 = A_end is A1 or np.may_share_memory(A_end, A1) or np.allclose(A_end, A1, atol=1e-12, rtol=1e-12)

    can_fast = (start_is_A1 or (c_start is not None)) and (end_is_A1 or (c_end is not None))

    if not can_fast:
        # Fallback to the original dense update (rare; keeps API general).
        noise_std = math.sqrt(2.0 * dt / beta)

        if K >= d:
            lam_eff = float(eigvals[-1])
            t_burn = 0.0
        else:
            lam_eff = float(eigvals[K])  # lambda_{K+1}
            if params.burnin_mode == "mult":
                t_burn = params.burnin_mult / lam_eff
            else:
                if K == 0:
                    E0 = 1.0
                else:
                    num = np.sum(1.0 / eigvals[K:]**2)
                    den = np.sum(1.0 / eigvals**2)
                    E0 = math.sqrt(num / den)
                t_burn = (1.0 / (2.0 * lam_eff)) * math.log(max(E0 / params.burnin_eps, 1.0))

        n_burn = int(math.ceil(t_burn / dt)) if t_burn > 0 else 0

        if params.print_burnin_check and params.burnin_mode == "eps":
            if K == 0:
                E0_dbg = 1.0
            else:
                num_dbg = float(np.sum(1.0 / eigvals[K:]**2))
                den_dbg = float(np.sum(1.0 / eigvals**2))
                E0_dbg = math.sqrt(num_dbg / den_dbg)
            E_t = E0_dbg * math.exp(-2.0 * lam_eff * t_burn) if t_burn > 0 else E0_dbg
            key = (d, K, round(params.burnin_eps, 12), round(t_burn, 12))
            if key not in _BURNIN_PRINTED:
                _BURNIN_PRINTED.add(key)
                print(f"[burn-in check] d={d}, K={K}, lam_eff={lam_eff:.3e}, t_burn={t_burn:.3e} => "
                      f"E(t_burn)={E_t:.3e} (target eps={params.burnin_eps:.3e})")

        x = sample_mpemba_x0(eigvals=eigvals, eigvecs=eigvecs, beta=beta, K=K, rng=rng)

        for _ in range(n_burn):
            xi = rng.normal(size=d)
            x += (-A_start @ x) * dt + noise_std * xi

        n_steps = max(1, int(math.ceil(params.tau / dt)))
        W = 0.0
        for k in range(n_steps):
            s = k / n_steps
            s_next = (k + 1) / n_steps
            A = (1.0 - s) * A_start + s * A_end
            A_next = (1.0 - s_next) * A_start + s_next * A_end
            dA = A_next - A
            W += 0.5 * float(x.T @ dA @ x)
            xi = rng.normal(size=d)
            x += (-A @ x) * dt + noise_std * xi
        return float(W)

    # --- Fast path (diagonal in eigenbasis of A1) ---

    # Work in eigenbasis: y = Q^T x
    # Mpemba init in eigenbasis
    if K == 0:
        y = np.zeros(d, dtype=float)
    else:
        std = np.zeros(d, dtype=float)
        std[:K] = np.sqrt(1.0 / (beta * eigvals[:K]))
        y = std * rng.normal(size=d)

    # Build diagonal "eigenvalues" of A_start and A_end in the A1 eigenbasis.
    if start_is_A1:
        lam_start = eigvals
    else:
        lam_start = np.full(d, float(c_start), dtype=float)

    if end_is_A1:
        lam_end = eigvals
    else:
        lam_end = np.full(d, float(c_end), dtype=float)

    # Effective slowest relaxation rate after Mpemba pre-thermalization
    if K >= d:
        lam_eff = float(lam_start[-1])
        t_burn = 0.0
    else:
        lam_eff = float(lam_start[K])  # lambda_{K+1} of A_start in this basis
        if params.burnin_mode == "mult":
            t_burn = params.burnin_mult / lam_eff
        else:
            if K == 0:
                E0 = 1.0
            else:
                num = float(np.sum(1.0 / (lam_start[K:] ** 2)))
                den = float(np.sum(1.0 / (lam_start ** 2)))
                E0 = math.sqrt(num / den)
            t_burn = (1.0 / (2.0 * lam_eff)) * math.log(max(E0 / params.burnin_eps, 1.0))

    # Optional debug: verify predicted covariance error at t_burn
    if params.print_burnin_check and params.burnin_mode == "eps":
        if K == 0:
            E0_dbg = 1.0
        else:
            num_dbg = float(np.sum(1.0 / (lam_start[K:] ** 2)))
            den_dbg = float(np.sum(1.0 / (lam_start ** 2)))
            E0_dbg = math.sqrt(num_dbg / den_dbg)
        E_t = E0_dbg * math.exp(-2.0 * lam_eff * t_burn) if t_burn > 0 else E0_dbg
        key = (d, K, round(params.burnin_eps, 12), round(t_burn, 12))
        if key not in _BURNIN_PRINTED:
            _BURNIN_PRINTED.add(key)
            print(f"[burn-in check] d={d}, K={K}, lam_eff={lam_eff:.3e}, t_burn={t_burn:.3e} => "
                  f"E(t_burn)={E_t:.3e} (target eps={params.burnin_eps:.3e})")

    # Exact OU burn-in: y(t) = y0*e^{-lam t} + sqrt((1-e^{-2 lam t})/(beta lam)) * N(0,1)
    if t_burn > 0.0:
        expfac = np.exp(-lam_start * t_burn)
        # handle numerical safety if any lam_start is tiny (shouldn't for SPD)
        var = (1.0 - np.exp(-2.0 * lam_start * t_burn)) / (beta * lam_start)
        y = y * expfac + np.sqrt(var) * rng.normal(size=d)

    # Drive A(t) linearly over tau with Euler–Maruyama updates in eigenbasis.
    n_steps = max(1, int(math.ceil(params.tau / dt)))
    noise_std = math.sqrt(2.0 * dt / beta)

    # dA is constant per step for linear schedule: (A_end - A_start)/n_steps
    dlam = (lam_end - lam_start) / float(n_steps)

    W = 0.0
    for k in range(n_steps):
        s = k / n_steps
        # diagonal entries of A at this step
        lam_t = (1.0 - s) * lam_start + s * lam_end

        # Work increment uses y at start of interval (Itô convention)
        W += 0.5 * float(np.dot(dlam, y * y))

        # Euler–Maruyama step: dy = -lam_t * y dt + sqrt(2 dt / beta) * xi
        y += (-lam_t * y) * dt + noise_std * rng.normal(size=d)

    return float(W)


def simulate_burnin_and_work_batch(
    A1: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    params: ProtocolParams,
    rng: np.random.Generator,
    n_traj: int,
    mpemba_K: int = 0,
    *,
    A_start: Optional[np.ndarray] = None,
    A_end: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Vectorized version of simulate_burnin_and_work() that returns an array of works of length n_traj.

    This preserves *all* I/O behavior of the script, but is much faster because it:
      - evolves all trajectories simultaneously in the eigenbasis of A1 (elementwise operations)
      - avoids per-trajectory Python loops and SeedSequence spawning overhead
      - supports reuse of the same simulated work samples for multiple N by prefix-averaging

    If the protocol is not diagonal in the eigenbasis of A1, it falls back to calling the scalar
    simulator in a loop (rare in this script).
    """
    d = A1.shape[0]
    beta = params.beta
    dt = params.dt
    K = int(max(0, min(mpemba_K, d)))

    if n_traj <= 0:
        return np.empty((0,), dtype=float)

    # Defaults match simulate_burnin_and_work
    if A_start is None:
        A_start = A1
    if A_end is None:
        A_end = params.a2 * np.eye(d)

    def _is_scalar_identity(M: np.ndarray) -> Optional[float]:
        if M.shape[0] != M.shape[1]:
            return None
        off = M - np.diag(np.diag(M))
        if np.any(off != 0.0):
            if float(np.max(np.abs(off))) > 1e-12:
                return None
        diag = np.diag(M)
        c = float(diag[0])
        if float(np.max(np.abs(diag - c))) > 1e-12:
            return None
        return c

    c_start = _is_scalar_identity(A_start)
    c_end = _is_scalar_identity(A_end)

    start_is_A1 = A_start is A1 or np.may_share_memory(A_start, A1) or np.allclose(A_start, A1, atol=1e-12, rtol=1e-12)
    end_is_A1 = A_end is A1 or np.may_share_memory(A_end, A1) or np.allclose(A_end, A1, atol=1e-12, rtol=1e-12)

    can_fast = (start_is_A1 or (c_start is not None)) and (end_is_A1 or (c_end is not None))
    if not can_fast:
        # Rare fallback; keeps correctness for arbitrary protocols.
        ss = np.random.SeedSequence(int(rng.integers(0, 2**32 - 1)))
        child = ss.spawn(n_traj)
        out = np.empty(n_traj, dtype=float)
        for i in range(n_traj):
            rng_i = np.random.default_rng(child[i])
            out[i] = simulate_burnin_and_work(
                A1=A1, eigvals=eigvals, eigvecs=eigvecs, params=params, rng=rng_i,
                mpemba_K=mpemba_K, A_start=A_start, A_end=A_end
            )
        return out

    # --- Fast path (diagonal in eigenbasis of A1) ---
    if K == 0:
        y = np.zeros((n_traj, d), dtype=float)
    else:
        std = np.zeros(d, dtype=float)
        std[:K] = np.sqrt(1.0 / (beta * eigvals[:K]))
        y = rng.normal(size=(n_traj, d)) * std[None, :]

    if start_is_A1:
        lam_start = eigvals.astype(float, copy=False)
    else:
        lam_start = np.full(d, float(c_start), dtype=float)

    if end_is_A1:
        lam_end = eigvals.astype(float, copy=False)
    else:
        lam_end = np.full(d, float(c_end), dtype=float)

    if K >= d:
        lam_eff = float(lam_start[-1])
        t_burn = 0.0
    else:
        lam_eff = float(lam_start[K])
        if params.burnin_mode == "mult":
            t_burn = params.burnin_mult / lam_eff
        else:
            if K == 0:
                E0 = 1.0
            else:
                num = float(np.sum(1.0 / (lam_start[K:] ** 2)))
                den = float(np.sum(1.0 / (lam_start ** 2)))
                E0 = math.sqrt(num / den)
            t_burn = (1.0 / (2.0 * lam_eff)) * math.log(max(E0 / params.burnin_eps, 1.0))

    if params.print_burnin_check and params.burnin_mode == "eps":
        if K == 0:
            E0_dbg = 1.0
        else:
            num_dbg = float(np.sum(1.0 / (lam_start[K:] ** 2)))
            den_dbg = float(np.sum(1.0 / (lam_start ** 2)))
            E0_dbg = math.sqrt(num_dbg / den_dbg)
        E_t = E0_dbg * math.exp(-2.0 * lam_eff * t_burn) if t_burn > 0 else E0_dbg
        key = (d, K, round(params.burnin_eps, 12), round(t_burn, 12))
        if key not in _BURNIN_PRINTED:
            _BURNIN_PRINTED.add(key)
            print(f"[burn-in check] d={d}, K={K}, lam_eff={lam_eff:.3e}, t_burn={t_burn:.3e} => "
                  f"E(t_burn)={E_t:.3e} (target eps={params.burnin_eps:.3e})")

    if t_burn > 0.0:
        expfac = np.exp(-lam_start * t_burn)  # (d,)
        var = (1.0 - np.exp(-2.0 * lam_start * t_burn)) / (beta * lam_start)
        y = y * expfac[None, :] + rng.normal(size=(n_traj, d)) * np.sqrt(var)[None, :]

    n_steps = max(1, int(math.ceil(params.tau / dt)))
    noise_std = math.sqrt(2.0 * dt / beta)

    dlam = (lam_end - lam_start) / float(n_steps)  # (d,)

    # Accumulate work per trajectory
    W = np.zeros(n_traj, dtype=float)

    # Drive loop (vectorized over trajectories; only time loop remains)
    for k in range(n_steps):
        s = k / n_steps
        lam_t = (1.0 - s) * lam_start + s * lam_end  # (d,)

        # Work increment: 0.5 * sum_i dlam_i * y_i^2
        W += 0.5 * (y * y) @ dlam

        # Euler–Maruyama in eigenbasis
        y += (-lam_t[None, :] * y) * dt + noise_std * rng.normal(size=(n_traj, d))

    return W


def _auto_a2(A: np.ndarray, params: ProtocolParams) -> float:
    """
    Choose a2 = tr(A) / d  (arithmetic mean of eigenvalues).

    This is the spectrum-free choice for thermodynamic computing:
      - Only the diagonal of A is needed — O(d), no diagonalisation.
      - Centers the work distribution exactly: mean_W = ½(tr A − d·a2) = 0,
        keeping the forward/reverse BAR distributions centred at 0 for any d.
      - Provides a fair benchmark: no spectral information beyond the trace is
        assumed, consistent with the thermodynamic computing setting where the
        full spectrum is not available classically.

    If params.a2 != 1.0, that value is used directly (explicit override for
    backward compatibility and unit tests; 1.0 is the sentinel for auto-scaling).
    """
    if params.a2 != 1.0:
        return float(params.a2)
    d = A.shape[0]
    return float(np.trace(A)) / d


def _safe_dt(dt: float, lam_max: float) -> float:
    """
    Return a dt that keeps the Euler-Maruyama integrator stable:
    require dt * lam_max <= 0.5 (generous stability margin for OU process).
    """
    dt_max = 0.5 / float(lam_max)
    return min(dt, dt_max)


def estimate_logdet_via_jarzynski(
    A1: np.ndarray,
    n_traj: int,
    params: ProtocolParams,
    mpemba_K: int = 0,
) -> Tuple[float, float, float]:
    """
    Returns (logdet_est, logdet_exact, abs_err_logdet).

    mpemba_K:
      - 0: baseline init x(0)=0 and burn-in set by lambda_1
      - K>0: Mpemba init with K slowest modes at equilibrium variance and burn-in set by lambda_{K+1}
    """
    _check_spd(A1, "A1")
    d = A1.shape[0]

    logdet_exact = exact_logdet(A1)

    # Eigendecomposition once per A1
    eigvals, eigvecs = np.linalg.eigh(0.5 * (A1 + A1.T))
    if np.min(eigvals) <= 0:
        return -np.inf, logdet_exact, np.inf

    # a2 = tr(A1)/d — spectrum-free O(d) choice (diagonal sum only).
    a2_eff = _auto_a2(A1, params)
    # Adapt dt for stability (critical for fixed-spectrum large d)
    dt_eff = _safe_dt(params.dt, float(eigvals[-1]))
    params_eff = ProtocolParams(
        beta=params.beta, a2=a2_eff, tau=params.tau, dt=dt_eff,
        burnin_mode=params.burnin_mode, burnin_mult=params.burnin_mult,
        burnin_eps=params.burnin_eps, seed=params.seed,
        print_burnin_check=params.print_burnin_check,
    )

    # det(A2)=a2^d
    logdet_A2 = d * math.log(a2_eff)

    # Build A2 with the effective (auto-scaled) a2
    A2 = a2_eff * np.eye(d)

    # Vectorized work sampling (much faster than per-trajectory loops)
    rng = np.random.default_rng(np.random.SeedSequence(params.seed))
    works = simulate_burnin_and_work_batch(
        A1=A1,
        eigvals=eigvals,
        eigvecs=eigvecs,
        params=params_eff,
        rng=rng,
        n_traj=int(n_traj),
        mpemba_K=mpemba_K,
        A_end=A2,
    )

    jarz = float(np.mean(np.exp(-params_eff.beta * works)))  # estimates exp(-beta ΔF)

    # det(A1)=det(A2)*(jarz)^2  => logdet(A1)=logdet(A2)+2 log(jarz)
    if jarz <= 0.0 or not np.isfinite(jarz):
        logdet_est = -np.inf
        err = np.inf
        return logdet_est, logdet_exact, err

    logdet_est = logdet_A2 + 2.0 * math.log(jarz)
    err = abs(logdet_est - logdet_exact)
    return logdet_est, logdet_exact, err


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid σ(x) = 1 / (1 + exp(x)).

    Uses the identity σ(x) = exp(-logaddexp(0, x)), which is overflow-free
    for all finite x: logaddexp never overflows because it subtracts the max
    internally, and the final exp argument is always ≤ 0.
    """
    return np.exp(-np.logaddexp(0.0, np.asarray(x, dtype=float)))


def _solve_bar_deltaF(
    works_f: np.ndarray,
    works_r: np.ndarray,
    beta: float,
    *,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> float:
    """Bennett Acceptance Ratio (BAR) estimate of free-energy difference ΔF.

    We currently use the equal-sample-size BAR equation, so callers must provide
    the same number of forward and reverse work samples.
    """
    wf = np.asarray(works_f, dtype=float)
    wr = np.asarray(works_r, dtype=float)
    if wf.ndim != 1 or wr.ndim != 1:
        raise ValueError("works_f and works_r must be 1D arrays")
    if wf.size == 0 or wr.size == 0:
        raise ValueError("Need at least one forward and one reverse work sample")
    if wf.size != wr.size:
        raise ValueError("This BAR implementation requires n_fwd == n_rev")
    if beta <= 0:
        raise ValueError("beta must be > 0")

    def f(deltaF: float) -> float:
        a = beta * (wf - deltaF)
        b = beta * (wr + deltaF)
        return float(np.sum(_sigmoid_stable(a)) - np.sum(_sigmoid_stable(b)))

    # Heuristic initial bracket based on the support of work values.
    lo = min(float(np.min(wf)), -float(np.max(wr))) - 50.0 / beta
    hi = max(float(np.max(wf)), -float(np.min(wr))) + 50.0 / beta

    flo = f(lo)
    fhi = f(hi)

    # Expand bracket if needed.
    expand = 0
    while flo * fhi > 0 and expand < 60:
        width = (hi - lo)
        lo -= width
        hi += width
        flo = f(lo)
        fhi = f(hi)
        expand += 1

    if flo * fhi > 0:
        # Fall back: if we cannot bracket, return a reasonable guess.
        # (This should be rare unless the work samples are pathological.)
        return float(np.median(wf) - np.median(wr)) / 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return float(mid)
        if flo * fmid <= 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid

    return float(0.5 * (lo + hi))


def _sample_crooks_work_batches(
    A1: np.ndarray,
    n_traj_fwd: int,
    n_traj_rev: int,
    params: ProtocolParams,
    mpemba_K_fwd: int = 0,
    *,
    eigvals: Optional[np.ndarray] = None,
    eigvecs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate forward and reverse Crooks work samples in batch.

    Both directions are simulated in the eigenbasis of A1. This is the correct
    fast path because A2 = a2 I is diagonal in any basis. The reverse protocol
    therefore uses the same eigensystem as the forward one, with
        A_start = A2, A_end = A1.

    Returns
    -------
    works_f : ndarray, shape (n_traj_fwd,)
        Forward work samples for A1 -> A2.
    works_r : ndarray, shape (n_traj_rev,)
        Reverse work samples for A2 -> A1.
    """
    _check_spd(A1, "A1")
    d = A1.shape[0]

    if eigvals is None or eigvecs is None:
        eigvals, eigvecs = np.linalg.eigh(0.5 * (A1 + A1.T))
    if np.min(eigvals) <= 0:
        raise ValueError("A1 must be SPD with strictly positive eigenvalues")

    # a2 = tr(A1)/d — spectrum-free O(d) choice. Also clamp dt for stability.
    a2_eff = _auto_a2(A1, params)
    dt_eff = _safe_dt(params.dt, float(eigvals[-1]))
    params_eff = ProtocolParams(
        beta=params.beta, a2=a2_eff, tau=params.tau, dt=dt_eff,
        burnin_mode=params.burnin_mode, burnin_mult=params.burnin_mult,
        burnin_eps=params.burnin_eps, seed=params.seed,
        print_burnin_check=params.print_burnin_check,
    )

    A2 = a2_eff * np.eye(d)

    ss = np.random.SeedSequence(params.seed)
    seeds = ss.spawn(2)

    rng_f = np.random.default_rng(seeds[0])
    works_f = simulate_burnin_and_work_batch(
        A1=A1,
        eigvals=eigvals,
        eigvecs=eigvecs,
        params=params_eff,
        rng=rng_f,
        n_traj=int(n_traj_fwd),
        mpemba_K=mpemba_K_fwd,
        A_start=A1,
        A_end=A2,
    )

    rng_r = np.random.default_rng(seeds[1])
    works_r = simulate_burnin_and_work_batch(
        A1=A1,
        eigvals=eigvals,
        eigvecs=eigvecs,
        params=params_eff,
        rng=rng_r,
        n_traj=int(n_traj_rev),
        mpemba_K=0,
        A_start=A2,
        A_end=A1,
    )

    return works_f, works_r, a2_eff


def estimate_logdet_via_crooks(
    A1: np.ndarray,
    n_traj_fwd: int,
    n_traj_rev: int,
    params: ProtocolParams,
    mpemba_K_fwd: int = 0,
) -> Tuple[float, float, float]:
    """Estimate logdet(A1) using Crooks' theorem via a bidirectional BAR estimator.

    Forward protocol:  A_start = A1, A_end = A2 = a2 I
    Reverse protocol:  A_start = A2, A_end = A1

    Crooks gives P_F(W)/P_R(-W) = exp(beta (W - DeltaF)); BAR is the standard
    finite-sample estimator of DeltaF built from the forward and reverse work samples.

    For the quadratic Hamiltonian H(x;A)=1/2 x^T A x,
        ΔF = (1/(2β)) [logdet(A2) - logdet(A1)]
    hence
        logdet(A1) = logdet(A2) - 2β ΔF.
    """
    _check_spd(A1, "A1")
    d = A1.shape[0]

    logdet_exact = exact_logdet(A1)

    K_eff = _effective_mpemba_K(mpemba_K_fwd, d, context="single-run Crooks estimate") if mpemba_K_fwd > 0 else 0

    try:
        works_f, works_r, a2_eff = _sample_crooks_work_batches(
            A1=A1,
            n_traj_fwd=n_traj_fwd,
            n_traj_rev=n_traj_rev,
            params=params,
            mpemba_K_fwd=K_eff,
        )
        deltaF = _solve_bar_deltaF(works_f, works_r, params.beta)
    except ValueError:
        return -np.inf, logdet_exact, np.inf

    logdet_A2 = d * math.log(a2_eff)
    logdet_est = logdet_A2 - 2.0 * params.beta * deltaF
    err = abs(logdet_est - logdet_exact)
    return float(logdet_est), float(logdet_exact), float(err)


# Backward-compatible alias: BAR is the numerical Crooks estimator used here.
def estimate_logdet_via_bar(*args, **kwargs):
    return estimate_logdet_via_crooks(*args, **kwargs)


def run_convergence_study(
    d_list: List[int],
    n_list: List[int],
    *,
    ensemble: str,
    n_A: int,
    wishart_m_factor: float,
    wishart_ridge: float,
    alpha_min: float,
    alpha_max: float,
    base_seed: int,
    proto: ProtocolParams,
    mpemba_K: int = 0,
    estimator: str = "crooks",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run convergence curves for a given A1 ensemble.

    Returns:
      Ns: (nN,) array of trajectory counts
      Ds: (nD,) array of dimensions
      abs_eps0: (nD, nN, nA) baseline absolute logdet errors
      abs_epsK: (nD, nN, nA) Mpemba-K absolute logdet errors (all inf if mpemba_K<=0)
      rel_eps0: (nD, nN, nA) baseline relative logdet errors
      rel_epsK: (nD, nN, nA) Mpemba-K relative logdet errors (all inf if mpemba_K<=0)
      tburn0: (nD, nA) baseline burn-in times used (continuous time)
      tburnK: (nD, nA) Mpemba-K burn-in times used (continuous time; 0 if K>=d)
    """
    Ds = np.array(list(map(int, d_list)), dtype=int)
    Ns = np.array(list(map(int, n_list)), dtype=int)

    abs_eps0 = np.empty((len(Ds), len(Ns), n_A), dtype=float)
    abs_epsK = np.full((len(Ds), len(Ns), n_A), np.inf, dtype=float)
    rel_eps0 = np.empty((len(Ds), len(Ns), n_A), dtype=float)
    rel_epsK = np.full((len(Ds), len(Ns), n_A), np.inf, dtype=float)

    tburn0 = np.empty((len(Ds), n_A), dtype=float)
    tburnK = np.full((len(Ds), n_A), np.inf, dtype=float)

    master = np.random.SeedSequence(base_seed)
    seeds_A = master.spawn(len(Ds) * n_A)

    total_tasks = len(Ds) * n_A
    desc = f"{ensemble} matrices"
    if tqdm is not None:
        pbar = tqdm(total=total_tasks, desc=desc, unit="A")
    else:
        pbar = None
        print(f"[info] tqdm not installed; running {total_tasks} matrices without progress bar.")

    idx = 0
    for i_d, d in enumerate(Ds):
        for j in range(n_A):
            rng_A = np.random.default_rng(seeds_A[idx])
            idx += 1

            A1 = make_A1(
                ensemble=ensemble,
                d=d,
                rng=rng_A,
                wishart_m_factor=wishart_m_factor,
                wishart_ridge=wishart_ridge,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
            _check_spd(A1, "A1")
            logdet_exact = exact_logdet(A1)

            # Burn-in time estimate (for x-axis "time")
            eigvals, eigvecs = np.linalg.eigh(0.5 * (A1 + A1.T))
            lam1 = float(eigvals[0])
            if proto.burnin_mode == "mult":
                tburn0[i_d, j] = proto.burnin_mult / lam1
            else:
                tburn0[i_d, j] = (1.0 / (2.0 * lam1)) * math.log(1.0 / proto.burnin_eps)

            if mpemba_K > 0:
                K = _effective_mpemba_K(mpemba_K, d, context="convergence study")
                lam_eff = float(eigvals[K])  # lambda_{K+1}
                if proto.burnin_mode == "mult":
                    tburnK[i_d, j] = proto.burnin_mult / lam_eff
                else:
                    num = np.sum(1.0 / eigvals[K:] ** 2)
                    den = np.sum(1.0 / eigvals ** 2)
                    E0 = math.sqrt(num / den)
                    tburnK[i_d, j] = (1.0 / (2.0 * lam_eff)) * math.log(max(E0 / proto.burnin_eps, 1.0))

            # Deterministic but different seeds for baseline vs Mpemba (and across A1)
            seed_traj0 = int(rng_A.integers(0, 2**32 - 1))
            seed_trajK = int(rng_A.integers(0, 2**32 - 1))

            # a2 = tr(A1)/d — spectrum-free O(d) choice. Also clamp dt.
            a2_eff = _auto_a2(A1, proto)
            dt_eff = _safe_dt(proto.dt, float(eigvals[-1]))
            logdet_A2 = d * math.log(a2_eff)

            # Rebuild params with effective a2 and dt
            params0 = ProtocolParams(
                beta=proto.beta, a2=a2_eff, tau=proto.tau, dt=dt_eff,
                burnin_mode=proto.burnin_mode, burnin_mult=proto.burnin_mult,
                burnin_eps=proto.burnin_eps, seed=seed_traj0,
            )
            paramsK = ProtocolParams(
                beta=proto.beta, a2=a2_eff, tau=proto.tau, dt=dt_eff,
                burnin_mode=proto.burnin_mode, burnin_mult=proto.burnin_mult,
                burnin_eps=proto.burnin_eps, seed=seed_trajK,
            )

            # --- speed: reuse one simulation up to Nmax for all N in Ns ---
            Nmax = int(np.max(Ns)) if Ns.size else 0
            works0_all = None
            worksK_all = None
            works0_f_all = None
            works0_r_all = None
            worksK_f_all = None
            worksK_r_all = None
            if Nmax > 0 and estimator == "jarzynski":
                # Reuse eigvals/eigvecs already computed above
                A2 = a2_eff * np.eye(d)

                rng0 = np.random.default_rng(np.random.SeedSequence(params0.seed))
                works0_all = simulate_burnin_and_work_batch(
                    A1=A1, eigvals=eigvals, eigvecs=eigvecs,
                    params=params0, rng=rng0, n_traj=Nmax, mpemba_K=0, A_end=A2
                )
                if mpemba_K > 0:
                    rngK = np.random.default_rng(np.random.SeedSequence(paramsK.seed))
                    worksK_all = simulate_burnin_and_work_batch(
                        A1=A1, eigvals=eigvals, eigvecs=eigvecs,
                        params=paramsK, rng=rngK, n_traj=Nmax, mpemba_K=K, A_end=A2
                    )

                exp0 = np.exp(-params0.beta * works0_all)
                csum0 = np.cumsum(exp0)
                if worksK_all is not None:
                    expK = np.exp(-paramsK.beta * worksK_all)
                    csumK = np.cumsum(expK)

            elif Nmax > 0 and estimator in ("bar", "crooks"):
                works0_f_all, works0_r_all, _ = _sample_crooks_work_batches(
                    A1=A1, n_traj_fwd=Nmax, n_traj_rev=Nmax, params=params0,
                    mpemba_K_fwd=0, eigvals=eigvals, eigvecs=eigvecs,
                )
                if mpemba_K > 0:
                    worksK_f_all, worksK_r_all, _ = _sample_crooks_work_batches(
                        A1=A1, n_traj_fwd=Nmax, n_traj_rev=Nmax, params=paramsK,
                        mpemba_K_fwd=K, eigvals=eigvals, eigvecs=eigvecs,
                    )

            for i_n, N in enumerate(Ns):
                N = int(N)
                if estimator == "jarzynski":
                    if works0_all is not None:
                        # Prefix-mean Jarzynski estimator using the same underlying work samples.
                        jarz0 = float(csum0[N-1] / float(N))
                        logdet0 = logdet_A2 + 2.0 * math.log(jarz0) if (jarz0 > 0 and np.isfinite(jarz0)) else -np.inf
                    else:
                        logdet0 = estimate_logdet_via_jarzynski(A1, n_traj=N, params=params0, mpemba_K=0)[0]
                    abs_eps0[i_d, i_n, j], rel_eps0[i_d, i_n, j] = logdet_errors(logdet_exact, logdet0)

                    if mpemba_K > 0:
                        if worksK_all is not None:
                            jarzK = float(csumK[N-1] / float(N))
                            logdetK = logdet_A2 + 2.0 * math.log(jarzK) if (jarzK > 0 and np.isfinite(jarzK)) else -np.inf
                        else:
                            logdetK = estimate_logdet_via_jarzynski(A1, n_traj=N, params=paramsK, mpemba_K=mpemba_K)[0]
                        abs_epsK[i_d, i_n, j], rel_epsK[i_d, i_n, j] = logdet_errors(logdet_exact, logdetK)
                elif estimator in ("bar", "crooks"):
                    if works0_f_all is not None and works0_r_all is not None:
                        deltaF0 = _solve_bar_deltaF(works0_f_all[:N], works0_r_all[:N], params0.beta)
                        logdet0 = logdet_A2 - 2.0 * params0.beta * deltaF0
                    else:
                        logdet0 = estimate_logdet_via_crooks(A1, n_traj_fwd=N, n_traj_rev=N, params=params0, mpemba_K_fwd=0)[0]
                    abs_eps0[i_d, i_n, j], rel_eps0[i_d, i_n, j] = logdet_errors(logdet_exact, logdet0)
                    if mpemba_K > 0:
                        if worksK_f_all is not None and worksK_r_all is not None:
                            deltaFK = _solve_bar_deltaF(worksK_f_all[:N], worksK_r_all[:N], paramsK.beta)
                            logdetK = logdet_A2 - 2.0 * paramsK.beta * deltaFK
                        else:
                            logdetK = estimate_logdet_via_crooks(A1, n_traj_fwd=N, n_traj_rev=N, params=paramsK, mpemba_K_fwd=mpemba_K)[0]
                        abs_epsK[i_d, i_n, j], rel_epsK[i_d, i_n, j] = logdet_errors(logdet_exact, logdetK)
                else:
                    raise ValueError("estimator must be 'jarzynski', 'crooks', or 'bar'")

            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()

    return Ns, Ds, abs_eps0, abs_epsK, rel_eps0, rel_epsK, tburn0, tburnK


def _summary_stats(e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(e, axis=1)
    q25 = np.nanquantile(e, 0.25, axis=1)
    q75 = np.nanquantile(e, 0.75, axis=1)
    return med, q25, q75


def _parse_float_list(csv: str) -> List[float]:
    """Parse a comma-separated list of floats."""
    out: List[float] = []
    for tok in csv.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out


def _interp_N_for_eps(N0: float, e0: float, N1: float, e1: float, e_target: float) -> float:
    """Log-log interpolation for N_epsilon between two bracketing points.

    Assumes (locally) a power-law dependence eps ~ N^{-b}, i.e. log(eps) is approximately
    linear in log(N). Returns an interpolated N such that eps(N)=e_target.

    Falls back to N1 if the two errors are (numerically) equal.
    """
    tiny = 1e-300
    e0 = max(float(e0), tiny)
    e1 = max(float(e1), tiny)
    e_target = max(float(e_target), tiny)

    x0, x1 = math.log(float(N0)), math.log(float(N1))
    y0, y1 = math.log(e0), math.log(e1)
    yT = math.log(e_target)

    if abs(y1 - y0) < 1e-15:
        return float(N1)

    xT = x0 + (x1 - x0) * (yT - y0) / (y1 - y0)
    return float(math.exp(xT))


def _compute_T_epsilon(
    Ns: np.ndarray,
    eps: np.ndarray,     # shape (nN, nA)
    tburn: np.ndarray,   # shape (nA,)
    tau: float,
    eps_target: float,
    stat: str = "median",
    interp: str = "powerlaw",
) -> float:
    """
    Compute the minimum total compute time T_tot = N*(t_burn + tau) such that the
    chosen statistic (median by default) of eps_log across A-realizations is <= eps_target.

    interp options:
      - 'none'     : snap to first simulated N that crosses eps_target.
      - 'loglog'   : log-log interpolation between the two bracketing N points
                     (or extrapolation if eps_target is already met at N[0]).
      - 'powerlaw' : fit log(eps) = a + b*log(N) globally across all N points,
                     then invert analytically. S_det is independent of eps_target
                     by construction.

    Returns np.nan if the threshold is never reached.
    """
    if stat not in ("median", "mean"):
        raise ValueError("stat must be 'median' or 'mean'")
    if interp not in ("none", "loglog", "powerlaw"):
        raise ValueError("interp must be 'none', 'loglog', or 'powerlaw'")
    if eps.ndim != 2:
        raise ValueError("eps must have shape (nN, nA)")
    if tburn.ndim != 1:
        raise ValueError("tburn must have shape (nA,)")

    if stat == "median":
        eps_stat = np.nanmedian(eps, axis=1)  # shape (nN,)
        tb = float(np.nanmedian(tburn))
    else:
        eps_stat = np.nanmean(eps, axis=1)
        tb = float(np.nanmean(tburn))

    if interp == "powerlaw":
        mask = np.isfinite(eps_stat) & (eps_stat > 0)
        if mask.sum() < 2:
            return float("nan")
        log_N = np.log(Ns[mask].astype(float))
        log_e = np.log(eps_stat[mask])
        b, a = np.polyfit(log_N, log_e, 1)  # log_e = a + b * log_N
        if b >= 0:
            return float("nan")
        N_star = float(np.exp((np.log(float(eps_target)) - a) / b))
        return N_star * (tb + float(tau))

    idx = np.where(eps_stat <= eps_target)[0]
    if idx.size == 0:
        return float("nan")

    i = int(idx[0])
    if interp == "none":
        N_star = float(Ns[i])
    elif i == 0 and len(Ns) >= 2:
        # Target is already met at the first grid point: extrapolate backwards
        # using the same log-log logic as the bracketed forward case.
        N_star = _interp_N_for_eps(float(Ns[0]), float(eps_stat[0]),
                                   float(Ns[1]), float(eps_stat[1]),
                                   eps_target)
        # Safety cap: don't extrapolate above Ns[0] if median is non-monotone
        N_star = min(N_star, float(Ns[0]))
    else:
        N0, N1 = float(Ns[i - 1]), float(Ns[i])
        e0, e1 = float(eps_stat[i - 1]), float(eps_stat[i])
        N_star = _interp_N_for_eps(N0, e0, N1, e1, eps_target)

    return N_star * (tb + float(tau))

def plot_convergence(
    Ns: np.ndarray,
    Ds: np.ndarray,
    err0: np.ndarray,
    errK: np.ndarray,
    tburn0: np.ndarray,
    tburnK: np.ndarray,
    tau: float,
    mpemba_K: int,
    xaxis: str,
    outpath: str,
    show: bool,
    ylabel: str = r"$\epsilon_{\log} = |\log\det(A) - \log\det(\hat A)|$",
    speedup_eps_list: Optional[List[float]] = None,
    speedup_stat: str = "median",
    speedup_interp: str = "powerlaw",
    title: Optional[str] = None,
) -> None:
    """
    Plot median eps_log vs compute budget with IQR error bars across A realizations.
    Styling conventions:
      - same color for baseline (K=0) and Mpemba (K=mpemba_K) at fixed dimension d
      - baseline: solid line, Mpemba: dashed line
      - annotate each point with N_traj
    """
    plt.figure(figsize=(7.6, 4.4))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get(
        "color", ["C0", "C1", "C2", "C3", "C4", "C5"]
    )

    for i_d, d in enumerate(Ds):
        color = colors[i_d % len(colors)]

        # Baseline (K=0)
        e0 = err0[i_d, :, :]  # (nN, nA)
        med0, q250, q750 = _summary_stats(e0)
        yerr0 = np.vstack([med0 - q250, q750 - med0])
        x0 = Ns if xaxis == "N" else Ns * (float(np.nanmedian(tburn0[i_d, :])) + tau)

        plt.errorbar(
            x0, med0, yerr=yerr0,
            marker="o", linewidth=1.2, capsize=3,
            linestyle="-", color=color,
            label=f"{d} (K=0)",
        )
        for x_pt, y_pt, N_pt in zip(x0, med0, Ns):
            plt.annotate(
                fr"$N_{{\rm traj}}={int(N_pt)}$",
                (x_pt, y_pt),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                alpha=0.7,
            )

        # Mpemba curve (K=mpemba_K), if requested
        if mpemba_K > 0:
            eK = errK[i_d, :, :]
            medK, q25K, q75K = _summary_stats(eK)
            yerrK = np.vstack([medK - q25K, q75K - medK])
            xK = Ns if xaxis == "N" else Ns * (float(np.nanmedian(tburnK[i_d, :])) + tau)

            plt.errorbar(
                xK, medK, yerr=yerrK,
                marker="s", linewidth=1.2, capsize=3,
                linestyle="--", color=color,
                label=f"{d} (K={mpemba_K})",
            )
            # offset in opposite direction to reduce overlap
            for x_pt, y_pt, N_pt in zip(xK, medK, Ns):
                plt.annotate(
                    fr"$N_{{\rm traj}}={int(N_pt)}$",
                    (x_pt, y_pt),
                    textcoords="offset points",
                    xytext=(4, -10),
                    fontsize=7,
                    alpha=0.7,
                )


    # ---- Inset: "right" determinant speedup S_eps(d) at fixed target error eps ----
    if mpemba_K > 0 and speedup_eps_list:
        # Compute S_eps(d) for each target eps using total compute time T_tot = N*(t_burn+tau)
        ax = plt.gca()
        axins = ax.inset_axes([0.08, 0.08, 0.35, 0.35])

        for eps_target in speedup_eps_list:
            S_list = []
            for i_d, d in enumerate(Ds):
                T0 = _compute_T_epsilon(Ns, err0[i_d, :, :], tburn0[i_d, :], tau, eps_target, stat=speedup_stat, interp=speedup_interp)
                TK = _compute_T_epsilon(Ns, errK[i_d, :, :], tburnK[i_d, :], tau, eps_target, stat=speedup_stat, interp=speedup_interp)
                if (not np.isfinite(T0)) or (not np.isfinite(TK)) or TK <= 0:
                    S_list.append(np.nan)
                else:
                    S_list.append(T0 / TK)
            axins.plot(Ds, S_list, marker="o", linewidth=1.2, label=fr"$\epsilon={eps_target:g}$")

        axins.set_xlabel(r"$d$", fontsize=8)
        axins.set_ylabel(r"$S_{\epsilon}$", fontsize=8)
        axins.tick_params(axis="both", which="both", labelsize=8)
        axins.set_title(fr"Speedup ($K={mpemba_K}$)", fontsize=8)
        axins.legend(frameon=False, fontsize=7, loc="best")
        # Keep inset tidy
        axins.grid(False)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(
        r"Number of trajectories $N_{\mathrm{traj}}$"
        if xaxis == "N"
        else r"Total compute time $T_{\mathrm{tot}} = N_{\mathrm{traj}}(t_{\mathrm{burn}}+\tau)$ (arb. units)"
    )
    plt.ylabel(ylabel)

    if title is not None:
        plt.title(title)

    plt.legend(title="Dimension", frameon=True, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    print(f"[saved] {outpath}")

    if show:
        plt.show()
    else:
        plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Crooks/BAR determinant estimator for Wishart SPD A1 + convergence plot.")
    ap.add_argument("--seed", type=int, default=0, help="Base RNG seed for reproducibility.")

    # Wishart parameters (single run)
    ap.add_argument("--d", type=int, default=10, help="Single-run dimension d.")
    ap.add_argument("--m", type=int, default=None, help="Single-run Wishart m (default: ceil(wishart_m_factor * d); must be >= d).")
    ap.add_argument("--ridge", type=float, default=0.2, help="Ridge added to Wishart to ensure conditioning.")

    # Protocol parameters (fast defaults)
    ap.add_argument("--beta", type=float, default=1.0, help="Inverse temperature beta.")
    ap.add_argument("--a2", type=float, default=1.0, help="A2 = a2 * I (reference).")
    ap.add_argument("--tau", type=float, default=0.2, help="Driving duration tau for A1 -> A2 (fast default).")
    ap.add_argument("--dt", type=float, default=5e-3, help="Euler-Maruyama time step (fast default).")
    ap.add_argument("--burnin_mode", type=str, default="eps", choices=["mult","eps"],
                    help="Burn-in criterion: fixed-multiplier or eps-based covariance convergence.")
    ap.add_argument("--burnin_mult", type=float, default=5.0,
                    help="Burn-in multiplier (used if burnin_mode=mult).")
    ap.add_argument("--burnin_eps", type=float, default=1e-2,
                    help="Target relative covariance error epsilon (used if burnin_mode=eps).")
    ap.add_argument("--print_burnin_check", action="store_true",
                    help="Print predicted covariance error E(t_burn) and compare to burnin_eps (debug).")

    # Single-run estimator
    ap.add_argument("--n_traj", type=int, default=500, help="Number of trajectories for a single estimate.")
    ap.add_argument("--n_traj_rev", type=int, default=None,
                    help="(Crooks/BAR) Number of reverse trajectories. Defaults to --n_traj.")
    ap.add_argument("--estimator", type=str, default="crooks", choices=["crooks", "bar", "jarzynski"],
                    help="Free-energy estimator: Crooks/BAR (bidirectional, recommended) or Jarzynski (forward only).")

    # Mpemba option
    ap.add_argument("--mpemba_K", type=int, default=0,
                    help="Pre-thermalize K slowest modes for A1 initialization (0 disables).")

    # Convergence study
    ap.add_argument("--convergence", action="store_true", help="Run convergence study and make plot.")
    ap.add_argument("--ensembles", type=str, default="wishart,fixed", help="Comma-separated: wishart,fixed")
    ap.add_argument("--d_list", type=str, default="5", help="Comma-separated dimensions for convergence.")

    # Matrix ensemble parameters (match lyapunov_covariance.py)
    ap.add_argument("--wishart_m_factor", type=float, default=1.2, help="m = ceil(wishart_m_factor * d)")
    ap.add_argument("--wishart_ridge", type=float, default=0.0, help="Ridge added to Wishart matrix.")
    ap.add_argument("--wishart_trials", type=int, default=10, help="Number of Wishart matrix realizations (A1 draws).")

    ap.add_argument("--alpha_min", type=float, default=0.5, help="Min eigenvalue for fixed-spectrum ensemble.")
    ap.add_argument("--alpha_max", type=float, default=0.5, help="Slope for max eigenvalue: max = alpha_max * d.")
    ap.add_argument("--fixed_trials", type=int, default=10, help="Number of fixed-spectrum realizations (Haar eigenvectors).")
    ap.add_argument("--n_list", type=str, default="100,1000, 5000", help="Comma-separated trajectory counts.")
    ap.add_argument("--plot_out", type=str, default="crooks_convergence_abs.png", help="Output PNG for absolute-error convergence plot.")
    ap.add_argument("--plot_out_rel", type=str, default="crooks_convergence_rel.png", help="Output PNG for relative-error convergence plot.")
    ap.add_argument("--xaxis", type=str, default="time", choices=["N","time"],
                    help="Convergence plot x-axis: N (trajectories) or time (N*(t_burn+tau)).")
    ap.add_argument("--speedup_eps_list", type=str, default="1e-1,1e-2,1e-3",
                    help="Comma-separated target errors eps for speedup inset: S_eps(d)=T_eps(K=0)/T_eps(K).")
    ap.add_argument("--speedup_stat", type=str, default="median", choices=["median","mean"],
                    help="Statistic across A-realizations used to define eps crossing (median recommended).")
    ap.add_argument("--speedup_interp", type=str, default="powerlaw", choices=["powerlaw","loglog","none"],
                    help="Method to extract N_star for the speedup inset. 'powerlaw' (default): global log-log fit, S_det independent of eps by construction. 'loglog': local two-point interpolation. 'none': snap to nearest N.")
    ap.add_argument("--show", action="store_true", help="Show plot window (in addition to saving PNG).")

    ap.add_argument("--save_npz", action="store_true",
                    help="If set (recommended for cluster runs), save plotted data arrays to an .npz file per ensemble.")
    ap.add_argument("--npz_out", type=str, default=None,
                    help="Optional base path/prefix for .npz outputs (suffix _wishart.npz / _fixed.npz will be appended). If not set, derived from --plot_out.")

    args = ap.parse_args()

    # Default single-run Wishart aspect ratio: m = ceil(wishart_m_factor * d) unless explicitly provided.
    if args.m is None:
        args.m = int(math.ceil(args.wishart_m_factor * args.d))

    if args.beta <= 0:
        raise ValueError("beta must be > 0")
    if args.a2 <= 0:
        raise ValueError("a2 must be > 0")
    if args.tau <= 0:
        raise ValueError("tau must be > 0")
    if args.dt <= 0:
        raise ValueError("dt must be > 0")
    if args.n_traj <= 0:
        raise ValueError("n_traj must be > 0")
    if args.m < args.d:
        raise ValueError("m must be >= d for the single-run Wishart matrix")
    if args.wishart_m_factor < 1.0:
        raise ValueError("wishart_m_factor must be >= 1 so that m = ceil(wishart_m_factor*d) >= d in convergence runs")
    if args.mpemba_K < 0:
        raise ValueError("mpemba_K must be >= 0")

    ensembles = [e.strip() for e in args.ensembles.split(",") if e.strip()]
    if args.convergence and "wishart" in ensembles:
        conv_m = int(math.ceil(args.wishart_m_factor * args.d))
        if args.m != conv_m or abs(args.ridge - args.wishart_ridge) > 0.0:
            warnings.warn(
                "Single-run Wishart parameters differ from convergence-study Wishart parameters; "
                f"single run uses m={args.m}, ridge={args.ridge}, while convergence uses m=ceil({args.wishart_m_factor}*d)={conv_m}, ridge={args.wishart_ridge}.",
                RuntimeWarning,
                stacklevel=2,
            )

    proto = ProtocolParams(
        beta=args.beta,
        a2=args.a2,
        tau=args.tau,
        dt=args.dt,
        burnin_mode=args.burnin_mode,
        burnin_mult=args.burnin_mult,
        burnin_eps=args.burnin_eps,
        print_burnin_check=args.print_burnin_check,
        seed=args.seed + 999,
    )

    # ----- Single run (quick sanity check) -----
    rng = np.random.default_rng(args.seed)
    A1 = make_J_wishart(d=args.d, m=args.m, rng=rng, ridge=args.ridge)
    _check_spd(A1, "A1")

    if args.estimator == "jarzynski":
        logdet_est, logdet_exact, err = estimate_logdet_via_jarzynski(
            A1, n_traj=int(args.n_traj), params=proto, mpemba_K=args.mpemba_K
        )
    else:
        nrev = int(args.n_traj if args.n_traj_rev is None else args.n_traj_rev)
        if nrev <= 0:
            raise ValueError("n_traj_rev must be > 0")
        if nrev != int(args.n_traj):
            raise ValueError("Current BAR implementation requires n_traj_rev == n_traj")
        logdet_est, logdet_exact, err = estimate_logdet_via_crooks(
            A1, n_traj_fwd=int(args.n_traj), n_traj_rev=nrev, params=proto, mpemba_K_fwd=args.mpemba_K
        )

    w = np.linalg.eigvalsh(A1)
    print(f"A1: d={args.d}, m={args.m}, ridge={args.ridge}, "
          f"min_eig={np.min(w):.3e}, max_eig={np.max(w):.3e}, cond~{np.max(w)/np.min(w):.3e}")
    abs_err, rel_err = logdet_errors(logdet_exact, logdet_est)

    print("Single estimate:")
    print(f"  mpemba_K       = {args.mpemba_K}")
    print(f"  logdet_exact   = {logdet_exact:.6f}")
    print(f"  logdet_est     = {logdet_est:.6f}")
    print(f"  abs_err(logdet)= {abs_err:.3e}")
    print(f"  rel_err(logdet)= {rel_err:.3e}")

    # ----- Convergence study + plot -----
    if args.convergence:
        d_list = [int(x.strip()) for x in args.d_list.split(",") if x.strip()]

        # Create a run-specific folder that encodes key parameters (helps distinguish cluster runs)
        if args.save_npz:
            # Deterministic run folder name: encodes (almost) all CLI parameters.
            # No timestamp: identical parameters -> identical folder path.
            def _san(v):
                if isinstance(v, bool):
                    return "1" if v else "0"
                if v is None:
                    return "None"
                if isinstance(v, float):
                    return f"{v:g}"
                s = str(v)
                # keep names filesystem-friendly
                for ch in ["/", " ", ":"]:
                    s = s.replace(ch, "-")
                s = s.replace(",", "-")
                return s

            ad = dict(vars(args))
            # Exclude output/display toggles from the run folder ID
            for k in ("show", "save_npz", "plot_out", "plot_out_rel", "npz_out"):
                ad.pop(k, None)
            # Deterministic short run folder name + hash (avoids OS path-length limits)
            import json, hashlib
            param_str = json.dumps(ad, sort_keys=True, default=str)
            param_hash = hashlib.md5(param_str.encode("utf-8")).hexdigest()[:8]
            # Dimension tag: this script supports either a single-run dimension (--d)
            # or a convergence sweep over a comma-separated list (--d_list).
            if hasattr(args, "d_list") and args.d_list is not None:
                try:
                    _ds = [int(x.strip()) for x in str(args.d_list).split(",") if x.strip()]
                except Exception:
                    _ds = []
                if len(_ds) == 1:
                    d_tag = f"d={_ds[0]}"
                elif len(_ds) > 1:
                    # Keep it short but informative; full list is stored in params.json
                    d_tag = f"d={min(_ds)}-{max(_ds)}-n{len(_ds)}"
                else:
                    d_tag = f"dlist={_san(args.d_list)}"
            else:
                d_tag = f"d={getattr(args, 'd', 'NA')}"
            if args.burnin_mode == "eps":
                burn_tag = f"burn=eps{args.burnin_eps:g}"
            else:
                burn_tag = f"burn=mult{args.burnin_mult:g}"
            run_id = "__".join([
                f"ens={_san(args.ensembles)}",
                d_tag,
                f"K={args.mpemba_K}",
                burn_tag,
                f"tau={args.tau:g}",
                f"seed={args.seed}",
                f"hash={param_hash}",
            ])
            data_dir = os.path.join("saved_data", run_id)
            os.makedirs(data_dir, exist_ok=True)
            # Save full parameters for reproducibility
            with open(os.path.join(data_dir, "params.json"), "w") as f:
                json.dump(ad, f, indent=2, sort_keys=True)
            print(f"[save directory] {data_dir}")
        else:
            data_dir = None
        n_list = [int(x.strip()) for x in args.n_list.split(",") if x.strip()]
        if any(d <= 0 for d in d_list):
            raise ValueError("All d in d_list must be positive.")
        if any(n <= 0 for n in n_list):
            raise ValueError("All N in n_list must be positive.")

        ensembles = [e.strip() for e in args.ensembles.split(",") if e.strip()]
        for ens in ensembles:
            if ens not in ("wishart", "fixed"):
                raise ValueError(f"Unknown ensemble '{ens}'. Use wishart,fixed.")

            nA = args.wishart_trials if ens == "wishart" else args.fixed_trials
            if nA <= 0:
                raise ValueError("Number of trials must be > 0 (wishart_trials / fixed_trials).")

            # ensemble-specific output path
            root, ext = os.path.splitext(args.plot_out)
            outpath = f"{root}_{ens}{ext}" if len(ensembles) > 1 else args.plot_out
            outpath = os.path.join(data_dir, os.path.basename(outpath)) if data_dir is not None else outpath

            Ns, Ds, abs_eps0, abs_epsK, rel_eps0, rel_epsK, tburn0, tburnK = run_convergence_study(
                d_list=d_list,
                n_list=n_list,
                ensemble=ens,
                n_A=nA,
                wishart_m_factor=args.wishart_m_factor,
                wishart_ridge=args.wishart_ridge,
                alpha_min=args.alpha_min,
                alpha_max=args.alpha_max,
                base_seed=args.seed + 2024 + (0 if ens == "wishart" else 9999),
                proto=proto,
                mpemba_K=args.mpemba_K,
                estimator=args.estimator,
            )


            # Save arrays used for plotting (useful for cluster runs: compute first, plot later)
            if args.save_npz:
                # Choose filename and save into the run folder (if enabled)
                if args.npz_out is not None:
                    fname = f"{args.npz_out}_{ens}.npz"
                else:
                    fname = f"convergence_{ens}.npz"
                if data_dir is not None:
                    npz_path = os.path.join(data_dir, fname)
                else:
                    npz_path = fname
                np.savez_compressed(
                    npz_path,
                    ensemble=ens,
                    Ns=Ns,
                    Ds=Ds,
                    abs_eps0=abs_eps0,
                    abs_epsK=abs_epsK,
                    rel_eps0=rel_eps0,
                    rel_epsK=rel_epsK,
                    tburn0=tburn0,
                    tburnK=tburnK,
                    tau=float(proto.tau),
                    mpemba_K=int(args.mpemba_K),
                    burnin_mode=str(args.burnin_mode),
                    burnin_eps=float(args.burnin_eps),
                    beta=float(proto.beta),
                    speedup_eps_list=str(args.speedup_eps_list),
                    speedup_stat=str(args.speedup_stat),
                    speedup_interp=str(args.speedup_interp),
                )
                print(f"[saved] {npz_path}")

            plot_convergence(
                Ns,
                Ds,
                abs_eps0,
                abs_epsK,
                tburn0,
                tburnK,
                tau=proto.tau,
                mpemba_K=args.mpemba_K,
                xaxis=args.xaxis,
                outpath=outpath,
                show=args.show,
                ylabel=r"$\epsilon_{\log}^{\rm abs} = |\log\det(A) - \log\det(\hat A)|$",
                speedup_eps_list=_parse_float_list(args.speedup_eps_list),
                speedup_stat=args.speedup_stat,
                speedup_interp=args.speedup_interp,
                title=(("Wishart" if ens == "wishart" else "Fixed") + " — absolute log-error"),
            )

            root_rel, ext_rel = os.path.splitext(args.plot_out_rel)
            outpath_rel = f"{root_rel}_{ens}{ext_rel}" if len(ensembles) > 1 else args.plot_out_rel
            outpath_rel = os.path.join(data_dir, os.path.basename(outpath_rel)) if data_dir is not None else outpath_rel

            plot_convergence(
                Ns,
                Ds,
                rel_eps0,
                rel_epsK,
                tburn0,
                tburnK,
                tau=proto.tau,
                mpemba_K=args.mpemba_K,
                xaxis=args.xaxis,
                outpath=outpath_rel,
                show=args.show,
                ylabel=r"$\epsilon_{\log}^{\rm rel} = |\log\det(A) - \log\det(\hat A)| / |\log\det(A)|$",
                speedup_eps_list=_parse_float_list(args.speedup_eps_list),
                speedup_stat=args.speedup_stat,
                speedup_interp=args.speedup_interp,
                title=(("Wishart" if ens == "wishart" else "Fixed") + " — relative log-error"),
            )


if __name__ == "__main__":
    main()
