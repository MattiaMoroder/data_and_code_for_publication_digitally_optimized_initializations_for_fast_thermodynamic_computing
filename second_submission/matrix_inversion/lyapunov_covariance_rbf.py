#!/usr/bin/env python3
"""
Deterministic Lyapunov (covariance) dynamics for thermodynamic computing /
Mpemba pre-thermalization -- RBF Gram ensemble version.

Same pipeline as lyapunov_covariance.py (closed-form eigenbasis error,
bisection first-passage times, per-trial seeding, .npz + params.json
output), with the matrix ensemble replaced by the regularized RBF kernel
Gram matrix

    J = G + sigma2 * I,   G_ij = exp(-|x_i - x_j|^2 / (2 ell^2)),
    x_i ~ N(0, s^2 I_mdim) i.i.d.,

with default parameters identical to the m = 3 panel of the spectral
comparison figure: mdim = 3, s = 1, ell = 0.5, sigma2 = 1e-6.

Produces a single figure in the style of Fig. 2: median E(t) traces for
several K (with min-max shaded band over disorder trials) and an inset
with the measured speedup S_eps(d) at several thresholds plus the
asymptotic spectral ratio <lambda_{K+1}/lambda_1>.

Note on time units: for this ensemble lambda_1 ~ sigma2 ~ 1e-6, so raw
relaxation times are ~1e6. The trace horizon is therefore set adaptively
as tmax_scaled / (mu * median(lambda_1)) so the plotted axis
t [mu^-1 lambda_1^-1] spans ~[0, tmax_scaled] regardless of parameters.

Usage to produce the appendix figure
-----
    python lyapunov_covariance_rbf.py --mdim 5 --ell 1.0
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
        "text.usetex": False,
        "mathtext.fontset": "cm",
        "font.family": "STIXGeneral",
        # Font sizes
        "font.size": 30,
        "axes.labelsize": 30,
        "axes.titlesize": 30,
        "legend.fontsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 24,
        # Ticks: inward, mirrored on all four sides
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Axes
        "axes.linewidth": 0.9,
        # Grid: major only, subtle
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "grid.color": "#aaaaaa",
        # Lines
        "lines.linewidth": 1.8,
        # Legend
        "legend.frameon": False,
        "legend.handlelength": 2.0,
        # Figure
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "figure.figsize": (8.0, 5.0),
    }
)

# Trace colors (K = 0, 1, 5, 10) and inset line styles, matching Fig. 2.
COLORS = ["#e78ac3", "#8da0cb", "#fc8d62", "#66c2a5"]
INSET_LINESTYLES = ["solid", "dashed", "dotted"]


# ---------------------------------------------------------------------------
# Matrix ensemble: regularized RBF Gram matrix
# ---------------------------------------------------------------------------

def make_J_rbf(
    d: int,
    mdim: int,
    s: float,
    ell: float,
    sigma2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Regularized RBF (squared-exponential) kernel Gram matrix.

        J = G + sigma2 * I,
        G_ij = exp(-|x_i - x_j|^2 / (2 ell^2)),
        x_i ~ N(0, s^2 I_mdim) i.i.d.

    This is the matrix inverted in Gaussian-process regression with
    observation-noise variance sigma2 (the "jitter"), i.e. the
    ridge-regularized inversion J = G + eps_reg * I.
    """
    if d <= 0:
        raise ValueError("d must be positive")
    if mdim <= 0:
        raise ValueError("mdim must be positive")
    if s <= 0 or ell <= 0:
        raise ValueError("s and ell must be > 0")
    if sigma2 < 0:
        raise ValueError("sigma2 must be >= 0")

    x = rng.normal(0.0, s, size=(d, mdim))
    D2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
    G = np.exp(-D2 / (2.0 * ell ** 2))
    G = 0.5 * (G + G.T)  # kill floating-point asymmetry
    return G + sigma2 * np.eye(d)


# ---------------------------------------------------------------------------
# Closed-form error in the eigenbasis of J  (unchanged from original script)
# ---------------------------------------------------------------------------

def E_of_t_from_eigs(evals: np.ndarray, k_thermalized: int, mu: float, kBT: float, t: float) -> float:
    """Closed-form absolute Frobenius error ||Sigma(t) - Sigma_eq||_F."""
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
    """Return (t, E(t)) on a uniform grid using the closed-form expression."""
    t = np.linspace(0.0, float(tmax), int(max(2, n_points)))
    tail = evals[k_thermalized:]
    if tail.size == 0:
        return t, np.zeros_like(t)
    expo = np.exp(-4.0 * mu * tail[:, None] * t[None, :])
    s = np.sum((1.0 / (tail * tail))[:, None] * expo, axis=0)
    E = kBT * np.sqrt(s)
    return t, E


# ---------------------------------------------------------------------------
# First-passage time  (unchanged from original script)
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
    """Smallest t in [0, tmax] with E(t) <= epsilon (monotone bisection)."""
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
# Seeding / disorder realizations
# ---------------------------------------------------------------------------

def _seed_for_trial(*, d: int, seed: int, tr: int) -> int:
    """Deterministic per-trial seed, same convention as the original script."""
    return int(seed) + 1000 * int(d) + int(tr)


def build_J_trials(
    *,
    d: int,
    seed: int,
    mdim: int,
    s: float,
    ell: float,
    sigma2: float,
    trials: int,
) -> list:
    """Build all RBF disorder realizations for a given d."""
    Js = []
    for tr in range(int(max(1, trials))):
        rng_tr = np.random.default_rng(_seed_for_trial(d=d, seed=seed, tr=tr))
        Js.append(make_J_rbf(int(d), mdim=mdim, s=s, ell=ell, sigma2=sigma2, rng=rng_tr))
    return Js


def eigs_trials_from_Js(Js: list) -> list:
    """Return sorted eigenvalue arrays for each matrix in Js."""
    return [np.linalg.eigvalsh(J) for J in Js]


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Lyapunov/covariance Mpemba speedup, RBF Gram ensemble.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dimensions
    ap.add_argument("--d_min", type=int, default=200, help="Smallest dimension to sweep.")
    ap.add_argument("--d_max", type=int, default=1000, help="Largest dimension to sweep.")
    ap.add_argument("--d_step", type=int, default=100, help="Step size for dimension sweep.")
    ap.add_argument("--d_trace", type=int, default=500, help="Dimension used for E(t) trace plots.")

    # Physics
    ap.add_argument("--mu", type=float, default=1.0, help="Relaxation rate mu.")
    ap.add_argument("--kBT", type=float, default=1.0, help="Thermal energy kBT.")

    # RBF ensemble parameters (defaults = m = 3 panel of the spectral figure)
    ap.add_argument("--mdim", type=int, default=3, help="Input-space dimension m.")
    ap.add_argument("--s_data", type=float, default=1.0, help="Std of the Gaussian input density.")
    ap.add_argument("--ell", type=float, default=0.5, help="Kernel lengthscale.")
    ap.add_argument("--sigma2", type=float, default=1e-6, help="Jitter / ridge (eps_reg).")
    ap.add_argument("--trials", type=int, default=50, help="Number of disorder realizations.")

    # Error thresholds for the inset
    ap.add_argument(
        "--epsilon_list",
        type=str,
        default="1e-2, 1e-4, 1e-6",
        help="Comma-separated epsilon thresholds for the speedup inset.",
    )

    # Mpemba parameters
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Pre-thermalized fraction for the inset: K(d) = ceil(alpha * d).",
    )
    ap.add_argument(
        "--k_list",
        type=str,
        default="0,1,5,10",
        help="Comma-separated k values (pre-thermalized modes) for E(t) traces.",
    )

    # Time horizons.
    ap.add_argument("--tmax_scaled", type=float, default=13.0,
                    help="Trace horizon in units of (mu lambda_1)^-1.")
    ap.add_argument("--tmax_fpt", type=float, default=1e8,
                    help="Raw-time horizon for first-passage bisection.")
    ap.add_argument("--n_points", type=int, default=2500, help="Number of time points for trace curves.")
    ap.add_argument(
        "--band_q",
        type=float,
        default=0.0,
        help="Shaded band quantile: 0 -> full min-max envelope; q in (0, 0.5) -> [q, 1-q] quantiles "
             "(e.g. 0.1 for a 10-90%% band, robust to the number of trials).",
    )

    # RNG
    ap.add_argument("--seed", type=int, default=0, help="Base random seed.")

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    def _clean(x: str) -> str:
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
        f"alpha={args.alpha}",
        f"eps={_clean(args.epsilon_list)}",
        f"mdim={args.mdim}",
        f"s={args.s_data}",
        f"ell={args.ell}",
        f"sigma2={args.sigma2}",
        f"trials={args.trials}",
        f"tmaxsc={args.tmax_scaled}",
        f"tmaxfpt={args.tmax_fpt}",
    ]
    out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    out_dir = os.path.join(out_base, "__".join(_clean(t) for t in param_tokens))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    k_list = [int(x) for x in args.k_list.split(",") if x.strip()]
    eps_list = [float(x) for x in args.epsilon_list.split(",") if x.strip()]
    d_list = np.arange(args.d_min, args.d_max + 1, args.d_step, dtype=int)

    # ------------------------------------------------------------------
    # Cache spectra: one diagonalization per (d, trial).
    # ------------------------------------------------------------------
    evals_cache: dict = {}

    def _get_evals_trials(d: int) -> list:
        key = int(d)
        if key not in evals_cache:
            Js = build_J_trials(
                d=key,
                seed=int(args.seed),
                mdim=int(args.mdim),
                s=float(args.s_data),
                ell=float(args.ell),
                sigma2=float(args.sigma2),
                trials=int(args.trials),
            )
            evals_cache[key] = eigs_trials_from_Js(Js)
        return evals_cache[key]

    def _mean_teps(evals_trials: list, k_th: int, eps: float) -> float:
        t_list = [
            first_passage_time_from_eigs(
                e, int(k_th), float(eps), float(args.mu), float(args.kBT), float(args.tmax_fpt)
            )
            for e in evals_trials
        ]
        return float(np.mean(np.asarray(t_list, dtype=float)))

    # Archive a representative trace matrix (trial 0 at d_trace).
    evals_trials_trace = _get_evals_trials(int(args.d_trace))
    J_save = build_J_trials(
        d=int(args.d_trace),
        seed=int(args.seed),
        mdim=int(args.mdim),
        s=float(args.s_data),
        ell=float(args.ell),
        sigma2=float(args.sigma2),
        trials=1,
    )[0]
    np.savez_compressed(
        os.path.join(out_dir, "trace_matrix.npz"),
        d_trace=args.d_trace,
        J_rbf=J_save,
        lam_rbf=np.linalg.eigvalsh(J_save),
    )

    # Shared trace horizon: all trials evaluated on the SAME raw grid, with
    # the axis scaled by the ensemble-median lambda_1. The resulting min-max
    # band width reflects the trial-to-trial dispersion of the relaxation
    # rates lambda_{k+1} (broad for this ensemble, since lambda_1 is the
    # extreme tail eigenvalue of a stretched-exponential spectrum) -- i.e.
    # the band is a genuine feature of the ensemble, not a convergence
    # artifact, and does not shrink with more trials.
    lam1_med = float(np.median([float(e[0]) for e in evals_trials_trace]))
    tmax_trace = float(args.tmax_scaled) / (float(args.mu) * lam1_med)
    t_grid = np.linspace(0.0, tmax_trace, int(args.n_points))
    tau_grid = t_grid * (float(args.mu) * lam1_med)  # scaled axis in [0, tmax_scaled]

    def _band(E_stack: np.ndarray):
        q = float(args.band_q)
        E_med = np.median(E_stack, axis=0)
        if q <= 0.0:
            return E_med, np.min(E_stack, axis=0), np.max(E_stack, axis=0)
        return E_med, np.quantile(E_stack, q, axis=0), np.quantile(E_stack, 1.0 - q, axis=0)

    # ================================================================
    # Figure: RBF -- E(t) traces (main) + speedup vs d (inset)
    # ================================================================
    fig, ax = plt.subplots(figsize=(8, 4.5))

    traces: dict = {}
    for i, k_th in enumerate(k_list):
        E_trials = [
            error_curve_from_eigs(
                evals=evals,
                k_thermalized=int(k_th),
                mu=float(args.mu),
                kBT=float(args.kBT),
                tmax=tmax_trace,
                n_points=len(t_grid),
            )[1]
            for evals in evals_trials_trace
        ]
        E_stack = np.vstack(E_trials)
        E_med, E_lo, E_hi = _band(E_stack)

        traces[int(k_th)] = {
            "t_scaled": tau_grid,
            "E": E_med,
            "E_min": E_lo,
            "E_max": E_hi,
        }
        color = COLORS[i % len(COLORS)]
        ax.semilogy(tau_grid, E_med, linestyle="-", linewidth=1.8,
                    color=color, label=rf"$K={k_th}$")
        ax.fill_between(
            tau_grid,
            E_lo,
            E_hi,
            alpha=0.2,
            linewidth=0.0,
            color=color,
        )

    ax.set_xlabel(r"Time $t \, [\mu^{-1}\lambda_1^{-1}]$")
    ax.set_ylabel(r"$\mathcal{E}$")
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6, color="#aaaaaa")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax.set_ylim([1e-10, 1e7])

    # Inset: speedup vs d
    axins = ax.inset_axes([0.58, 0.44, 0.38, 0.43])

    # Per-dimension number of pre-thermalized modes: K(d) = ceil(alpha * d)
    K_of_d = {int(d): int(np.ceil(float(args.alpha) * int(d))) for d in d_list}

    inset_tk0, inset_tk, inset_S = [], [], []
    for i, eps in enumerate(eps_list):
        t_k0 = np.array([_mean_teps(_get_evals_trials(int(d)), 0, eps) for d in d_list], dtype=float)
        t_k = np.array(
            [_mean_teps(_get_evals_trials(int(d)), K_of_d[int(d)], eps) for d in d_list], dtype=float
        )
        S = np.divide(
            t_k0, t_k,
            out=np.full_like(t_k0, np.nan),
            where=np.isfinite(t_k0) & np.isfinite(t_k) & (t_k > 0),
        )
        inset_tk0.append(t_k0)
        inset_tk.append(t_k)
        inset_S.append(S)

        mask = np.isfinite(S)
        exp_label = int(np.floor(np.log10(eps) + 1e-12))
        axins.plot(
            d_list[mask], S[mask],
            color=COLORS[-1],
            marker="o",
            markersize=3,
            linewidth=1.4,
            label=rf"$\epsilon_t=10^{{{exp_label}}}$",
            linestyle=INSET_LINESTYLES[i % len(INSET_LINESTYLES)],
        )

    # Asymptotic spectral ratio R = <lambda_{K(d)+1} / lambda_1>, K(d) = ceil(alpha*d)
    R = np.array(
        [
            np.nanmean(
                [
                    (e[K_of_d[int(d)]] / e[0])
                    if (len(e) > K_of_d[int(d)] and e[0] > 0)
                    else np.nan
                    for e in _get_evals_trials(int(d))
                ]
            )
            for d in d_list
        ],
        dtype=float,
    )
    axins.plot(d_list, R, linewidth=1.4, color="black",
               label=r"$\lambda_{K(d)+1}/\lambda_1$")

    axins.grid(False)
    axins.set_xlabel(r"$d$", fontsize=22)
    axins.set_ylabel(r"$\mathcal{S}_{\epsilon_t}$", fontsize=22)
    axins.tick_params(
        axis="both", which="both", labelsize=18,
        direction="in", top=True, right=True, length=3,
    )
    axins.legend(
        loc="center right", bbox_to_anchor=(-0.2, 0.5),
        frameon=False, fontsize=18, ncol=2,
    )
    axins.set_title(rf"Speedup, $\alpha={args.alpha:g}$", fontsize=22)
    for spine in axins.spines.values():
        spine.set_linewidth(0.7)
    axins.set_facecolor("white")

    np.savez_compressed(
        os.path.join(out_dir, "figure_rbf_data.npz"),
        **{f"trace_k{kk}_t_scaled": vv["t_scaled"] for kk, vv in traces.items()},
        **{f"trace_k{kk}_E": vv["E"] for kk, vv in traces.items()},
        **{f"trace_k{kk}_E_min": vv["E_min"] for kk, vv in traces.items()},
        **{f"trace_k{kk}_E_max": vv["E_max"] for kk, vv in traces.items()},
        d_list=d_list,
        eps_list=np.array(eps_list, dtype=float),
        alpha=float(args.alpha),
        alpha_inset=float(args.alpha),
        k_inset_list=np.array([K_of_d[int(d)] for d in d_list], dtype=int),
        K_of_d=np.array([K_of_d[int(d)] for d in d_list], dtype=int),
        lam1_trace=lam1_med,
        t_eps_k0=np.array(inset_tk0, dtype=float),
        t_eps_k=np.array(inset_tk, dtype=float),
        S_eps=np.array(inset_S, dtype=float),
        R_theory=R,
    )

    ax.set_title("RBF")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "RBF.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "RBF.png"), dpi=200, bbox_inches="tight")
    print(f"saved figure and data to {out_dir}")
    print(f"median lambda_1(d={args.d_trace}) = {lam1_med:.4g}, "
          f"raw trace horizon = {tmax_trace:.4g} (band_q={args.band_q})")
    plt.show()


if __name__ == "__main__":
    main()
