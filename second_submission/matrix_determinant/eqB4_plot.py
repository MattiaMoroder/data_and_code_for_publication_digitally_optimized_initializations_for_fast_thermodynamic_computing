#!/usr/bin/env python3
"""
Appendix figure for referee comments 3 & 4: numerical solution of the
optimality condition Eq. (B4),

    1/lambda_{K+1} - 1/lambda_{K+2}  =  C_hw / tau_c ,
    C_hw = 4 nu mu d^2 P / (N_traj L F),

for the two matrix ensembles, keeping the hardware time constant tau_c free.

Left panel  : the spectral marginal saving 1/lambda_{K+1}-1/lambda_{K+2} vs K
              (black), with horizontal RHS lines C_hw/tau_c for several tau_c.
              Each intersection is the optimum K*(tau_c). Slower hardware
              (larger tau_c) -> lower line -> larger K*.
Right panel : K*(tau_c) read off directly, for both ensembles, with the
              Ref. [4] value tau_c ~ 1 us marked.

All non-tau_c constants are folded into C_hw and stated in the caption; changing
nu, L, or F only rescales which tau_c a given line corresponds to, so the
tau_c axis should be read as the combination C_hw/tau_c.

Reproducible via SeedSequence. Regenerate F on your own hardware if you want a
machine-specific FLOP rate; it only shifts the tau_c axis by a constant factor.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------- parameters -------------------------------------
d       = 500
nu      = 1.0        # Lanczos iterations per eigenpair (O(1) for a separated edge)
mu      = 1.0
P       = 1          # trajectory parallelism
N_traj  = 10_000
L       = 12.0       # ln(E0/eps_t); weak function of K, treated as constant
F       = 1e9        # sustained FLOP rate [1/s] (fiducial)
C_hw    = 4*nu*mu*d**2*P/(N_traj*L*F)

# ensemble parameters (match crooks_det.py)
amin, amax, wm_factor, ridge = 0.5, 0.5, 1.2, 0.0
wishart_trials = 40
seed = 0

tau_lines = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]      # highlighted tau_c [s]; 1e-6 = Ref [4]
tau_dense = np.logspace(-8.5, -3.5, 240)        # for the K*(tau_c) curve

rng = np.random.default_rng(seed)
print(f"C_hw = {C_hw:.3e}   (RHS = C_hw / tau_c)")

# --------------------------- spectra ----------------------------------------
def marg_saving_fixed(d):
    """1/lam_{K+1} - 1/lam_{K+2}, K = 0..d-2, deterministic fixed spectrum."""
    lam = np.linspace(amin, amax*d, d)
    return 1.0/lam[:-1] - 1.0/lam[1:]

def marg_saving_wishart(d, trials):
    """Ensemble-averaged 1/lam_{K+1} - 1/lam_{K+2} for Wishart matrices."""
    acc = []
    for _ in range(trials):
        m = int(math.ceil(wm_factor*d))
        X = rng.standard_normal((m, d))
        lam = np.linalg.eigvalsh(X.T@X/m) + ridge
        acc.append(1.0/lam[:-1] - 1.0/lam[1:])
    return np.mean(acc, axis=0)

lhs = {"fixed": marg_saving_fixed(d), "wishart": marg_saving_wishart(d, wishart_trials)}

def Kstar(curve, rhs):
    """Optimal K: first index where the (decreasing) spectral curve <= rhs."""
    below = np.where(curve <= rhs)[0]
    return int(below[0]) if below.size else len(curve)   # len -> saturates at d-1

# --------------------------- figure -----------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.9))
colors = plt.cm.viridis(np.linspace(0.12, 0.82, len(tau_lines)))
ens_style = {"fixed": dict(color="tab:blue",   marker="o"),
             "wishart": dict(color="tab:orange", marker="s")}
Kax = np.arange(len(lhs["fixed"]))              # K = 0 .. d-2

# ---- left: Eq. (B4) as curve + RHS lines ----
axL.plot(Kax + 1, lhs["fixed"],   color="tab:blue",   lw=2, label="fixed spectrum")
axL.plot(Kax + 1, lhs["wishart"], color="tab:orange", lw=2, label="Wishart")
for c, tau in zip(colors, tau_lines):
    axL.axhline(C_hw/tau, color=c, ls="--", lw=1.2)
    lbl = rf"$\tau_c=10^{{{int(np.log10(tau))}}}$s" + (" [4]" if abs(np.log10(tau)+6) < 1e-9 else "")
    axL.plot([], [], color=c, ls="--", lw=1.2, label=lbl)
axL.set_xscale("log"); axL.set_yscale("log"); axL.set_xlim(1, d)
axL.set_xlabel(r"$K$")
axL.set_ylabel(r"$1/\lambda_{K+1}-1/\lambda_{K+2}$  and  $C_{\rm hw}/\tau_c$")
axL.set_title("Spectral curve vs. hardware line", fontsize=9.5)
axL.grid(alpha=0.25, which="both")
axL.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=1)

# ---- right: K*(tau_c) for both ensembles ----
for name in ("fixed", "wishart"):
    Ks = np.array([Kstar(lhs[name], C_hw/t) for t in tau_dense], dtype=float)
    st = ens_style[name]
    axR.plot(tau_dense, Ks, color=st["color"], lw=2, label=name if name == "fixed" else "Wishart")
    # highlight the tabulated tau_c points
    for c, tau in zip(colors, tau_lines):
        axR.plot(tau, Kstar(lhs[name], C_hw/tau), st["marker"], color=st["color"],
                 ms=6, mec="k", mew=0.5, zorder=5)
axR.axhline(d, color="0.6", ls=":", lw=1.2)
axR.text(tau_dense[0], d, r" $K=d$ (full diagonalization)", va="bottom", ha="left",
         fontsize=7, color="0.4")
axR.axvline(1e-6, color="0.55", ls=":", lw=1.2)
axR.text(1e-6, 1.3, r" Ref.[4] $\tau_c\!\sim\!1\,\mu$s", rotation=90, va="bottom",
         ha="right", fontsize=7, color="0.4")
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlabel(r"hardware time constant $\tau_c$  [s]")
axR.set_ylabel(r"optimal $K^\star$")
axR.set_title(r"$K^\star(\tau_c)$", fontsize=9.5)
axR.grid(alpha=0.25, which="both")
axR.legend(fontsize=7.5, frameon=False, loc="lower right")
axR.set_ylim(1, 1.4*d)

fig.suptitle(rf"Optimal number of prethermalized modes ($d={d}$, "
             rf"$N_{{\rm traj}}=10^4$, $C_{{\rm hw}}={C_hw:.1e}$)", fontsize=9)
fig.tight_layout()
fig.savefig("eqB4_Kstar.png", dpi=220)
fig.savefig("eqB4_Kstar.pdf")
print("[saved] eqB4_Kstar.png / .pdf")

# --------------------------- report -----------------------------------------
print(f"{'tau_c[s]':>10s} {'K*_fixed':>10s} {'K*_wishart':>12s}")
for tau in tau_lines:
    print(f"{tau:>10.0e} {Kstar(lhs['fixed'], C_hw/tau):>10d} {Kstar(lhs['wishart'], C_hw/tau):>12d}")
