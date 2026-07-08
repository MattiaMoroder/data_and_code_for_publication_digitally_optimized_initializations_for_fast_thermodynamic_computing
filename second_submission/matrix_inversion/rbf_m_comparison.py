import numpy as np
import matplotlib.pyplot as plt
from math import comb

# ----------------------------- parameters -----------------------------
S_DATA = 1.0
ELL = 1.0
SIGMA2 = 1e-6
D = 500
SEED = 12345

# ------------------------------ helpers -------------------------------
def rbf_eigvals(d, m, rng, s=S_DATA, ell=ELL, sigma2=SIGMA2):
    x = rng.normal(0.0, s, size=(d, m))
    D2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=-1)
    G = np.exp(-D2 / (2.0 * ell ** 2))
    G = 0.5 * (G + G.T)
    return np.linalg.eigvalsh(G + sigma2 * np.eye(d))   # ascending

def zwr_mu_md(nvals, m, s=S_DATA, ell=ELL):
    a = 1.0 / (4.0 * s ** 2)
    b = 1.0 / (2.0 * ell ** 2)
    c = np.sqrt(a ** 2 + 2.0 * a * b)
    A_, B_ = a + b + c, b / (a + b + c)
    mu0 = np.sqrt(2.0 * a / A_)
    out, K = [], 0
    while len(out) < nvals:
        out.extend([mu0 ** m * B_ ** K] * comb(K + m - 1, m - 1))
        K += 1
    return np.array(out[:nvals])

# ------------------------------- styling ------------------------------
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

# ------------------------------- data ---------------------------------
ss = np.random.SeedSequence(SEED)
rng_m1 = np.random.default_rng(ss.spawn(1)[0])
rng_m5 = np.random.default_rng(1000 * D + 0)
cases = [(1, rng_m1), (5, rng_m5)]

fig, axes = plt.subplots(2, 1, figsize=(6.6, 9.6), sharex=True, sharey=True)

for ax, (m, rng) in zip(axes, cases):
    lam = rbf_eigvals(D, m, rng)[::-1]                  # descending
    th = D * zwr_mu_md(D, m) + SIGMA2
    kstar = int(np.sum(lam > 2 * SIGMA2))               # above-floor modes

    ax.semilogy(np.arange(D), lam, "o", ms=2.5, mfc="none", color="C0",
                label=r"eigenvalues of $\mathbf{A}$")
    ax.semilogy(np.arange(D), th, "-", color="C1", lw=1.4,
                label=r"$n\,\mu_k+\sigma^2$")
    ax.axhline(SIGMA2, color="k", ls="--", lw=1, label=r"jitter $\sigma^2$")
    ax.set_xlabel(r"index $k$ (descending)")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(rf"$m={m}$:  $k^*={kstar}$ modes above floor")
    if m == 1:
        ax.annotate(r"$\mathbf{A}\approx\sigma^2\mathbb{1}+$ rank-$k^*$",
                    xy=(0.42, 0.32), xycoords="axes fraction", fontsize=12)
        ax.legend(frameon=False)
    else:
        ax.annotate(r"$\lambda_k\sim e^{-c_m k^{1/m}}$"
                    "\n(full-spectrum decay)",
                    xy=(0.45, 0.55), xycoords="axes fraction", fontsize=12)
    print(f"m={m}: lam_max={lam[0]:.3g}, lam_min={lam[-1]:.3g}, "
          f"k*={kstar}, S(50)={lam[::-1][50]/lam[::-1][0]:.3g}, "
          f"S(250)={lam[::-1][250]/lam[::-1][0]:.3g}")

fig.tight_layout()

# Save in the current folder on your machine.
fig.savefig("rbf_m_comparison_vertical_largefonts_v2.pdf", bbox_inches="tight")
fig.savefig("rbf_m_comparison_vertical_largefonts_v2.png", dpi=200, bbox_inches="tight")

print("saved rbf_m_comparison_vertical_largefonts_v2.pdf / .png")
plt.show()
