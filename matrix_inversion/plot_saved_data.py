#!/usr/bin/env python3
"""
Replot data saved by lyapunov_covariance_clean_savedata.py
using (nearly) the exact same figure structure/styling as the original script.

Workflow:
1) load the data
2) plot it
3) savefig() as PDF
4) plt.show() to show both figures

Expected files in run_dir:
  - figure1_fixed_data.npz
  - figure2_wishart_data.npz

Example run:
  python3 plot_saved_data.py --run_dir data/<run_name>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Matplotlib style (kept the same as the original script)
# -----------------------------------------------------------------------------
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


COLORS = ["#e78ac3", "#8da0cb", "#fc8d62", "#66c2a5"]
INSET_LINESTYLES = ["solid", "dashed", "dotted"]


# -----------------------------------------------------------------------------
# Data loading / discovery helpers
# -----------------------------------------------------------------------------
def load_npz(path: Path) -> Dict[str, np.ndarray]:
    """Load a .npz file into a plain dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}



def newest_run_dir(data_root: Path) -> Path:
    """Return the most recently modified run directory under data_root."""
    if not data_root.exists():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")

    candidates = [p for p in data_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run folders found under: {data_root}")

    return max(candidates, key=lambda p: p.stat().st_mtime)



def infer_k_list(npz: Dict[str, np.ndarray]) -> List[int]:
    """Infer the sorted list of K values from keys like trace_k10_E."""
    ks = set()
    for key in npz.keys():
        if not key.startswith("trace_k"):
            continue

        rest = key[len("trace_k") :]
        num = ""
        for ch in rest:
            if ch.isdigit() or (ch == "-" and not num):
                num += ch
            else:
                break

        if num:
            ks.add(int(num))

    return sorted(ks)


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def plot_trace_family(ax: plt.Axes, fig_npz: Dict[str, np.ndarray]) -> None:
    """Plot the main semilogy traces and optional min/max bands."""
    ks = infer_k_list(fig_npz)

    for i, k_th in enumerate(ks):
        x = np.asarray(fig_npz[f"trace_k{k_th}_t_scaled"]).ravel()
        y = np.asarray(fig_npz[f"trace_k{k_th}_E"]).ravel()
        n = min(len(x), len(y))

        color = COLORS[i % len(COLORS)]
        ax.semilogy(
            x[:n],
            y[:n],
            linestyle="-",
            label=fr"$K={k_th}$",
            linewidth=1.8,
            color=color,
        )

        key_min = f"trace_k{k_th}_E_min"
        key_max = f"trace_k{k_th}_E_max"
        if key_min in fig_npz and key_max in fig_npz:
            y_min = np.asarray(fig_npz[key_min]).ravel()
            y_max = np.asarray(fig_npz[key_max]).ravel()
            nn = min(n, len(y_min), len(y_max))
            ax.fill_between(
                x[:nn],
                y_min[:nn],
                y_max[:nn],
                alpha=0.2,
                color=color,
                linewidth=0,
            )



def plot_speedup_inset(
    axins: plt.Axes,
    fig_npz: Dict[str, np.ndarray],
    *,
    legend_anchor_y: float,
    clip_ylim: bool = False,
) -> None:
    """Plot the speedup inset using the original styling and legend placement."""
    d_list = np.asarray(fig_npz["d_list"]).astype(int).ravel()
    eps_list = np.asarray(fig_npz["eps_list"]).astype(float).ravel()
    S_eps = np.asarray(fig_npz["S_eps"]).astype(float)
    R_theory = np.asarray(fig_npz["R_theory"]).astype(float).ravel()

    for i, eps in enumerate(eps_list):
        if i >= S_eps.shape[0]:
            continue

        S = S_eps[i, :]
        mask = np.isfinite(S)
        exp_label = int(np.floor(np.log10(eps) + 1e-12))
        linestyle = INSET_LINESTYLES[i % len(INSET_LINESTYLES)]

        axins.plot(
            d_list[mask],
            S[mask],
            linewidth=1.4,
            linestyle=linestyle,
            marker="o",
            markersize=3,
            label=rf"$\epsilon_t=10^{{{exp_label}}}$",
            color=COLORS[-1],
        )

    axins.plot(
        d_list,
        R_theory,
        color="black",
        label=r"$\lambda_{11}/\lambda_1$",
        linewidth=1.4,
    )

    axins.grid(False)
    axins.set_xlabel(r"$d$", fontsize=22)
    axins.set_ylabel(r"$\mathcal{S}_{\epsilon_t}$", fontsize=22)
    axins.tick_params(
        axis="both",
        which="both",
        labelsize=18,
        direction="in",
        top=True,
        right=True,
        length=3,
    )
    axins.legend(
        loc="center right",
        bbox_to_anchor=(-0.2, legend_anchor_y),
        frameon=False,
        fontsize=18,
        ncol=2,
    )
    axins.set_title(rf"Speedup, $K={int(fig_npz.get('k_speed', 0))}$", fontsize=22)

    for spine in axins.spines.values():
        spine.set_linewidth(0.7)
    axins.set_facecolor("white")

    if clip_ylim:
        all_S_vals = S_eps[np.isfinite(S_eps)].ravel()
        all_vals = np.concatenate([all_S_vals, R_theory[np.isfinite(R_theory)]])
        if len(all_vals) > 0:
            data_min = all_vals.min()
            data_max = all_vals.max()
            data_span = data_max - data_min if data_max > data_min else 1.0
            pad = 0.3 * data_span
            y_lo = max(8, data_min - pad)
            y_hi = min(25, data_max + pad)
            axins.set_ylim([y_lo, y_hi])


# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------
def replot_fixed(fig_npz: Dict[str, np.ndarray]) -> plt.Figure:
    """Reproduce Figure 1 (Haar) with the original layout/styling."""
    fig_fix, ax_fix = plt.subplots(figsize=(8, 4.5))

    plot_trace_family(ax_fix, fig_npz)

    ax_fix.set_xlabel(r"Time $t \, [\mu^{-1}\lambda_1^{-1}]$")
    ax_fix.set_ylabel(r"$\mathcal{E}$")
    ax_fix.grid(True, which="major", alpha=0.25, linewidth=0.6, color="#aaaaaa")
    ax_fix.legend(loc="lower left", frameon=False, ncol=2)
    ax_fix.set_ylim([1e-10, 1e5])

    axins_fix = ax_fix.inset_axes([0.58, 0.44, 0.38, 0.43])
    plot_speedup_inset(axins_fix, fig_npz, legend_anchor_y=0.8, clip_ylim=True)

    ax_fix.set_title("Haar")
    fig_fix.tight_layout()
    fig_fix.canvas.draw()

    return fig_fix



def replot_wishart(fig_npz: Dict[str, np.ndarray]) -> plt.Figure:
    """Reproduce Figure 2 (Wishart) with the original layout/styling."""
    fig_w, ax_w = plt.subplots(figsize=(8, 4.5))

    plot_trace_family(ax_w, fig_npz)

    ax_w.set_xlabel(r"Time $t \,[\mu^{-1}\lambda_1^{-1}]$")
    ax_w.set_ylabel(r"$\mathcal{E}$")
    ax_w.grid(True, which="major", alpha=0.25, linewidth=0.6, color="#aaaaaa")
    ax_w.legend(loc="lower left", frameon=False, ncol=2)
    ax_w.set_ylim([1e-10, 1e7])

    axins_w = ax_w.inset_axes([0.58, 0.44, 0.38, 0.43])
    plot_speedup_inset(axins_w, fig_npz, legend_anchor_y=0.95, clip_ylim=False)

    ax_w.set_title("Wishart")
    fig_w.tight_layout()
    fig_w.canvas.draw()

    return fig_w


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Run directory containing the saved .npz files.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="If --run_dir is not given, pick newest run under this (default: ./data).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Where to write PDFs (default: <run_dir>/replots_exact).",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    here = Path(__file__).resolve().parent
    data_root = Path(args.data_root).resolve() if args.data_root else (here / "data")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else newest_run_dir(data_root)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (run_dir / "replots_exact")
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_npz = load_npz(run_dir / "figure1_fixed_data.npz")
    wish_npz = load_npz(run_dir / "figure2_wishart_data.npz")

    fig1 = replot_fixed(fixed_npz)
    fig2 = replot_wishart(wish_npz)

    fig1.savefig(out_dir / "Haar.pdf", bbox_inches="tight")
    fig2.savefig(out_dir / "Wishart.pdf", bbox_inches="tight")

    print(f"Loaded from: {run_dir}")
    print(f"Saved PDFs to: {out_dir}")

    plt.show()


if __name__ == "__main__":
    main()
