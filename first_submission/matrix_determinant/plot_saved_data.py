#!/usr/bin/env python3
"""plot_saved_data.py

Load convergence data from a folder of .npz files and show up to four plots:
  - Wishart absolute / relative log-error
  - Fixed (Haar) absolute / relative log-error

Expected files inside the folder:
  convergence_wishart.npz  and/or  convergence_fixed.npz

Usage:
  python3 plot_saved_data.py <folder> --plot_d 100 \
      --speedup_eps_list_wishart 0.1,0.01 \
      --speedup_eps_list_fixed 0.001,0.0001
"""

from __future__ import annotations

import argparse
import math
import os
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COLOR_K0 = "#e78ac3"
COLOR_KN = "#66c2a5"


def _apply_pubstyle() -> None:
    """Apply the publication plotting style used by the original script."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 24,
            "axes.labelsize": 26,
            "axes.titlesize": 26,
            "xtick.labelsize": 21,
            "ytick.labelsize": 21,
            "legend.fontsize": 18,
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
            "axes.linewidth": 0.9,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": True,
            "axes.grid.which": "major",
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "grid.color": "#aaaaaa",
            "lines.linewidth": 1.4,
            "lines.markersize": 11.0,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
            "legend.handlelength": 2.0,
            "figure.dpi": 130,
            "savefig.dpi": 300,
            "figure.figsize": (8.0, 5.0),
        }
    )


def _summary_stats(e: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return median and IQR across A-realizations. Expects shape (nN, nA)."""
    med = np.nanmedian(e, axis=1)
    q25 = np.nanquantile(e, 0.25, axis=1)
    q75 = np.nanquantile(e, 0.75, axis=1)
    return med, q25, q75


def _load_error_arrays(
    data: np.lib.npyio.NpzFile, error_type: str
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load the requested error arrays from the .npz file."""
    if error_type == "abs":
        if "abs_eps0" in data and "abs_epsK" in data:
            return (
                np.asarray(data["abs_eps0"], dtype=float),
                np.asarray(data["abs_epsK"], dtype=float),
                "abs",
            )
        if "eps0" in data and "epsK" in data:
            return (
                np.asarray(data["eps0"], dtype=float),
                np.asarray(data["epsK"], dtype=float),
                "abs",
            )
        raise KeyError("Could not find absolute-error arrays in the npz file.")

    if error_type == "rel":
        if "rel_eps0" in data and "rel_epsK" in data:
            return (
                np.asarray(data["rel_eps0"], dtype=float),
                np.asarray(data["rel_epsK"], dtype=float),
                "rel",
            )
        raise KeyError("Could not find relative-error arrays in the npz file.")

    raise ValueError(f"Unsupported error_type={error_type!r}")


def _base_title_from_ensemble(ensemble: str) -> str:
    """Map ensemble identifiers to the panel title used in the figures."""
    ensemble_l = ensemble.lower()
    if ensemble_l == "wishart":
        return "Wishart"
    if ensemble_l == "fixed":
        return "Haar"
    return ensemble if ensemble else "Data"


def _compute_T_epsilon(
    Ns: np.ndarray,
    eps: np.ndarray,
    tburn: np.ndarray,
    tau: float,
    eps_target: float,
    stat: str = "median",
) -> float:
    """Return T* at which the error curve crosses eps_target.

    Interpolates in the same (T, R) log-log space as the dashed lines in the
    main plot. Prints a warning and returns nan if eps_target is outside the
    observed range.
    """
    if stat == "median":
        eps_stat = np.nanmedian(eps, axis=1)
        tb = float(np.nanmedian(tburn))
    else:
        eps_stat = np.nanmean(eps, axis=1)
        tb = float(np.nanmean(tburn))

    T = Ns * (tb + float(tau))

    valid = np.isfinite(eps_stat) & (eps_stat > 0)
    if valid.sum() < 2:
        print(
            f"  [warning] fewer than 2 valid points — speedup undefined for eps={eps_target:g}"
        )
        return float("nan")

    T_v = T[valid]
    eps_v = eps_stat[valid]
    eps_max = float(np.max(eps_v))
    eps_min = float(np.min(eps_v))

    if eps_target > eps_max:
        print(
            f"  [warning] eps={eps_target:g} not reached: errors already below threshold "
            f"(observed range [{eps_min:.3g}, {eps_max:.3g}]). Choose a larger epsilon."
        )
        return float("nan")

    if eps_target < eps_min:
        print(
            f"  [warning] eps={eps_target:g} not reached: curve never drops to threshold "
            f"within simulated range (observed range [{eps_min:.3g}, {eps_max:.3g}]). "
            f"Choose a smaller epsilon or simulate more trajectories."
        )
        return float("nan")

    cross_idx = np.where(eps_v <= eps_target)[0]
    if cross_idx.size == 0:
        print(f"  [warning] eps={eps_target:g} not reached (no crossing found).")
        return float("nan")

    i1 = int(cross_idx[0])
    if i1 == 0:
        print(
            f"  [warning] eps={eps_target:g} not reached: error already <= threshold "
            f"at smallest N (observed range [{eps_min:.3g}, {eps_max:.3g}]). "
            f"Choose a larger epsilon."
        )
        return float("nan")

    i0 = i1 - 1
    log_T0, log_T1 = np.log(float(T_v[i0])), np.log(float(T_v[i1]))
    log_e0, log_e1 = np.log(float(eps_v[i0])), np.log(float(eps_v[i1]))
    log_eT = np.log(float(eps_target))

    frac = (log_eT - log_e0) / (log_e1 - log_e0)
    return float(np.exp(log_T0 + frac * (log_T1 - log_T0)))


def _add_speedup_inset(
    ax: plt.Axes,
    Ns: np.ndarray,
    Ds: np.ndarray,
    err0: np.ndarray,
    errK: np.ndarray,
    tburn0: np.ndarray,
    tburnK: np.ndarray,
    tau: float,
    mpemba_K: int,
    speedup_eps_list: List[float],
    speedup_stat: str = "median",
) -> None:
    """Add the speedup inset without changing the original plotting logic."""
    eps_markers = ["o", "s", "^", "D", "v", "p", "h", "X"]

    curves: List[Tuple[float, List[float]]] = []
    for eps_target in speedup_eps_list:
        S_list = []
        for i_d in range(len(Ds)):
            T0 = _compute_T_epsilon(
                Ns, err0[i_d, :, :], tburn0[i_d, :], tau, eps_target, stat=speedup_stat
            )
            TK = _compute_T_epsilon(
                Ns, errK[i_d, :, :], tburnK[i_d, :], tau, eps_target, stat=speedup_stat
            )
            S_list.append(
                T0 / TK if (np.isfinite(T0) and np.isfinite(TK) and TK > 0) else np.nan
            )
        curves.append((eps_target, S_list))

    axins = ax.inset_axes([0.16, 0.12, 0.33, 0.38])
    axins.set_xticks(Ds)
    marker_sizes = [10, 8, 8, 8, 8, 8, 8, 8]

    for idx_e, (eps_target, S_list) in enumerate(curves):
        axins.plot(
            Ds,
            S_list,
            marker=eps_markers[idx_e % len(eps_markers)],
            markersize=marker_sizes[idx_e % len(marker_sizes)],
            linewidth=0,
            linestyle="none",
            color=COLOR_KN,
            alpha=0.75,
            markeredgecolor="black",
            markeredgewidth=0.8,
            label=fr"$\epsilon={eps_target:g}$",
            clip_on=False,
        )

    axins.margins(x=0.24)

    y_vals = [s for _, S_list in curves for s in S_list if np.isfinite(s)]
    if y_vals:
        y_mid = np.mean(y_vals)
        half_width = max(5, np.ptp(y_vals) * 0.6)
        axins.set_ylim(y_mid - half_width * 1.4, y_mid + half_width * 1.4)

    axins.set_xlabel(r"$d$", fontsize=20, labelpad=-2)
    axins.set_ylabel(r"$S_{\mathrm{det}}$", fontsize=20)
    axins.tick_params(
        axis="both",
        which="both",
        labelsize=17,
        direction="in",
        top=True,
        right=True,
        length=3,
    )
    axins.set_title(fr"Speedup ($K={mpemba_K}$)", fontsize=19, pad=3)
    for spine in axins.spines.values():
        spine.set_linewidth(0.7)
    axins.set_facecolor("white")
    axins.grid(False)

    if len(curves) > 1:
        axins.legend(
            fontsize=14,
            frameon=True,
            framealpha=1.0,
            edgecolor="black",
            loc="center right",
            bbox_to_anchor=(1.6, 0.3),
            handlelength=0.8,
            labelspacing=0.3,
            borderpad=0.5,
        )


def plot_from_npz(
    npz_path: str,
    error_type: str = "abs",
    xaxis: str = "time",
    annotate_N: bool = False,
    speedup_eps_list: Optional[List[float]] = None,
    speedup_stat: str = "median",
    plot_d: Optional[int] = None,
) -> None:
    """Create one plot from a convergence .npz file.

    The annotate_N argument is kept for CLI compatibility with the original
    script, even though it is not used in the current plotting logic.
    """
    data = np.load(npz_path, allow_pickle=True)

    ensemble = str(data.get("ensemble", ""))
    base_title = _base_title_from_ensemble(ensemble)
    Ns = np.asarray(data["Ns"], dtype=float)
    Ds = np.asarray(data["Ds"], dtype=int)
    tburn0 = np.asarray(data["tburn0"], dtype=float)
    tburnK = np.asarray(data["tburnK"], dtype=float)
    tau = float(data["tau"])
    mpemba_K = int(data.get("mpemba_K", 0))

    if speedup_eps_list is None:
        speedup_eps_list = [0.1]
    if "speedup_stat" in data and speedup_stat == "median":
        speedup_stat = str(data["speedup_stat"])

    eps0, epsK, _ = _load_error_arrays(data, error_type=error_type)

    try:
        if error_type == "rel":
            speedup_err0 = np.asarray(data["rel_eps0"], dtype=float)
            speedup_errK = np.asarray(data["rel_epsK"], dtype=float)
        else:
            key0 = "abs_eps0" if "abs_eps0" in data else "eps0"
            keyK = "abs_epsK" if "abs_epsK" in data else "epsK"
            speedup_err0 = np.asarray(data[key0], dtype=float)
            speedup_errK = np.asarray(data[keyK], dtype=float)
    except KeyError:
        speedup_err0 = speedup_errK = None

    if plot_d is not None:
        matches = np.where(Ds == plot_d)[0]
        if len(matches) == 0:
            raise ValueError(f"plot_d={plot_d} not found. Available: {list(Ds)}")
        main_indices = [int(matches[0])]
    else:
        main_indices = list(range(len(Ds)))

    N_markers = ["o", "s", "^", "D", "v", "p", "h", "X"]

    fig, ax = plt.subplots()

    for i_d in main_indices:
        e0 = eps0[i_d, :, :]
        med0, q250, q750 = _summary_stats(e0)
        yerr0 = np.vstack([med0 - q250, q750 - med0])
        x0 = Ns if xaxis == "N" else Ns * (float(np.nanmedian(tburn0[i_d, :])) + tau)

        ax.plot(
            x0,
            med0,
            linestyle="--",
            linewidth=1.1,
            color=COLOR_K0,
            alpha=0.35,
            zorder=2,
        )
        for i_n, (xv, yv, ye, N_pt) in enumerate(zip(x0, med0, yerr0.T, Ns)):
            label_k0 = (
                fr"$N_{{\rm traj}}={int(N_pt)}$" if i_d == main_indices[0] else "_nolegend_"
            )
            ax.errorbar(
                [xv],
                [yv],
                yerr=[[ye[0]], [ye[1]]],
                marker=N_markers[i_n % len(N_markers)],
                linewidth=0,
                capsize=2.5,
                capthick=0.8,
                elinewidth=0.8,
                color=COLOR_K0,
                label=label_k0,
                zorder=3,
            )

        if mpemba_K > 0:
            eK = epsK[i_d, :, :]
            medK, q25K, q75K = _summary_stats(eK)
            yerrK = np.vstack([medK - q25K, q75K - medK])
            xK = Ns if xaxis == "N" else Ns * (float(np.nanmedian(tburnK[i_d, :])) + tau)

            ax.plot(
                xK,
                medK,
                linestyle="--",
                linewidth=1.1,
                color=COLOR_KN,
                alpha=0.35,
                zorder=2,
            )
            for i_n, (xv, yv, ye, N_pt) in enumerate(zip(xK, medK, yerrK.T, Ns)):
                ax.errorbar(
                    [xv],
                    [yv],
                    yerr=[[ye[0]], [ye[1]]],
                    marker=N_markers[i_n % len(N_markers)],
                    linewidth=0,
                    capsize=2.5,
                    capthick=0.8,
                    elinewidth=0.8,
                    color=COLOR_KN,
                    label="_nolegend_",
                    zorder=3,
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_on()

    ylo, yhi = ax.get_ylim()
    ax.set_ylim(10 ** (math.log10(ylo) - 0.6), yhi)

    ax.set_xlabel(
        r"Number of trajectories $N_{\mathrm{traj}}$" if xaxis == "N" else r"Compute time $T$ [a.u.]"
    )
    ax.set_ylabel(r"$\mathcal{R}$")
    ax.set_title(base_title)

    data_handles, data_labels = ax.get_legend_handles_labels()
    n_pairs = [(h, l) for h, l in zip(data_handles, data_labels) if not l.startswith("_")]

    black_handles = []
    for h, _ in n_pairs:
        try:
            new_h = Line2D(
                [],
                [],
                marker=h[0].get_marker(),
                color="black",
                linestyle="none",
                markersize=h[0].get_markersize(),
            )
        except Exception:
            new_h = h
        black_handles.append(new_h)

    color_handles = [Patch(facecolor=COLOR_K0, edgecolor="none", label=r"$K=0$")]
    if mpemba_K > 0:
        color_handles.append(
            Patch(facecolor=COLOR_KN, edgecolor="none", label=fr"$K={mpemba_K}$")
        )

    labels_N = [l for _, l in n_pairs]
    labels_K = [h.get_label() for h in color_handles]

    legend_N = ax.legend(
        black_handles,
        labels_N,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.78, 1.0),
        handlelength=1.5,
        labelspacing=0.35,
        frameon=False,
        borderaxespad=0.2,
    )
    ax.add_artist(legend_N)

    ax.legend(
        color_handles,
        labels_K,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        handlelength=1.5,
        labelspacing=0.35,
        frameon=False,
        borderaxespad=0.2,
    )

    if mpemba_K > 0 and speedup_eps_list and speedup_err0 is not None and speedup_errK is not None:
        _add_speedup_inset(
            ax=ax,
            Ns=Ns,
            Ds=Ds,
            err0=speedup_err0,
            errK=speedup_errK,
            tburn0=tburn0,
            tburnK=tburnK,
            tau=tau,
            mpemba_K=mpemba_K,
            speedup_eps_list=speedup_eps_list,
            speedup_stat=speedup_stat,
        )

    fig.tight_layout()


def _parse_eps_list(s: str) -> List[float]:
    """Parse a comma-separated epsilon list from the CLI."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show Wishart/Fixed convergence plots from a folder of .npz files."
    )
    parser.add_argument(
        "folder",
        help="Folder containing convergence_wishart.npz and/or convergence_fixed.npz",
    )
    parser.add_argument(
        "--xaxis",
        choices=["time", "N"],
        default="time",
        help="Plot against compute time (default) or number of trajectories.",
    )
    parser.add_argument("--no_annotate_N", action="store_true")
    parser.add_argument(
        "--speedup_eps_list",
        type=str,
        default="0.1",
        metavar="EPS1,EPS2,...",
        help="Comma-separated ε thresholds for the speedup inset (both ensembles). Default: 0.1.",
    )
    parser.add_argument(
        "--speedup_eps_list_wishart",
        type=str,
        default=None,
        metavar="EPS1,EPS2,...",
        help="ε thresholds for the Wishart speedup inset. Overrides --speedup_eps_list.",
    )
    parser.add_argument(
        "--speedup_eps_list_fixed",
        type=str,
        default=None,
        metavar="EPS1,EPS2,...",
        help="ε thresholds for the Fixed speedup inset. Overrides --speedup_eps_list.",
    )
    parser.add_argument(
        "--plot_d",
        type=int,
        default=None,
        metavar="D",
        help="Show only this dimension in the main panel (all dims used for speedup inset).",
    )
    args = parser.parse_args()

    speedup_eps_default = _parse_eps_list(args.speedup_eps_list)
    speedup_eps_wishart = (
        _parse_eps_list(args.speedup_eps_list_wishart)
        if args.speedup_eps_list_wishart
        else speedup_eps_default
    )
    speedup_eps_fixed = (
        _parse_eps_list(args.speedup_eps_list_fixed)
        if args.speedup_eps_list_fixed
        else speedup_eps_default
    )

    if not os.path.isdir(args.folder):
        raise FileNotFoundError(f"Folder not found: {args.folder}")

    _apply_pubstyle()

    wishart_file = os.path.join(args.folder, "convergence_wishart.npz")
    fixed_file = os.path.join(args.folder, "convergence_fixed.npz")
    made_any_plot = False

    if os.path.exists(wishart_file):
        plot_from_npz(
            wishart_file,
            error_type="abs",
            xaxis=args.xaxis,
            annotate_N=not args.no_annotate_N,
            plot_d=args.plot_d,
            speedup_eps_list=speedup_eps_wishart,
        )
        made_any_plot = True
        try:
            plot_from_npz(
                wishart_file,
                error_type="rel",
                xaxis=args.xaxis,
                annotate_N=not args.no_annotate_N,
                plot_d=args.plot_d,
                speedup_eps_list=speedup_eps_wishart,
            )
        except KeyError as exc:
            print(f"[warning] Skipping Wishart relative plot: {exc}")

    if os.path.exists(fixed_file):
        plot_from_npz(
            fixed_file,
            error_type="abs",
            xaxis=args.xaxis,
            annotate_N=not args.no_annotate_N,
            plot_d=args.plot_d,
            speedup_eps_list=speedup_eps_fixed,
        )
        made_any_plot = True
        try:
            plot_from_npz(
                fixed_file,
                error_type="rel",
                xaxis=args.xaxis,
                annotate_N=not args.no_annotate_N,
                plot_d=args.plot_d,
                speedup_eps_list=speedup_eps_fixed,
            )
        except KeyError as exc:
            print(f"[warning] Skipping Fixed relative plot: {exc}")

    if not made_any_plot:
        raise FileNotFoundError(
            f"Could not find convergence_wishart.npz or convergence_fixed.npz inside {args.folder}"
        )

    plt.show()


if __name__ == "__main__":
    main()
