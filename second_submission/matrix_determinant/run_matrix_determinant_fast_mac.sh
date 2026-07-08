#!/usr/bin/env bash
# Mac-friendly run script for the accelerated determinant simulation.
# The #SBATCH lines are intentionally removed: run this locally with bash/tmux.
# Matrix/scientific parameters are kept identical to your original script;
# only runtime/performance settings are changed.

set -euo pipefail

# Avoid oversubscription on macOS / NumPy backends.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg

# ================================
# Matrix / protocol parameters
# ================================
ensembles="wishart,fixed"

d=5 # used only for the optional first small printed check; skipped below for long runs.
d_list="50, 100, 200" # main panel dimensions; plotting script can select one dimension later.

mpemba_K=5
burnin_mode="eps"
burnin_eps=1e-3

# Performance parameter: with crooks_det_alpha_inset_fast_mac.py the drive uses exact OU
# updates, so dt no longer needs to be 1e-4 for stability. Test 2e-2 vs 1e-2 once if needed.
dt=0.01
tau=2.0

# speedup inset settings
speedup_eps_list="1e-1,1e-2"
speedup_interp="powerlaw"  # "loglog"
speedup_stat="median"

# ensemble settings -- unchanged
wishart_m_factor=1.5
wishart_ridge=0. # keep zero
wishart_trials=50 # unchanged
fixed_trials=100 # unchanged

# sampling / averaging -- unchanged
n_list="10, 100, 1000, 10000"
seed=0

# alpha=K/d, constant with d in the inset -- unchanged
alpha_inset=0.05

# Fast Python implementation generated for the Mac run.
SCRIPT=${SCRIPT:-crooks_det_alpha_inset_fast_mac.py}

# ================================
# Run
# ================================
python3 "$SCRIPT" \
  --estimator crooks \
  --a2 1.0 \
  --convergence \
  --ensembles "$ensembles" \
  --d "$d" \
  --d_list "$d_list" \
  --mpemba_K "$mpemba_K" \
  --burnin_mode "$burnin_mode" \
  --burnin_eps "$burnin_eps" \
  --tau "$tau" \
  --dt "$dt" \
  --n_list "$n_list" \
  --fixed_trials "$fixed_trials" \
  --wishart_m_factor "$wishart_m_factor" \
  --wishart_ridge "$wishart_ridge" \
  --wishart_trials "$wishart_trials" \
  --speedup_eps_list "$speedup_eps_list" \
  --speedup_interp "$speedup_interp" \
  --speedup_stat "$speedup_stat" \
  --seed "$seed" \
  --xaxis time \
  --save_npz \
  --no_plots \
  --skip_single \
  --plot_out convergence_abs.png \
  --plot_out_rel convergence_rel.png \
  --alpha_inset "$alpha_inset"
