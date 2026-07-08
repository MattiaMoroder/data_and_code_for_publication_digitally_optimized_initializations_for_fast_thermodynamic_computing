#!/bin/bash

#SBATCH --job-name=TCM_inv

#SBATCH --ntasks=1

#SBATCH --export=ALL
#SBATCH --mail-type=NONE
#SBATCH --partition=cluster  #,large  #th-ws,th-cl,cluster

#SBATCH --time=7-00:00:00    #7-00:00:00                     #10-00:00:00
#SBATCH --mem=400GB   #800GB
#SBATCH --output=/project/th-scratch/m/Mattia.Moroder/accelerating-thermodynamic-computing-with-Mpemba/matrix_inversion/slurm/output_%j.sout
#SBATCH --error=/project/th-scratch/m/Mattia.Moroder/accelerating-thermodynamic-computing-with-Mpemba/matrix_inversion/slurm/output_%j.serr

# parameters for lyapunov_covariance.py
d_min=200 #200
d_max=1000 #1000
d_step=25 #50

d_trace=500 #500  #NOTE: This is the dimension of the matrix for the main plot

mu=1.0
kBT=1.0

epsilon_fixed_list="1e-2, 1e-3, 1e-4"
epsilon_wishart_list="1e-1, 1e-2, 1e-3"

k_speed=10 #NOTE: This is not used anymore. We used the "alpha_inset" parameter now
k_list="0,1,5,10"


alpha_min=0.5
alpha_max=0.5
fixed_trials=100

wishart_m_factor=1.5  #needs to be >1. A factor closer to 1 gives a larger speedup!
wishart_ridge=0.
wishart_trials=100

tmax_fixed=25.0
tmax_wishart=300 # #1010.0

dt=-1.0
dt_safety=0.05

seed=0

# alpha= K/d, constant with d in the inset
alpha_inset=0.05

python3 lyapunov_covariance_alpha_inset.py \
  --d_min $d_min \
  --d_max $d_max \
  --d_step $d_step \
  --mu $mu \
  --kBT $kBT \
  --epsilon_fixed_list "$epsilon_fixed_list" \
  --epsilon_wishart_list "$epsilon_wishart_list" \
  --k_speed $k_speed \
  --k_list "$k_list" \
  --d_trace $d_trace \
  --alpha_min $alpha_min \
  --alpha_max $alpha_max \
  --fixed_trials $fixed_trials \
  --wishart_m_factor $wishart_m_factor \
  --wishart_ridge $wishart_ridge \
  --wishart_trials $wishart_trials \
  --tmax_fixed $tmax_fixed \
  --tmax_wishart $tmax_wishart \
  --dt $dt \
  --dt_safety $dt_safety \
  --seed $seed \
  --alpha_inset $alpha_inset
