#!/usr/bin/env bash

export MPLBACKEND="Agg"

mkdir -p results
for erte in 0.05 0.03 0.01; do
  for rste in 0.005 0.0005 0.0001; do
    for lam in 150 80 50 30 20 10 5; do
      for weighting in 0 1; do
        for zWeight in 1 0.8 0.5 0.3 0.2; do
          python3 5_joint_inversion.py $erte $rste $lam $weighting $zWeight
          python3 6_plot_inv_results.py
          mv 4PM_joint_inversion.png results/inv_${erte}_${rste}_${lam}_${weighting}_${zWeight}.png
          mv datafit.png results/fit_${erte}_${rste}_${lam}_${weighting}_${zWeight}.png
        done
      done
    done
  done
done
