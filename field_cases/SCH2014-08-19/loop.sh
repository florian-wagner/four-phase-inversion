#!/usr/bin/env bash

export MPLBACKEND="Agg"

mkdir -p results
for erte in 0.03 0.02; do
  for rste in 0.0005 0.0003; do
    for lam in 80 50 30 20 10 5; do
      for weighting in 0 1; do
        for zWeight in 0.5 0.25 0.1; do
          python3 5_joint_inversion.py $erte $rste $lam $weighting $zWeight && python3 6_plot_inv_results.py || continue
          mv 4PM_joint_inversion.png results/inv_${erte}_${rste}_${lam}_${weighting}_${zWeight}.png || continue
          mv datafit.png results/fit_${erte}_${rste}_${lam}_${weighting}_${zWeight}.png || continue
        done
      done
    done
  done
done
