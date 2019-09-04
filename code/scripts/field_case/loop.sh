#!/usr/bin/env bash

export MPLBACKEND="Agg"

mkdir -p results
mkdir -p calcs

task(){
  name="mod_${mod}_z${z}_lam${lam}"
  echo $name
  sleep 1
  folder="calcs/${name}"
  mkdir -p $folder
  cp -r *.py *.txt *.bms *.npz *.npy *.dat *.data $folder
  cd $folder
  for case in 1 2; do
    python3 5_joint_inversion.py $mod $lam $z $case || continue
    mv datafit_${case}.png ../../results/fit_${name}.png || continue
  done
  python3 6_plot_inv_results.py || continue
  mv 4PM_joint_inversion.png ../../results/inv_${name}.png || continue
  cd ../..
}

subtask(){
  for lam in 50 30 10 80; do
    for z in 0.3 0.2 0.1; do
      task || continue
    done
  done
}

for mod in 0 1; do
  subtask &
done
