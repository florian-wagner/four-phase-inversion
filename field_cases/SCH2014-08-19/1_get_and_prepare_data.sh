#!/usr/bin/env bash
# Copy data and correct with numerically calculcated geometric factor

# Define data files
ert="SCH2014-08-19_u_i_Pellet-et-al2016.txt"
rst="SCH2014-08-19_tt_Pellet-et-al-2016.txt"
# ert="SCH2014-08-19_u_i.txt"
# rst="SCH2014-08-19_tt.txt"

# Copy seismic data
cp -f ../../field_data/$rst rst.data

# Copy ERT data and calculate K / rhoa based on primary potentials in a
# temporary folder
mkdir -p bert
cd bert
cp ../../../field_data/$ert ert.ohm
bertNew2DTopo ert.ohm > bert.cfg
bert bert.cfg meshs pot filter
mv -f ert.data ..
cd ..
rm -rf bert
