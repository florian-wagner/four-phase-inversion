#!/usr/bin/env python3
"""
This script aims to replot the results from Pellet et al. (2016)
"""

#############################################
# to find "invlib" in the main folder
import sys
import os
sys.path.insert(0, os.path.abspath("../.."))
#############################################

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

import pygimli as pg
from invlib import FourPhaseModel

dxy = 0.5  # spacing used in x and y direction


def plot_boreholes(ax, **kwargs):
    elevation_5198 = 0.12
    elevation_5000 = 0.65
    depth_5198 = 2.1 + elevation_5198 # topo
    depth_5000 = 2.2 + elevation_5000 # topo
    ax.plot([10, 10], [-10, -elevation_5198], "k-", **kwargs)
    ax.plot([26, 26], [-20, -elevation_5000], "k-", **kwargs)
    ax.plot([9, 11], [-depth_5198, -depth_5198], "k-", **kwargs)
    ax.plot([25, 27], [-depth_5000, -depth_5000], "k-", **kwargs)


def load_result(path):
    mat = np.loadtxt(path)
    vec = np.flipud(mat).flatten()
    cov = np.isfinite(vec)
    return dict(mat=mat, vec=vec, cov=cov)


labs = [
    r"Porosity $\phi$", r"Ice content / $\phi$", r"Water content / $\phi$",
    r"Air content / $\phi$"
]
fa = load_result("./SCH2014-08-19_Pellet-et-al-2016_fa.txt")
fi = load_result("./SCH2014-08-19_Pellet-et-al-2016_fi.txt")
fw = load_result("./SCH2014-08-19_Pellet-et-al-2016_fw.txt")
phi = load_result("./SCH2014-08-19_Pellet-et-al-2016_porosity.txt")

# %% Create pygimli grid according to results from Pellet et al.
nx, ny = fa["mat"].shape
mesh = pg.createGrid(nx + 1, ny + 1)
# Scale to 0.5 m cell sizes
mesh.scale(dxy)
# Flip upside down
mesh.translate([0.0, -mesh.xmax(), 0.0])

# Plot
fig = plt.figure(figsize=(4, 10))

grid = AxesGrid(fig, 111, nrows_ncols=(4, 1), cbar_mode="single",
                cbar_location="right", cbar_pad=0.1, cbar_size="2%",
                axes_pad=0.3)

for i, (ax, vec) in enumerate(zip(grid, [phi, fi, fw, fa])):
    if i == 0:
        quant = vec["vec"] * 100
    else:
        quant = vec["vec"] / phi["vec"] * 100
    quant[~vec["cov"]] = 0
    pg.show(mesh, quant, coverage=vec["cov"], logScale=False, cMap="jet_r",
            cMin=0, cMax=100, ax=ax)
    ax.set_ylim(-15, 0)
    ax.set_title(labs[i])
    plot_boreholes(ax)

pg.mplviewer.createColorBarOnly(cMin=0, cMax=100, cMap="jet_r",
                                orientation="vertical", ax=grid.cbar_axes[0])
fig.tight_layout()
fig.savefig("pellet_2016_fig7a-d.pdf", bbox_inches="tight")
mesh.save("grid.bms")
np.savetxt("phi_grid.dat", phi["vec"])

fpm = FourPhaseModel(phi=phi["vec"], va=300., vi=3500., vw=1500, m=1.4, n=2.4,
                     rhow=60, vr=6000)

vel = 1 / fpm.slowness(fw["vec"], fi["vec"], fa["vec"])
rho = fpm.rho(fw["vec"], fi["vec"], fa["vec"])

np.savez("pellet.npz", fa=fa["vec"], fi=fi["vec"], fw=fw["vec"],
         phi=phi["vec"], mask=vec["cov"], vel=vel, rho=rho)
