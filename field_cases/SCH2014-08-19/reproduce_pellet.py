#!/usr/bin/env python3
"""
This script aims to replot the results from Pellet et al. (2016)
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import AxesGrid

import pygimli as pg

dxy = 0.5  # spacing used in x and y direction

def plot_boreholes(ax):
    ax.plot([10, 10], [-10, 0], "k-")
    ax.plot([26, 26], [-20, 0], "k-")
    ax.plot([9, 11], [-2, -2], "k-")
    ax.plot([25, 27], [-2.2, -2.2], "k-")

def load_result(path):
    mat = np.loadtxt(path)
    vec = np.flipud(mat).flatten()
    cov = np.isfinite(vec)
    return dict(mat=mat, vec=vec, cov=cov)


labs = [
    r"Porosity $\phi$", r"Ice content / $\phi$",
    r"Water content / $\phi$", r"Air content / $\phi$"
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
