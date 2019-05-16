#############################################
# to find "invlib" in the main folder
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
#############################################
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import ImageGrid

import pygimli as pg
from invlib import add_inner_title, logFormat, rst_cov, set_style
from pygimli.mplviewer import drawModel
from reproduce_pellet import plot_boreholes, load_result
fs = 4.5
set_style(fs, style="seaborn-dark")

# Load data
gridmesh = pg.load("grid.bms")

mesh = pg.load("mesh.bms")
meshj = pg.load("paraDomain.bms")
est = np.load("conventional.npz")
joint = np.load("joint_inversion.npz")
pellet = np.load("pellet.npz")
sensors = np.loadtxt("sensors.npy")

# Pellet et al. (2006)
vel, rho, fr, fa, fi, fw, gridcov = pellet["vel"], pellet["rho"], 1 - pellet[
    "phi"], pellet["fa"], pellet["fi"], pellet["fw"], pellet["mask"]

fa /= pellet["phi"]
fi /= pellet["phi"]
fw /= pellet["phi"]

# Conventional inversion
velest, rhoest, fae, fie, fwe, mask = est["vel"], est["rho"], est["fa"], est[
    "fi"], est["fw"], est["mask"]

phie = fae + fie + fwe
fae /= phie
fie /= phie
fwe /= phie
fre = 1 - phie

# Joint inversion
veljoint, rhojoint, faj, fij, fwj, frj, maskj = joint["vel"], joint[
    "rho"], joint["fa"], joint["fi"], joint["fw"], joint["fr"], joint["mask"]

faj /= 1 - frj
fij /= 1 - frj
fwj /= 1 - frj


# Some helper functions
def update_ticks(cb, label="", cMin=None, cMax=None):
    t = ticker.FixedLocator([cMin, cMax])
    cb.set_ticks(t)
    ticklabels = cb.ax.yaxis.get_ticklabels()
    for i, tick in enumerate(ticklabels):
        if i == 0:
            tick.set_verticalalignment("bottom")
        if i == len(ticklabels) - 1:
            tick.set_verticalalignment("top")

    cb.ax.annotate(label, xy=(1, 0.5), xycoords='axes fraction',
                   xytext=(10, 0), textcoords='offset pixels',
                   horizontalalignment='center', verticalalignment='center',
                   rotation=90, fontsize=fs, fontweight="regular")


def lim(data):
    """Return appropriate colorbar limits."""
    data = np.array(data)
    print("dMin", data.min(), "dMax", data.max())
    if data.min() < 0:
        dmin = 0.0
    else:
        dmin = np.around(data.min(), 2)
    dmax = np.around(data.max(), 2)
    kwargs = {"cMin": dmin, "cMax": dmax}
    return kwargs


def draw(ax, mesh, model, **kwargs):
    model = np.array(model)
    if not np.isclose(model.min(), 0.0, atol=9e-3) and (model < 0).any():
        model = np.ma.masked_where(model < 0, model)
        model = np.ma.masked_where(model > 1, model)

    if "coverage" in kwargs:
        model = np.ma.masked_where(kwargs["coverage"] == 0, model)
    gci = drawModel(ax, mesh, model, rasterized=True, nLevs=2, **kwargs)
    return gci


def minmax(data):
    """Return minimum and maximum of data as a 2-line string."""
    tmp = np.array(data)
    print("max", tmp.max())
    if np.isclose(tmp.min(), 0, atol=9e-3):
        min = 0
    else:
        min = tmp.min()
    if np.max(tmp) > 10 and np.max(tmp) < 1e4:
        return "min: %d | max: %d" % (min, tmp.max())
    if np.max(tmp) > 1e4:
        return "min: %d" % min + " | max: " + logFormat(tmp.max())
    else:
        return "min: %.2f | max: %.2f" % (min, tmp.max())


# %%
fig = plt.figure(figsize=(7, 4.5))
grid = ImageGrid(fig, 111, nrows_ncols=(6, 3), axes_pad=[0.03, 0.03],
                 share_all=True, add_all=True, cbar_location="right",
                 cbar_mode="edge", cbar_size="5%", cbar_pad=0.05, aspect=True)

cov = rst_cov(meshj, np.loadtxt("rst_coverage.dat"))

labels = ["v (m/s)", r"$\rho$ ($\Omega$m)"]
labels.extend([r"f$_{\rm %s}$ / $\phi$" % x for x in "wia"])
labels.extend([r"f$_{\rm %s}$" % x for x in "r"])

long_labels = [
    "Velocity", "Resistivity", "Water content", "Ice content", "Air content",
    "Rock content"
]
meshs = [gridmesh, meshj, meshj]
cmaps = ["viridis", "Spectral_r", "Blues", "Purples", "Greens", "Oranges"]
datas = [(vel, velest, veljoint), (rho, rhoest, rhojoint), (fw, fwe, fwj),
         (fi, fie, fij), (fa, fae, faj), (fr, fre, frj)]

for i, (row, data, label,
        cmap) in enumerate(zip(grid.axes_row, datas, labels, cmaps)):
    print("Plotting", label)
    if i == 0:
        lims = {"cMin": 1000, "cMax": 4000}
    elif i == 1:
        lims = {"cMin": 600, "cMax": 2000}
    elif i == 2:  # water
        lims = {"cMin": 0.4, "cMax": 0.65}
    elif i == 3:  # ice
        lims = {"cMin": 0, "cMax": 0.5}
    elif i == 4:  # air
        lims = {"cMin": 0, "cMax": 0.5}
    elif i == 5:  # rock
        lims = {"cMin": 0.3, "cMax": 0.9}
    else:
        lims = lim(list(data[0][cov > 0]) + list(data[1][cov > 0]))
    print(lims)
    logScale = True if "rho" in label else False
    ims = []
    for j, ax in enumerate(row):
        if data[j] is None:
            ims.append(None)
            continue
        coverage = gridcov if j == 0 else cov
        #color = "k" if j is 0 and i not in (1, 3, 5) else "w"
        ims.append(
            draw(ax, meshs[j], data[j], **lims, logScale=logScale,
                 coverage=coverage))
        ax.text(0.987, 0.05, minmax(data[j][coverage > 0]),
                transform=ax.transAxes, fontsize=fs, ha="right", color="w")
        ims[j].set_cmap(cmap)

    cb = fig.colorbar(ims[1], cax=grid.cbar_axes[i])
    if not logScale:
        update_ticks(cb, label=label, **lims)

for ax, title in zip(
        grid.axes_row[0],
    ["Pellet et al. (2016)", "Conventional inversion and 4PM", "Petrophysical joint inversion"]):
    ax.set_title(title, fontsize=fs)  #, fontweight="bold")

labs = ["replotted"] * 6
for ax, lab in zip(grid.axes_column[0], labs):
    add_inner_title(ax, lab, loc=3, size=fs, fw="regular", frame=False, c="w")

labs = [
    "inverted", "inverted", "transformed", "transformed", "transformed",
    "assumed"
]
for ax, lab in zip(grid.axes_column[1], labs):
    add_inner_title(ax, lab, loc=3, size=fs, fw="regular", frame=False, c="w")

labs = [
    "transformed", "transformed", "inverted", "inverted", "inverted",
    "inverted"
]

for ax, lab in zip(grid.axes_column[2], labs):
    add_inner_title(ax, lab, loc=3, size=fs, fw="regular", frame=False, c="w")

for i, ax in enumerate(grid.axes_all):
    ax.set_facecolor("0.45")
    ax.plot(sensors[:,0], sensors[:,1], "k.", ms=0.5)
    ax.tick_params(axis='both', which='major')
    ax.set_ylim(-16, 0)
    if i < 3:
        c = "w"
    else:
        c = "k"
    plot_boreholes(ax, lw=0.5, color=c)
    # add_inner_title(ax, ascii_uppercase[i], loc=2, frame=False, c="w", fw="bold")

for row in grid.axes_row[:-1]:
    for ax in row:
        ax.xaxis.set_visible(False)

for ax in grid.axes_column[-1]:
    ax.yaxis.set_visible(False)

for ax in grid.axes_row[-1]:
    ax.set_xlabel("x (m)")

for i, (ax, label) in enumerate(zip(grid.axes_column[0], long_labels)):
    ax.set_yticks([-0, -5, -10, -15])
    ax.set_yticklabels([" 0", " 5", "10", "15"])
    ax.set_ylabel("Depth (m)", labelpad=1)
    # add_inner_title(ax, label, loc=2, c="k", frame=False)

ax = grid.axes_column[0][2]

fig.savefig("4PM_joint_inversion.pdf", dpi=300, bbox_inches="tight")
