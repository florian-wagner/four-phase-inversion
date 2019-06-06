#############################################
# to find "invlib" in the main folder
import os
import sys
sys.path.insert(0, os.path.abspath("../.."))
#############################################

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import ImageGrid

import pygimli as pg
from invlib import add_inner_title, logFormat, rst_cov, set_style
from pygimli.mplviewer import drawModel

fs = 5.5
set_style(fs, style="seaborn-dark")


if len(sys.argv) > 1:
    scenario = "Fig2"
    case = 2
else:
    scenario = "Fig1"
    case = 1

# Load data
mesh = pg.load("mesh.bms")
meshj = pg.load("paraDomain_%d.bms" % case)
true = np.load("true_model.npz")
est = np.load("conventional_%s.npz" % scenario)
joint = np.load("joint_inversion_%s.npz" % scenario)
sensors = np.load("sensors.npy", allow_pickle=True)

# True model
veltrue, rhotrue, fa, fi, fw, fr = true["vel"], true["rho"], true["fa"], true[
    "fi"], true["fw"], true["fr"]

# Conventional inversion
velest, rhoest, fae, fie, fwe, mask = est["vel"], est["rho"], est["fa"], est[
    "fi"], est["fw"], est["mask"]

# Joint inversion
veljoint, rhojoint, faj, fij, fwj, frj, maskj = joint["vel"], joint[
    "rho"], joint["fa"], joint["fi"], joint["fw"], joint["fr"], joint["mask"]


# Some helper functions
def update_ticks(cb, log=False, label="", cMin=None, cMax=None):
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
    tol = 1e-3 # do not hide values that are very close (but below) zero
    if not np.isclose(model.min(), 0.0, atol=tol) and (model < 0).any():
        model = np.ma.masked_where(model < -tol, model)

    if "coverage" in kwargs:
        model = np.ma.masked_where(kwargs["coverage"] == 0, model)
    gci = drawModel(ax, mesh, model, rasterized=True, nLevs=2, **kwargs)
    return gci


def minmax(data):
    """Return minimum and maximum of data as a 2-line string."""
    tmp = np.array(data)
    print("max", tmp.max())
    min = tmp.min()
    if np.max(tmp) > 10 and np.max(tmp) < 1e4:
        return "min: %d | max: %d" % (min, tmp.max())
    if np.max(tmp) > 1e4:
        return "min: %d" % min + " | max: " + logFormat(tmp.max())
    else:
        if min < 0:
            min = ("%.3f" % min).rstrip("0")
        else:
            min = ("%.2f" % min).rstrip("0")
        min = min.rstrip(".")
        max = ("%.2f" % tmp.max()).rstrip("0")
        return "min: %s | max: %s" % (min, max)


# %%
fig = plt.figure(figsize=(7, 4.5))
grid = ImageGrid(fig, 111, nrows_ncols=(6, 3), axes_pad=[0.03, 0.03],
                 share_all=True, add_all=True, cbar_location="right",
                 cbar_mode="edge", cbar_size="5%", cbar_pad=0.05, aspect=True)

cov = rst_cov(meshj, np.loadtxt("rst_coverage_%d.dat" % case))

fre = 1 - fwe - fae - fie

labels = ["v (m/s)", r" $\rho$ ($\Omega$m)"]
labels.extend([r"f$_{\rm %s}$" % x for x in "wiar"])

long_labels = [
    "Velocity", "Resistivity", "Water content", "Ice content", "Air content",
    "Rock content"
]
meshs = [mesh, meshj, meshj]
cmaps = ["viridis", "Spectral_r", "Blues", "Purples", "Greens", "Oranges"]
datas = [(veltrue, velest, veljoint), (rhotrue, rhoest, rhojoint),
         (fw, fwe, fwj), (fi, fie, fij), (fa, fae, faj), (fr, fre, frj)]

for i, (row, data, label, cmap) in enumerate(
        zip(grid.axes_row, datas, labels, cmaps)):
    print("Plotting", label)
    borderpad = 0.2
    if i == 0:
        lims = {"cMin": 1500, "cMax": 5000}
    elif i == 1:
        lims = {"cMin": 1e3, "cMax": 1e5}
        borderpad = 0.07
    elif i == 2:
        lims = {"cMin": 0.02, "cMax": 0.35}
    elif i == 3:
        lims = {"cMin": 0, "cMax": 0.3}
    elif i == 4:
        lims = {"cMin": 0, "cMax": 0.15}
    elif i == 5:
        lims = {"cMin": 0.6, "cMax": 0.8}
    else:
        lims = lim(
            list(data[0]) + list(data[1][cov > 0]) + list(data[2][cov > 0]))
    print(lims)
    logScale = True if "rho" in label else False
    ims = []
    for j, ax in enumerate(row):
        coverage = np.ones(mesh.cellCount()) if j == 0 else cov
        color = "k" if j == 0 and i not in (1, 3, 5) else "w"
        ims.append(draw(ax, meshs[j], data[j], **lims, logScale=logScale,
                   coverage=coverage))
        # ax.text(0.987, 0.05, minmax(data[j]), transform=ax.transAxes, fontsize=fs,
        #         ha="right", color=color)
        add_inner_title(ax, minmax(data[j][coverage > 0]), loc=4, size=fs,
                        fw="regular", frame=False, c=color,
                        borderpad=borderpad)
        ims[j].set_cmap(cmap)

    cb = fig.colorbar(ims[0], cax=grid.cbar_axes[i])
    update_ticks(cb, log=logScale, label=label, **lims)

for ax, title, num in zip(grid.axes_row[0], [
        "True model", "Conventional inversion and 4PM",
        "Petrophysical joint inversion"], "abc"):
    ax.set_title(title, fontsize=fs + 1, fontweight="bold")
    ax.set_title("(%s)" % num, loc="left", fontsize=fs + 1, fontweight="bold")

labs = [
    "Inverted", "Inverted", "Transformed", "Transformed", "Transformed",
    "Assumed"
]
for ax, lab in zip(grid.axes_column[1], labs):
    add_inner_title(ax, lab, loc=3, size=fs, fw="regular", frame=False, c="w")

labs = [
    "Transformed", "Transformed", "Inverted", "Inverted", "Inverted",
    "Assumed & fixed"
]

geom = pg.load("geom.bms")
if scenario == "Fig2":
    labs[-1] = "Inverted"

    # Add labels for covariance reference
    ax = grid.axes_all[0]
    mid = geom.xmax() / 2

    kwargs = dict(va="center", ha="center", fontsize=fs, fontweight="semibold")
    ax.text(mid, -2.9, "I", color="w", **kwargs)
    ax.text(20, -10, "II", color="w", **kwargs)
    ax.text(mid, -10, "III", **kwargs)
    ax.text(mesh.xmax() - 20, -10, "IV", color="w", **kwargs)
    ax.text(mid, -20, "V", **kwargs)

for ax, lab in zip(grid.axes_column[2], labs):
    add_inner_title(ax, lab, loc=3, size=fs, fw="regular", frame=False, c="w")

for i, (ax, label) in enumerate(zip(grid.axes_column[0], long_labels)):
    ax.set_yticks([-5, -15, -25])
    ax.set_yticklabels([" 5", "15", "25\n"])
    ax.set_ylabel("Depth (m)", labelpad=1)
    color = "k" if i not in (1, 3, 5) else "w"
    add_inner_title(ax, label, loc=3, c=color, frame=False)

    if i == 5:
        for bound in geom.boundaries():
            style = "solid"
            if np.isclose(bound.size(), 14.14, atol=0.5):
                style = "dotted"
            pg.mplviewer.drawSelectedMeshBoundaries(
                ax, [bound], linewidth=0.5, linestyles=style, color="k")
    else:
        pg.mplviewer.drawPLC(ax, geom, fillRegion=False, lw=0.5)

for i, ax in enumerate(grid.axes_all):
    ax.set_facecolor("0.45")
    ax.plot(sensors,
            np.zeros_like(sensors) + 0.1, marker="o", lw=0, color="k", ms=0.6)
    ax.tick_params(axis='both', which='major')
    ax.set_xticks([25, 50, 75, 100, 125])
    ax.set_ylim(-27, 0)
    ax.set_aspect(1.85)

for row in grid.axes_row[:-1]:
    for ax in row:
        ax.xaxis.set_visible(False)

for ax in grid.axes_row[-1]:
    ax.set_xlabel("x (m)", labelpad=0.2)

# fig.savefig("4PM_joint_inversion.png", dpi=150, bbox_inches="tight")
fig.savefig("%s_two_columns.pdf" % scenario, dpi=500, bbox_inches="tight",
            pad_inches=0.0)
