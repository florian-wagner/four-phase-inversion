import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.offsetbox import AnchoredText
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import ticker

import pybert as pb
import pygimli as pg
from pygimli.mplviewer import drawModel as draw

seaborn.set(font="Fira Sans", style="ticks")
plt.rcParams["image.cmap"] = "viridis"

# Load data
mesh = pg.load("mesh.bms")
true = np.load("true_model.npz")
est = np.load("conventional.npz")
sensors = np.load("sensors.npy")

veltrue, rhotrue, fa, fi, fw = true["vel"], true["rho"], true["fa"], true["fi"], true["fw"]
velest, rhoest, fae, fie, fwe = est["vel"], est["rho"], est["fa"], est["fi"], est["fw"]

labels = ["$v$ (m/s)", r"$\rho$ ($\Omega$m)", "$f_a$", "$f_i$", "$f_w$"]
long_labels = [
    "Velocity",
    "Resistivity",
    "Air content",
    "Ice content",
    "Water content"
]

def add_inner_title(ax, title, loc, size=None, **kwargs):
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'])
    at = AnchoredText(title, loc=loc, prop=size, pad=0., borderpad=0.4,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=2)])
    at.patch.set_ec("none")
    at.patch.set_alpha(0.5)
    return at

def update_ticks(cb, log=False, label=""):
    if log:
        t = ticker.LogLocator(numticks=2)
    else:
        t = ticker.LinearLocator(numticks=2)
    cb.ax.annotate(label,
                xy=(1, 0.5), xycoords='axes fraction',
                xytext=(100, 0), textcoords='offset pixels',
                horizontalalignment='center',
                verticalalignment='center', rotation=90)
    cb.set_ticks(t)
    cb.update_ticks()
    ticks = cb.get_clim()
    if not log and ticks[1] < 1:
        cb.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in ticks])

def lim(data):
    dmin = np.around(data.min(), 2)
    dmax = np.around(data.max(), 2)
    if dmin < 0.02:
        dmin = 0
    kwargs = {
        "cMin": dmin,
        "cMax": dmax
    }
    return kwargs

fig = plt.figure(figsize=(8, 12))
grid = ImageGrid(fig, 111, nrows_ncols=(5, 2), axes_pad=0.15, share_all=True,
                 add_all=True, cbar_location="right", cbar_mode="edge",
                 cbar_size="5%", cbar_pad=0.15, aspect=True)

im = draw(grid.axes_row[0][0], mesh, veltrue, **lim(veltrue), logScale=False)
im = draw(grid.axes_row[0][1], mesh, velest, **lim(veltrue), logScale=False)
cb = fig.colorbar(im, cax=grid.cbar_axes[0])
update_ticks(cb, label=labels[0])

im = draw(grid.axes_row[1][0], mesh, rhotrue, cmap="Spectral_r", **lim(rhotrue))
im = draw(grid.axes_row[1][1], mesh, rhoest, cmap="Spectral_r", **lim(rhotrue))
cb = fig.colorbar(im, cax=grid.cbar_axes[1])
update_ticks(cb, log=True, label=labels[1])

im = draw(grid.axes_row[2][0], mesh, fa, logScale=False, cmap="Greens", **lim(fa))
im = draw(grid.axes_row[2][1], mesh, fae, logScale=False, cmap="Greens", **lim(fa))
cb = fig.colorbar(im, cax=grid.cbar_axes[2])
update_ticks(cb, label=labels[2])

im = draw(grid.axes_row[3][0], mesh, fi, logScale=False, cmap="Purples", **lim(fi))
im = draw(grid.axes_row[3][1], mesh, fie, logScale=False, cmap="Purples", **lim(fi))
cb = fig.colorbar(im, cax=grid.cbar_axes[3])
update_ticks(cb, label=labels[3])

im = draw(grid.axes_row[4][0], mesh, fw, logScale=False, cmap="Blues", **lim(fw))
im = draw(grid.axes_row[4][1], mesh, fwe, logScale=False, cmap="Blues", **lim(fw))
cb = fig.colorbar(im, cax=grid.cbar_axes[4])
update_ticks(cb, label=labels[4])

for ax, title in zip(grid.axes_row[0],
                     ["True model", "Conventional inversion + 4PM"]):
    ax.set_title(title, fontweight="bold")

for ax in grid.axes_all:
    ax.plot(sensors, np.zeros_like(sensors), 'wv')
    ax.set_aspect(2)

for row in grid.axes_row[:-1]:
    for ax in row:
        ax.xaxis.set_visible(False)

for ax in grid.axes_column[-1]:
    ax.yaxis.set_visible(False)

for ax in grid.axes_row[-1]:
    ax.set_xlabel("x (m)")

for ax, label in zip(grid.axes_column[0], long_labels):
    add_inner_title(ax, label, loc=3)
    ax.set_ylabel("y (m)")

fig.savefig("conventional.png", dpi=150)
pg.wait()
