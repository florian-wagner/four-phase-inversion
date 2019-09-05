#!/usr/bin/env python
# coding: utf-8

# # Plot settings

from fpinv import set_style
fs = 6
set_style(fs, style="seaborn-dark")
n = 5

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MultipleLocator, FormatStrFormatter, NullFormatter, FixedLocator
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from mpl_toolkits.axes_grid1 import ImageGrid


# # Plot model covariance matrices

J = np.load("jacJoint.npz", allow_pickle=True)
colsum = np.sum(J, axis=0)

# normalize to cumulative sensitivity (row sums of J)
MCMc = np.load("./MCMconst.npz", allow_pickle=True)
MCM = np.load("./MCM.npz", allow_pickle=True)

# Plot
size=  3.67
fig = plt.figure(figsize=(size, size), dpi=200)
grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.03,
                 add_all=True, cbar_location="right", cbar_mode="edge",
                 cbar_size="5%", cbar_pad=0.025, aspect=True)

minv, maxv = -0.01, 0.01
for ax, mat in zip(grid.axes_all, [MCM, MCMc]):
    im = ax.imshow(mat, cmap="RdGy_r", vmin=minv, vmax=maxv)
    ax.set_aspect("equal")
    ticks = np.linspace(-0.5, 4*n -0.5, 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.set_xlim(-0.5, 4*n - 0.5)
    ax.set_ylim(-0.5, 4*n - 0.5)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    minorLocator = FixedLocator(ticks[:-1] + n/2)
    subMinorLocator = FixedLocator(np.arange(4*n))
    majorFormatter = NullFormatter()
    ax.xaxis.set_minor_locator(minorLocator)
    ax.xaxis.set_major_formatter(majorFormatter)

    ax.yaxis.set_minor_locator(subMinorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)

    labels = ["Water", "Ice", "Air","Rock"]
    long_labels = [r"%s (f$_{\rm %s}$)" % (lab, x) for lab, x in zip(labels, "wiar")]
    ylabs = ["I", "II", "III", "IV", "V"] * 4
    ax.set_xticklabels(long_labels,minor=True)
    ax.set_yticklabels(ylabs, minor=True, fontsize=5.5)
    ax.set_xticklabels(long_labels, minor=True, fontsize=5.5)

    ax.tick_params(axis='x', which='minor')
    ax.tick_params(axis='x', which='minor', bottom=False, top=True, labelbottom=False, labeltop=True)

    ax.grid(which="major", color="0.5", linestyle='--')

for ax, title, letter in zip(grid.axes_all, [
        "Unconstrained\n", "With volume\nconservation constraints"
], "ab"):
    ax.set_title(title, fontsize=fs + 1, fontweight="bold")
    ax.set_title("(%s)\n" % letter, fontsize=fs + 1, fontweight="bold", loc="left")

ax.yaxis.tick_left()
cax = grid.cbar_axes[0]

cbar = fig.colorbar(im, orientation="vertical", cax=cax, pad=0)#, extend="both")
cbar.set_ticks([minv, 0, maxv])
cbar.set_ticklabels([" %.2f" % minv, "0", " %.2f" % maxv])
cbar.set_label(label="Model covariance", labelpad=-6, fontweight="semibold", fontsize=fs)
fig.savefig("Fig4_one_column.pdf")
