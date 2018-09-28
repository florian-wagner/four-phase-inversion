from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib import ticker
from matplotlib.offsetbox import AnchoredText
from matplotlib.path import Path
from matplotlib.patheffects import withStroke
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.spatial import ConvexHull

import pygimli as pg
from pygimli.mplviewer import addCoverageAlpha, drawModel

seaborn.set_context("paper")
seaborn.set(font="Noto Sans")



seaborn.set(style="dark")
plt.rcParams["image.cmap"] = "viridis"

config = dict(fontsize=6)
plt.rcParams['font.size'] = config['fontsize']
plt.rcParams['axes.labelsize'] = config['fontsize']
plt.rcParams['xtick.labelsize'] = config['fontsize']
plt.rcParams['ytick.labelsize'] = config['fontsize']
plt.rcParams['legend.fontsize'] = config['fontsize']
plt.rcParams['xtick.major.pad'] = 1
plt.rcParams['ytick.major.pad'] = 1

# Load data
mesh = pg.load("mesh.bms")
meshj = pg.load("paraDomain.bms")
true = np.load("true_model.npz")
est = np.load("conventional.npz")
joint = np.load("joint_inversion.npz")
sensors = np.load("sensors.npy")

# True model
veltrue, rhotrue, fa, fi, fw, fr = true["vel"], true["rho"], true["fa"], true[
    "fi"], true["fw"], true["fr"]

# Conventional inversion
velest, rhoest, fae, fie, fwe, mask = est["vel"], est["rho"], est["fa"], est[
    "fi"], est["fw"], est["mask"]
# fae = np.ma.array(fae, mask=est["mask"])
# fie = np.ma.array(fie, mask=est["mask"])
# fwe = np.ma.array(fwe, mask=est["mask"])

# Joint inversion
veljoint, rhojoint, faj, fij, fwj, frj, maskj = joint["vel"], joint[
    "rho"], joint["fa"], joint["fi"], joint["fw"], joint["fr"], joint["mask"]

labels = [
    "$v$ (m/s)", r"$\rho$ ($\Omega$m)", "$f_a$", "$f_i$", "$f_w$", "$f_r$"
]
long_labels = [
    "Velocity", "Resistivity", "Air content", "Ice content", "Water content",
    "Rock content"
]


def add_inner_title(ax, title, loc, size=None, c="k", frame=True, **kwargs):
    if size is None:
        size = dict(size=plt.rcParams['legend.fontsize'], color=c)
    else:
        size = dict(size=size, color=c)
    at = AnchoredText(title, loc=loc, prop=size, pad=0., borderpad=0.4,
                      frameon=False, **kwargs)
    ax.add_artist(at)
    if frame:
        at.txt._text.set_path_effects(
            [withStroke(foreground="w", linewidth=1)])
        at.patch.set_ec("none")
        at.patch.set_alpha(0.5)
    return at


def update_ticks(cb, log=False, label=""):
    if log:
        t = ticker.LogLocator(numticks=3)
    else:
        t = ticker.LinearLocator(numticks=2)
    cb.ax.annotate(label, xy=(1, 0.5), xycoords='axes fraction', xytext=(25,
                                                                         0),
                   textcoords='offset pixels', horizontalalignment='center',
                   verticalalignment='center', rotation=90)
    cb.set_ticks(t)
    cb.update_ticks()
    ticks = cb.get_clim()
    if not log and ticks[1] < 1:
        cb.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in ticks])


def lim(data):
    data = np.array(data)
    print(data.min(), data.max())
    if data.min() < 0.05:
        dmin = 0.0
    else:
        dmin = np.around(data.min(), 2)
    dmax = np.around(data.max(), 2)
    kwargs = {"cMin": dmin, "cMax": dmax}
    print(dmin, dmax)
    return kwargs


# %%

fig = plt.figure(figsize=(7, 4.5))
grid = ImageGrid(fig, 111, nrows_ncols=(6, 3), axes_pad=[0.02, 0.1],
                 share_all=True, add_all=True, cbar_location="right",
                 cbar_mode="edge", cbar_size="5%", cbar_pad=0.05, aspect=True)

ert_cov = np.loadtxt("ert_coverage.dat")
rst_cov = np.loadtxt("rst_coverage.dat")

# ert_cov = pg.interpolate(mesh, ert_cov, meshj.cellCenters()).array()
# rst_cov = pg.interpolate(mesh, rst_cov, meshj.cellCenters()).array()

# Extract convex hull

points_all = np.column_stack((
    pg.x(meshj.cellCenters()),
    pg.y(meshj.cellCenters()),
))

points = points_all[np.nonzero(rst_cov)[0]]
hull = ConvexHull(points)
hull_path = Path(points[hull.vertices])

covs = []
for cell in mesh.cells():
    if not hull_path.contains_point(points_all[cell.id()]):
        covs.append(ert_cov[cell.id()])


def joint_cov(ert_cov, rst_cov, mesh):
    """ Joint ERT and RST coverage for visualization. """

    points_all = np.column_stack((
        pg.x(mesh.cellCenters()),
        pg.y(mesh.cellCenters()),
    ))
    cov = np.zeros_like(ert_cov)
    for cell in mesh.cells():
        if hull_path.contains_point(points_all[cell.id()]):
            cov[cell.id()] = 1
        else:
            cov[cell.id()] = 0

    return cov


cov = joint_cov(ert_cov, rst_cov, meshj)

# for cell in mesh.cells():
#     if 10**ert_cov[cell.id()] > 10**ert_cov.max() * 0.03:
#         rst_cov[cell.id()] = 1
#
# cov[np.nonzero(rst_cov)[0]] = ert_cov.max()
# cov[np.nonzero(rst_cov)[0]] = ert_cov.max()
# cov[np.nonzero(rst_cov)[0]] += ert_cov.max()


def draw(ax, mesh, model, **kwargs):
    model = np.array(model)
    if not np.allclose(model.min(), 0.0):
        if (model < 0).any():
            model = np.ma.masked_where(model < 0, model)
            # model[model < 0] = -99
            # palette = copy(plt.get_cmap(kwargs["cmap"]))
            # palette.set_under('r', 1)
            # kwargs["cmap"] = palette

    if "coverage" in kwargs:
        model = np.ma.masked_where(kwargs["coverage"] == 0, model)

    gci = drawModel(ax, mesh, model, rasterized=True, **kwargs)
    # if "coverage" in kwargs:
    #     addCoverageAlpha(gci, kwargs["coverage"])

    # for simplex in hull.simplices:
    #     x = points[simplex, 0]
    #     y = points[simplex, 1]
    #     if (y < 0.1).all():
    #         ax.plot(x, y, 'w-', lw=1.5, alpha=0.5)

    return gci

def rms(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def eps(inv, true):
    b = np.array(inv)
    # a = np.array(true)
    # if len(inv) > 10000:
    #     a = pg.interpolate(mesh, a, meshj.cellCenters()).array()
    #     a = a[cov == 1]
    #     b = b[cov == 1]
    #     return r"$\epsilon$ = %e" % rms(a, b)

    # a = []
    # b = []
    #
    # points_all = np.column_stack((
    #     pg.x(meshj.cellCenters()),
    #     pg.y(meshj.cellCenters()),
    # ))
    #
    # for cell in meshj.cells():
    #     if hull_path.contains_point(points_all[cell.id()]):
    #         a.append(true_int[cell.id()])
    #         b.append(inv[cell.id()])
    if np.allclose(b.min(), 0):
        min = 0
    else:
        min = b.min()
    return "min: %.2f\nmax: %.2f" % (min, b.max())


fre = 1 - fwe - fae - fie

allvel = list(veltrue) + list(velest[cov > 0]) + list(veljoint[cov > 0])
allrho = list(rhotrue) + list(rhoest[cov > 0]) + list(rhojoint[cov > 0])
allfa = list(fa) + list(fae[cov > 0]) + list(faj[cov > 0])
allfw = list(fw) + list(fwe[cov > 0]) + list(fwj[cov > 0])
allfi = list(fi) + list(fie[cov > 0]) + list(fij[cov > 0])
allfr = list(fr) + list(fre[cov > 0]) + list(frj[cov > 0])

im = draw(grid.axes_row[0][0], mesh, veltrue, cmap="viridis", **lim(allvel),
          logScale=False)
draw(grid.axes_row[0][1], meshj, velest, cmap="viridis", **lim(allvel),
     logScale=False, coverage=cov)
draw(grid.axes_row[0][2], meshj, veljoint, cmap="viridis", **lim(allvel),
     logScale=False, coverage=cov)
cb = fig.colorbar(im, cax=grid.cbar_axes[0])
update_ticks(cb, label=labels[0])

im = draw(grid.axes_row[1][0], mesh, rhotrue, cmap="Spectral_r", **lim(allrho),
          logScale=True)
draw(grid.axes_row[1][1], meshj, rhoest, cmap="Spectral_r", **lim(allrho),
     logScale=True, coverage=cov)
draw(grid.axes_row[1][2], meshj, rhojoint, cmap="Spectral_r", **lim(allrho),
     logScale=True, coverage=cov)
cb = fig.colorbar(im, cax=grid.cbar_axes[1], aspect=5)
update_ticks(cb, log=True, label=labels[1])

im = draw(grid.axes_row[2][0], mesh, fa, logScale=False, cmap="Greens",
          **lim(allfa))
add_inner_title(grid.axes_row[2][0], eps(fa, fa), loc=4, frame=False, c="k",
                size=config["fontsize"])
draw(grid.axes_row[2][1], meshj, fae, logScale=False, cmap="Greens",
     **lim(allfa), coverage=cov)
add_inner_title(grid.axes_row[2][1], eps(fae, fa), loc=4, frame=False, c="w",
                size=config["fontsize"])
draw(grid.axes_row[2][2], meshj, faj, logScale=False, cmap="Greens",
     **lim(allfa), coverage=cov)
add_inner_title(grid.axes_row[2][2], eps(faj, fa), loc=4, frame=False, c="w",
                size=config["fontsize"])
cb = fig.colorbar(im, cax=grid.cbar_axes[2])
update_ticks(cb, label=labels[2])

im = draw(grid.axes_row[3][0], mesh, fi, logScale=False, cmap="Purples",
          **lim(allfi))
add_inner_title(grid.axes_row[3][0], eps(fi, fi), loc=4, frame=False, c="w",
                size=config["fontsize"])
draw(grid.axes_row[3][1], meshj, fie, logScale=False, cmap="Purples",
     **lim(allfi), coverage=cov)
add_inner_title(grid.axes_row[3][1], eps(fie, fi), loc=4, frame=False, c="w",
                size=config["fontsize"])
draw(grid.axes_row[3][2], meshj, fij, logScale=False, cmap="Purples",
     **lim(allfi), coverage=cov)
add_inner_title(grid.axes_row[3][2], eps(fij, fi), loc=4, frame=False, c="w",
                size=config["fontsize"])
cb = fig.colorbar(im, cax=grid.cbar_axes[3])
update_ticks(cb, label=labels[3])

im = draw(grid.axes_row[4][0], mesh, fw, logScale=False, cmap="Blues",
          **lim(allfw))
add_inner_title(grid.axes_row[4][0], eps(fw, fw), loc=4, frame=False, c="k",
                size=config["fontsize"])
draw(grid.axes_row[4][1], meshj, fwe, logScale=False, cmap="Blues",
     **lim(allfw), coverage=cov)
add_inner_title(grid.axes_row[4][1], eps(fwe, fw), loc=4, frame=False, c="w",
                size=config["fontsize"])
draw(grid.axes_row[4][2], meshj, fwj, logScale=False, cmap="Blues",
     **lim(allfw), coverage=cov)
add_inner_title(grid.axes_row[4][2], eps(fwj, fw), loc=4, frame=False, c="w",
                size=config["fontsize"])
cb = fig.colorbar(im, cax=grid.cbar_axes[4])
update_ticks(cb, label=labels[4])

im = draw(grid.axes_row[5][0], mesh, fr, logScale=False, cmap="Oranges",
          **lim(allfr))
add_inner_title(grid.axes_row[5][0], eps(fr, fr), loc=4, frame=False, c="w",
                size=config["fontsize"])
draw(grid.axes_row[5][1], meshj, fre, logScale=False, cmap="Oranges",
     **lim(allfr), coverage=cov)
add_inner_title(grid.axes_row[5][1], eps(fre, fr), loc=4, frame=False, c="w",
                size=config["fontsize"])
draw(grid.axes_row[5][2], meshj, frj, logScale=False, cmap="Oranges",
     **lim(allfr), coverage=cov)
add_inner_title(grid.axes_row[5][2], eps(frj, fr), loc=4, frame=False, c="w",
                size=config["fontsize"])
cb = fig.colorbar(im, cax=grid.cbar_axes[5])
update_ticks(cb, label=labels[5])

for ax, title in zip(grid.axes_row[0], [
        "True model", "Conventional inversion + 4PM",
        "Petrophysical joint inversion"
]):
    ax.set_title(title, fontsize=config["fontsize"], fontweight="bold")

labs = [
    "inverted", "inverted", "transformed", "transformed", "transformed",
    "assumed"
]
for ax, lab in zip(grid.axes_column[1], labs):
    add_inner_title(ax, lab, loc=3, size=config["fontsize"], frame=False,
                    c="w")

labs = [
    "transformed", "transformed", "inverted", "inverted", "inverted",
    "assumed and fixed"
]
labs = [
    "transformed", "transformed", "inverted", "inverted", "inverted",
    "inverted"
]
for ax, lab in zip(grid.axes_column[2], labs):
    add_inner_title(ax, lab, loc=3, size=config["fontsize"], frame=False,
                    c="w")

for ax in grid.axes_all:
    ax.set_facecolor("0.5")
    ax.plot(sensors, np.zeros_like(sensors), 'kv', ms=3)
    ax.set_aspect(1.5)
    ax.tick_params(axis='both', which='major', pad=-3)
    ax.set_xticks([25, 50, 75, 100, 125])

for row in grid.axes_row[:-1]:
    for ax in row:
        ax.xaxis.set_visible(False)

for ax in grid.axes_column[-1]:
    ax.yaxis.set_visible(False)

for ax in grid.axes_row[-1]:
    ax.set_xlabel("x (m)")

for ax, label in zip(grid.axes_column[0], long_labels):
    #    old_labels = [x.get_text() for x in list(ax.get_yticklabels())]
    #    print(old_labels)
    #    new_ticks = [x.replace("-", "") for x in old_labels]
    #    ax.set_yticklabels(new_ticks)
    ax.set_yticks([-5, -15])
    ax.set_yticklabels([" 5", "15"])
    ax.set_ylabel("Depth (m)", labelpad=1)
    add_inner_title(ax, label, loc=3)

# for ax in grid.axes_column[1][2:]:
#     # Mask unphysical values
#     im = draw(ax, meshj, est["mask"], coverage=cov, logScale=False,
#          cmap="gray_r", grid=False)

fig.tight_layout()
fig.show()
# fig.savefig("4PM_joint_inversion.png", dpi=150, bbox_inches="tight")
fig.savefig("4PM_joint_inversion.pdf", dpi=300, bbox_inches="tight")
# pg.wait()
