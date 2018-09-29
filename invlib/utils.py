import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from matplotlib.path import Path
from matplotlib.patheffects import withStroke
from scipy.spatial import ConvexHull

import pygimli as pg


def NN_interpolate(inmesh, indata, outmesh, nan=99.9):
    """ Nearest neighbor interpolation. """
    outdata = []
    for pos in outmesh.cellCenters():
        cell = inmesh.findCell(pos)
        if cell:
            outdata.append(indata[cell.id()])
        else:
            outdata.append(nan)
    return np.array(outdata)


def rst_cov(mesh, cov):
    """ Simplify refraction coverage with convex hull. """
    points_all = np.column_stack((
        pg.x(mesh.cellCenters()),
        pg.y(mesh.cellCenters()),
    ))

    points = points_all[np.nonzero(cov)[0]]
    hull = ConvexHull(points)
    hull_path = Path(points[hull.vertices])

    covs = []
    for cell in mesh.cells():
        if hull_path.contains_point(points_all[cell.id()]):
            covs.append(1)
        else:
            covs.append(0)
    return np.array(covs)


def add_inner_title(ax, title, loc, size=None, c="k", frame=True, **kwargs):
    """ Add inner title to plot. """
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
