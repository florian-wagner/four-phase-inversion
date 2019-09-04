#############################################
# to find "invlib" in the main folder
import sys, os
path = os.popen("git rev-parse --show-toplevel").read().strip("\n")
sys.path.insert(0, path)
#############################################

from fpinv import FourPhaseModel

# Inversion settings
zWeight = 0.25 # four times more smoothing in lateral direction
erte = 0.03 # 3 %
rste = 0.0003 # 0.3 ms
maxIter = 50 # maximum number of iterations

# Petrophysical settings
poro = 0.53 # porosity
phi = poro
fpm = FourPhaseModel(phi=poro, va=300., vi=3500., vw=1500, m=1.4, n=2.4,
                     rhow=60, vr=6000)
                     
elevation_5198 = 0.12
elevation_5000 = 0.65
depth_5198 = 2.1 + elevation_5198 # topo
depth_5000 = 2.2 + elevation_5000 # topo


def plot_boreholes(ax, **kwargs):
    rad = 2
    ax.plot([10, 10], [-10, -elevation_5198], "k-", **kwargs)
    ax.plot([26, 26], [-20, -elevation_5000], "k-", **kwargs)
    ax.plot([10 - rad, 10 + rad], [-depth_5198, -depth_5198], "k-", **kwargs)
    ax.plot([26 - rad, 26 + rad], [-depth_5000, -depth_5000], "k-", **kwargs)
