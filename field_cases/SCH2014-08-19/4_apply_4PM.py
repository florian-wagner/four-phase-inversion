#############################################
# to find "invlib" in the main folder
import sys
import os
sys.path.insert(0, os.path.abspath("../.."))
#############################################

import numpy as np

import pygimli as pg
from invlib import FourPhaseModel

mesh = pg.load("mesh.bms")
pd = pg.load("paraDomain.bms")
resinv = np.loadtxt("res_conventional.dat")
vest = np.loadtxt("vel_conventional.dat")

grid = pg.load("grid.bms")
phigrid = np.loadtxt("phi_grid.dat")
phigrid[np.isnan(phigrid)] = 0.0
phi = pg.interpolate(grid, phigrid, pd.cellCenters()).array()

# pg.show(grid, phigrid)
# pg.show(pd, phi, label=True)
# pg.wait()
phi = 0.35

# Save some stuff
fpm = FourPhaseModel(phi=phi, va=300., vi=3500., vw=1500, m=1.4, n=2.4,
                     rhow=60, vr=6000)
fae, fie, fwe, maske = fpm.all(resinv, vest)
print(np.min(fwe), np.max(fwe))
np.savez("conventional.npz", vel=np.array(vest), rho=np.array(resinv), fa=fae,
         fi=fie, fw=fwe, mask=maske)
