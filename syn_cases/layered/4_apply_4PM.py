#############################################
# to find "invlib" in the main folder
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
#############################################

import numpy as np
import pygimli as pg
from invlib import FourPhaseModel

mesh = pg.load("mesh.bms")
pd = pg.load("paraDomain.bms")
resinv = np.loadtxt("res_conventional.dat")
vest = np.loadtxt("vel_conventional.dat")

if len(sys.argv) > 1:
    scenario = "Fig2"
    phi = 0.3  # Porosity assumed to calculate fi, fa, fw with 4PM
else:
    scenario = "Fig1"
    frtrue = np.load("true_model.npz")["fr"]
    phi = 1 - pg.interpolate(mesh, frtrue, pd.cellCenters()).array()

# Save some stuff
fpm = FourPhaseModel(phi=phi)
fae, fie, fwe, maske = fpm.all(resinv.array(), vest.array())
np.savez("conventional_%s.npz" % scenario, vel=np.array(vest),
         rho=np.array(resinv), fa=fae, fi=fie, fw=fwe, mask=maske)
