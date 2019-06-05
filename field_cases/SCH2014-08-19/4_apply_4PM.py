#############################################
# to find "invlib" in the main folder
import sys, os
path = os.popen("git rev-parse --show-toplevel").read().strip("\n")
sys.path.insert(0, path)
#############################################

import numpy as np

import pygimli as pg
from invlib import FourPhaseModel

pd = pg.load("paraDomain_1.bms")
resinv = np.loadtxt("res_conventional.dat")
vest = np.loadtxt("vel_conventional.dat")

phi = 0.53

# Save some stuff
# fpm = FourPhaseModel(phi=phi, va=300., vi=3500., vw=1500, m=1.56, n=2,
                     # rhow=57.5, vr=6000)
fpm = FourPhaseModel(phi=phi, va=300., vi=3500., vw=1500, m=1.4, n=2.4,
                     rhow=60, vr=6000)
fae, fie, fwe, maske = fpm.all(resinv, vest)
print(np.min(fwe), np.max(fwe))
np.savez("conventional.npz", vel=np.array(vest), rho=np.array(resinv), fa=fae,
         fi=fie, fw=fwe, mask=maske)
