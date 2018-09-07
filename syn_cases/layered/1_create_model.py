#############################################
# to find "invlib" in the main folder
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
#############################################

import numpy as np

import pygimli as pg
import pygimli.meshtools as mt

from invlib import FourPhaseModel

# matplotlib.use("Agg")

# %%
# Model creation
world = mt.createWorld([0, -15], [117, 0], layers=[-5, -10], worldMarker=False)
block = mt.createPolygon([[0, -5], [40, -5], [60, -10], [0, -10]],
                         isClosed=True)
geom = mt.mergePLC([world, block])
mesh = mt.createMesh(geom, area=1.0)

for cell in mesh.cells():
    if cell.marker() == 3:
        cell.setMarker(4)
    if cell.marker() == 0:
        cell.setMarker(3)

pg.show(mesh, markers=True, savefig="mesh_with_markers.png")

fpm = FourPhaseModel()

rholayers = np.array([0, 8000, 200000, 2000, 6000])  # 0 is a dummy value
vellayers = np.array([0, 1750, 3500, 2000, 4200])  # 0 is a dummy value

# Model creation based on pore fractions
# fwlayers = np.array([0, 0.05, 0, 0.2, 0.05])
# filayers = np.array([0, 0.2, 0.37, 0, 0.37])
# falayers = np.ones(len(fwlayers)) - fwlayers - filayers
#
# rholayers = fpm.rho(fwlayers, filayers, falayers)
# rholayers = fpm.slowness(fwlayers, filayers, falayers)

rhotrue = rholayers[mesh.cellMarkers()]
veltrue = vellayers[mesh.cellMarkers()]

# Save sensors, true model and mesh
fa, fi, fw, mask = fpm.all(rhotrue, veltrue)
np.savez("true_model.npz", rho=rhotrue, vel=veltrue, fa=fa, fi=fi, fw=fw)

sensors = np.linspace(10, 107, 30)
sensors.dump("sensors.npy")

mesh.save("mesh.bms")
np.savetxt("rhotrue.dat", rhotrue)
np.savetxt("veltrue.dat", veltrue)
