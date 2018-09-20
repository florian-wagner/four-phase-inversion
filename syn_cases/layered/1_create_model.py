#############################################
# to find "invlib" in the main folder
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
#############################################

import numpy as np

import pygimli as pg
import pygimli.meshtools as mt
from invlib import FourPhaseModel

# matplotlib.use("Agg")

# Model creation
world = mt.createWorld([0, -30], [150, 0], layers=[-5, -15], worldMarker=False)
block = mt.createPolygon([[60, -5], [90, -5], [100, -15], [50, -15]],
                         isClosed=True)
geom = mt.mergePLC([world, block])

geom.addRegionMarker((80, -10), 5)
mesh = mt.createMesh(geom, area=1.0)

for cell in mesh.cells():
    if cell.marker() == 3:
        cell.setMarker(4)
    if cell.marker() == 0:
        cell.setMarker(3)

for cell in mesh.cells():
    cell.setMarker(cell.marker() - 1)

# pg.show(mesh, markers=True, savefig="mesh_with_markers.png")


# rholayers = np.array([8000, 200000, 2000, 6000])
# vellayers = np.array([1750, 3500, 2000, 4200])


# Model creation based on pore fractions
philayers = np.array([0.5, 0.4, 0.4, 0.3, 0.4])
frlayers = 1 - philayers
fwlayers = np.array([0.3, 0.2, 0.15, 0.02, 0.02])
filayers = np.array([0.05, 0.1, 0.2, 0.27, 0.3])
falayers = philayers - fwlayers - filayers

print(falayers)
assert (falayers >= 0).all()

fpm = FourPhaseModel(phi = philayers)

print(falayers + filayers + fwlayers)
rholayers = fpm.rho(fwlayers, filayers, falayers, frlayers)
vellayers = 1. / fpm.slowness(fwlayers, filayers, falayers, frlayers)

def to_mesh(data):
    return data[mesh.cellMarkers()]

rhotrue = to_mesh(rholayers)
veltrue = to_mesh(vellayers)

# %%
# Save sensors, true model and mesh

fa = to_mesh(falayers)
fi = to_mesh(filayers)
fw = to_mesh(fwlayers)
fr = 1 - fa - fi - fw

assert np.allclose(fa + fi + fw + fr, 1)

fpm.fr = fr
fpm.phi = 1 - fr
fpm.show(mesh, rhotrue, veltrue)

np.savez("true_model.npz", rho=rhotrue, vel=veltrue, fa=fa, fi=fi, fw=fw, fr=fr)

sensors = np.arange(10, 141, 3.5)
sensors.dump("sensors.npy")

mesh.save("mesh.bms")
np.savetxt("rhotrue.dat", rhotrue)
np.savetxt("veltrue.dat", veltrue)
