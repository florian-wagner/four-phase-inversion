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

# Model creation
world = mt.createWorld([0, -30], [150, 0], layers=[-5, -15], worldMarker=False)
block = mt.createPolygon([[60, -5], [90, -5], [100, -15], [50, -15]],
                         isClosed=True)
geom = mt.mergePLC([world, block])
geom.addRegionMarker((80, -2.5), 0)
geom.addRegionMarker((20, -10), 1)
geom.addRegionMarker((80, -10), 2)
geom.addRegionMarker((120, -10), 3)
geom.addRegionMarker((80, -25), 4)
geom.save("geom.bms")

mesh = mt.createMesh(geom, area=1.0)

pg.show(mesh, markers=True)
pg.wait()

# Model creation based on pore fractions
philayers = np.array([0.4, 0.3, 0.3, 0.3, 0.2])
frlayers = 1 - philayers
fwlayers = np.array([0.3, 0.18, 0.02, 0.1, 0.02])
filayers = np.array([0.0, 0.1, 0.28, 0.18, 0.18])
falayers = philayers - fwlayers - filayers

falayers[np.isclose(falayers, 0.0)] = 0.0

print(falayers)

fpm = FourPhaseModel(phi=philayers)

print(falayers + filayers + fwlayers + frlayers)
rholayers = fpm.rho(fwlayers, filayers, falayers, frlayers)
vellayers = 1. / fpm.slowness(fwlayers, filayers, falayers, frlayers)

print(rholayers)
print(vellayers)

def to_mesh(data):
    return data[mesh.cellMarkers()]

rhotrue = to_mesh(rholayers)
veltrue = to_mesh(vellayers)


# %%
# Save sensors, true model and mesh

fa = to_mesh(falayers)
fi = to_mesh(filayers)
fw = to_mesh(fwlayers)
fr = to_mesh(frlayers)

fpm.fr = fr
fpm.phi = 1 - fr
fpm.show(mesh, rhotrue, veltrue)

assert np.allclose(fa + fi + fw + fr, 1)

np.savez("true_model.npz", rho=rhotrue, vel=veltrue, fa=fa, fi=fi, fw=fw, fr=fr)

sensors = np.arange(10, 141, 3.5)
sensors.dump("sensors.npy")

mesh.save("mesh.bms")
np.savetxt("rhotrue.dat", rhotrue)
np.savetxt("veltrue.dat", veltrue)
