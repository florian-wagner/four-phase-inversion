import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt

from pybert.manager import ERTManager
from pygimli.physics import Refraction
from pygimli.physics.traveltime import createRAData

mesh = pg.load("mesh.bms")
sensors = np.load("sensors.npy", allow_pickle=True)
rhotrue = np.loadtxt("rhotrue.dat")
veltrue = np.loadtxt("veltrue.dat")

pg.boxprint("Simulate apparent resistivities")

# Create more realistic data set
ertScheme = pb.createData(sensors, "dd", spacings=[1,2,4])
k = pb.geometricFactors(ertScheme)
ertScheme.markInvalid(pg.abs(k) > 5000)
ertScheme.removeInvalid()

ert = ERTManager()

# Create suitable mesh for ert forward calculation
# NOTE: In the published results paraMaxCellSize=1.0 was used, which is
# increased here to allow testing on Continuous Integration services.
meshERTFWD = mt.createParaMesh(ertScheme, quality=33.5, paraMaxCellSize=2.0,
                               paraDX=0.2, boundaryMaxCellSize=50,
                               smooth=[1, 10], paraBoundary=30)
pg.show(meshERTFWD)

res = pg.RVector()
pg.interpolate(mesh, rhotrue, meshERTFWD.cellCenters(), res)
res = mt.fillEmptyToCellArray(meshERTFWD, res, slope=True)
ert.setMesh(meshERTFWD)
ert.fop.createRefinedForwardMesh()
ertData = ert.simulate(meshERTFWD, res, ertScheme, noiseLevel=0.05,
                       noiseAbs=0.0)
ertData.save("erttrue.dat")
ert.setData(ertData)
ert.setMesh(meshERTFWD)
ert.inv.setData(ertData("rhoa"))

pg.boxprint("Simulate traveltimes")
meshRSTFWD = pg.Mesh()
meshRSTFWD.createMeshByMarker(meshERTFWD, 2)

vel = pg.RVector()
pg.interpolate(mesh, veltrue, meshRSTFWD.cellCenters(), vel)
vel = mt.fillEmptyToCellArray(meshRSTFWD, vel, slope=False)

ttScheme = createRAData(sensors)
rst = Refraction(verbose=True)

error = 0.0005 # = 0.5 ms
meshRSTFWD.createSecondaryNodes(3)
ttData = rst.simulate(meshRSTFWD, 1. / vel, ttScheme,
                      noisify=True, noiseLevel=0.0, noiseAbs=error)
ttData.set("err", np.ones(ttData.size()) * error)

rst.setData(ttData)
rst.dataContainer.save("tttrue.dat")
