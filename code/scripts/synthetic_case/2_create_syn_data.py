import numpy as np

import pygimli as pg
import pygimli.meshtools as mt

from pygimli.physics import ert
from pygimli.physics import ERTManager
from pygimli.physics import TravelTimeManager
from pygimli.physics.traveltime import createRAData

mesh = pg.load("mesh.bms")
sensors = np.load("sensors.npy", allow_pickle=True)
rhotrue = np.loadtxt("rhotrue.dat")
veltrue = np.loadtxt("veltrue.dat")

pg.boxprint("Simulate apparent resistivities")

# Create more realistic data set
ertScheme = ert.createData(sensors, "dd", spacings=[1,2,4])
k = ert.createGeometricFactors(ertScheme)
ertScheme.markInvalid(pg.abs(k) > 5000)
ertScheme.removeInvalid()

ert = ERTManager()

# Create suitable mesh for ert forward calculation
# NOTE: In the published results paraMaxCellSize=1.0 was used, which is
# increased here to allow testing on Continuous Integration services.
meshERTFWD = mt.createParaMesh(ertScheme, quality=33.5, paraMaxCellSize=2.0,
                               paraDX=0.2, boundaryMaxCellSize=50,
                               paraBoundary=30)
pg.show(meshERTFWD)

res = pg.Vector()
pg.interpolate(mesh, rhotrue, meshERTFWD.cellCenters(), res)
res = mt.fillEmptyToCellArray(meshERTFWD, res, slope=True)
ert.setMesh(meshERTFWD)
ert.fop.createRefinedForwardMesh()
ertData = ert.simulate(mesh=meshERTFWD, res=res, scheme=ertScheme, noiseLevel=0.05,
                       noiseAbs=0.0)
ertData.save("erttrue.dat")
ert.setData(ertData)
ert.setMesh(meshERTFWD)
ert.inv.dataVals = ertData("rhoa")

pg.boxprint("Simulate traveltimes")
meshRSTFWD = pg.Mesh()
meshRSTFWD.createMeshByMarker(meshERTFWD, 2)

vel = pg.Vector()
pg.interpolate(mesh, veltrue, meshRSTFWD.cellCenters(), vel)
vel = mt.fillEmptyToCellArray(meshRSTFWD, vel, slope=False)

ttScheme = createRAData(sensors)
rst = TravelTimeManager(verbose=True)

error = 0.0005 # = 0.5 ms
meshRSTFWD.createSecondaryNodes(3)
ttData = rst.simulate(mesh=meshRSTFWD, slowness=1. / vel, scheme=ttScheme,
                      noisify=True, noiseLevel=0.0, noiseAbs=error)
ttData.set("err", np.ones(ttData.size()) * error)

rst.setData(ttData)
rst.fop.data.save("tttrue.dat")
