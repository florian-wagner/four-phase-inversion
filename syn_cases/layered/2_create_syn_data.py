#############################################
# to find "invlib" in the main folder
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
#############################################

import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt

from pybert.manager import ERTManager
from pygimli.physics import Refraction
from pygimli.physics.traveltime import createRAData

mesh = pg.load("mesh.bms")
sensors = np.load("sensors.npy")
rhotrue = np.loadtxt("rhotrue.dat")
veltrue = np.loadtxt("veltrue.dat")

print("-Simulate ERT" + "-" * 50)
ertScheme = pb.createData(sensors, "dd")
ert = ERTManager()

# Create suitable mesh for ert forward calculation
meshERTFWD = mt.createParaMesh(ertScheme, quality=33.5, paraMaxCellSize=1.0,
                               paraDX=0.2, boundaryMaxCellSize=50,
                               smooth=[1, 10], paraBoundary=30)
pg.show(meshERTFWD)

res = pg.RVector()
pg.interpolate(mesh, rhotrue, meshERTFWD.cellCenters(), res)
res = mt.fillEmptyToCellArray(meshERTFWD, res, slope=True)
# pg.show(meshERTFWD, res)
ert.setMesh(meshERTFWD)
ert.fop.createRefinedForwardMesh()
ertData = ert.simulate(meshERTFWD, res, ertScheme, noiseLevel=0.05,
                       noiseAbs=0.0)
ertData.save("erttrue.dat")
ert.setData(ertData)
ert.setMesh(meshERTFWD)
ert.inv.setData(ertData("rhoa"))
# ert.fop.regionManager().setZWeight(0.5)


print("-Simulate refraction seismics" + "-" * 50)
meshRSTFWD = pg.Mesh()
meshRSTFWD.createMeshByMarker(meshERTFWD, 2)

vel = pg.RVector()
pg.interpolate(mesh, veltrue, meshRSTFWD.cellCenters(), vel)
vel = mt.fillEmptyToCellArray(meshRSTFWD, vel, slope=False)

ttScheme = createRAData(sensors)
rst = Refraction(verbose=True)
# rst.useFMM(True)
# rst.setMesh(meshRSTFWD)
# rst.fop.createRefinedForwardMesh()

error = 0.0005 # seconds
ttData = rst.simulate(meshRSTFWD, 1. / vel, ttScheme, noisify=True,
                      noiseLevel=0.0, noiseAbs=error)
ttData.set("err", np.ones(ttData.size()) * error)

rst.setData(ttData)
rst.dataContainer.save("tttrue.dat")
