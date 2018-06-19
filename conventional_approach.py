import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from petro import FourPhaseModel, testFourPhaseModel
from pybert.manager import ERTManager
from pygimli.physics import Refraction
from pygimli.physics.traveltime import createRAData

pg.setThreadCount(8)

# %%
# Model creation
world = mt.createWorld([0, -15], [117, 0], layers=[-5, -10], worldMarker=False)
block = mt.createPolygon([[0,-5],[40,-5],[60,-10],[0,-10]], isClosed=True)
geom = mt.mergePLC([world, block])
mesh = mt.createMesh(geom, area=1.0)

for cell in mesh.cells():
    if cell.marker() == 3:
        cell.setMarker(4)
    if cell.marker() == 0:
        cell.setMarker(3)

pg.show(mesh, markers=True)
# %%
mesh.save("mesh.bms")

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

fa, fi, fw, mask = fpm.all(rhotrue, veltrue)
np.savez("true_model.npz", rho=rhotrue, vel=veltrue, fa=fa, fi=fi, fw=fw)

# ERT
sensors = np.linspace(10, 107, 30)
sensors.dump("sensors.npy")

# %%
print("-Simulate ERT" + "-" * 50)
ertScheme = pb.createData(sensors, "dd")
ert = ERTManager()

# Create suitable mesh for ert forward calculation
meshERT = mt.createParaMesh(ertScheme, quality=33.5, paraMaxCellSize=1.0, paraDX=0.2,
                            boundaryMaxCellSize=50, smooth=[1, 10], paraBoundary=30)

pg.show(meshERT)

res = pg.RVector()
pg.interpolate(mesh, rhotrue, meshERT.cellCenters(), res)
res = mt.fillEmptyToCellArray(meshERT, res, slope=True)
# pg.show(meshERT, res)
ert.setMesh(meshERT)
ert.fop.createRefinedForwardMesh()
ertData = ert.simulate(meshERT, res, ertScheme, noiseLevel=0.05, noiseAbs=0.0)
ertData.save("erttrue.dat")
ert.setData(ertData)
ert.setMesh(meshERT)
ert.inv.setData(ertData("rhoa"))
# ert.fop.regionManager().setZWeight(0.5)

meshERT = mt.createParaMesh(ertScheme, quality=33.5, paraMaxCellSize=1.0, paraDX=0.2,
                            boundary=50, smooth=[1, 10], paraBoundary=3)
pg.show(meshERT, markers=True)
# CM = pg.utils.geostatistics.covarianceMatrix(ert.paraDomain, I=[40, 3])
# ert.fop.setConstraints(pg.matrix.Cm05Matrix(CM))

resinv = ert.invert(ertData, mesh=meshERT, lam=5, zWeight=1)

print("ERT chi:", ert.inv.chi2())
print("ERT rms:", ert.inv.relrms())
# resinv = ert.inv.runChi1()
rhoest = pg.interpolate(ert.paraDomain, resinv, mesh.cellCenters())
ert_cov = pg.interpolate(ert.paraDomain, ert.coverageDC(), mesh.cellCenters())
ert_cov.save("ert_coverage.dat")



print("-Simulate refraction seismics" + "-" * 50)
meshRST = pg.Mesh()
meshRST.createMeshByMarker(meshERT, 2)

vel = pg.RVector()
pg.interpolate(mesh, veltrue, meshRST.cellCenters(), vel)
vel = mt.fillEmptyToCellArray(meshRST, vel, slope=False)

ttScheme = createRAData(sensors)
rst = Refraction(verbose=True)
# rst.useFMM(True)
# rst.setMesh(meshRST)
# rst.fop.createRefinedForwardMesh()

ttData = rst.simulate(meshRST, 1. / vel, ttScheme, noisify=True, noiseLevel=0.0001, noiseAbs=0.001)

rst.setData(ttData)
rst.dataContainer.save("tttrue.dat")
ttData = rst.dataContainer

rst.createMesh(depth=15., paraMaxCellSize=1., quality=33.5, boundary=0, smooth=[1,10],
               paraDX=0.1, paraBoundary=3)
rst.mesh.save("paraDomain.bms")
# rst.fop.regionManager().setZWeight(0.5)
rst.inv.setData(ttData("s"))
rst.inv.setMaxIter(50)
# ttData.set("err", np.ones(len(ttData("s"))) * 0.001)
# rst.inv.setRelativeError(0.001)

# CM = pg.utils.geostatistics.covarianceMatrix(rst.mesh, I=[40, 3])
# rst.fop.setConstraints(pg.matrix.Cm05Matrix(CM))
vest = rst.invert(ttData, zWeight=1, lam=15, useGradient=True, vtop=1000, vbottom=4000)
# vest = rst.inv.runChi1()
print("RST chi:", rst.inv.chi2())
print("RST rms:", rst.inv.relrms())
velest = pg.interpolate(rst.mesh, vest, mesh.cellCenters())
fae, fie, fwe, maske = fpm.all(rhoest, velest)
np.savez("conventional.npz", vel=velest, rho=rhoest, fa=fae, fi=fie, fw=fwe)

rst_cov = pg.interpolate(rst.paraDomain(), rst.rayCoverage(), mesh.cellCenters())
rst_cov.save("rst_coverage.dat")
