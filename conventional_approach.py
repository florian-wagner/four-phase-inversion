import matplotlib
import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from petro import FourPhaseModel
from pybert.manager import ERTManager
from pygimli.physics import Refraction
from pygimli.physics.traveltime import createRAData

# matplotlib.use("Agg")

pg.setThreadCount(8)

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

# Mesh for inversion
meshRST = mt.createParaMesh(ertScheme, paraDepth=15, paraDX=0.2, smooth=[1, 2],
                            paraMaxCellSize=.5, quality=33.5, boundary=0,
                            paraBoundary=3)
meshRST.save("paraDomain.bms")

# meshERT = mt.createParaMesh(ertScheme, quality=33.5, paraMaxCellSize=1.0,
#                             paraDX=0.2, boundary=50, smooth=[1, 10],
#                             paraBoundary=3)
meshERT = mt.appendTriangleBoundary(meshRST, xbound=500, ybound=500, quality=33.5, isSubSurface=True)
meshERT.save("meshERT.bms")

# CM = pg.utils.geostatistics.covarianceMatrix(ert.paraDomain, I=[40, 3])
# ert.fop.setConstraints(pg.matrix.Cm05Matrix(CM))

resinv = ert.invert(ertData, mesh=meshERT, lam=5, zWeight=1)

print("ERT chi:", ert.inv.chi2())
print("ERT rms:", ert.inv.relrms())
# resinv = ert.inv.runChi1()
# rhoest = pg.interpolate(ert.paraDomain, resinv, mesh.cellCenters())
# ert_cov = pg.interpolate(ert.paraDomain, ert.coverageDC(), mesh.cellCenters())
# ert.coverageDC().save("ert_coverage.dat")
np.savetxt("ert_coverage.dat", ert.coverageDC())

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

error = 0.0001 # seconds
ttData = rst.simulate(meshRSTFWD, 1. / vel, ttScheme, noisify=True,
                      noiseLevel=0.0, noiseAbs=error)
ttData.set("err", np.ones(ttData.size()) * error)

rst.setData(ttData)
rst.dataContainer.save("tttrue.dat")
ttData = rst.dataContainer

# INVERSION
# meshRST = pg.Mesh()
# meshRST.createMeshByMarker(meshERTFWD, 2)

# meshRST = mt.createParaMesh(ertScheme, paraDepth=15, paraDX=0.2,
#                             smooth=[1, 2], paraMaxCellSize=1, quality=33.5,
#                             boundary=0, paraBoundary=3)

rst.setMesh(meshRST)
# rst.fop.regionManager().setZWeight(0.5)
rst.inv.setData(ttData("t"))
rst.inv.setMaxIter(50)
# ttData.set("err", np.ones(len(ttData("s"))) * 0.001)
# rst.inv.setRelativeError(0.001)

# CM = pg.utils.geostatistics.covarianceMatrix(rst.mesh, I=[40, 3])
# rst.fop.setConstraints(pg.matrix.Cm05Matrix(CM))
from pygimli.physics.traveltime.ratools import createGradientModel2D
startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue), np.max(veltrue))
np.savetxt("rst_startmodel.dat", 1/startmodel)
vest = rst.invert(ttData, zWeight=1, startModel=startmodel, lam=50)
# vest = rst.inv.runChi1()
print("RST chi:", rst.inv.chi2())
print("RST rms:", rst.inv.relrms())
velest = pg.interpolate(rst.mesh, vest, mesh.cellCenters())
fae, fie, fwe, maske = fpm.all(resinv, vest)
np.savez("conventional.npz", vel=vest, rho=resinv, fa=fae, fi=fie, fw=fwe, mask=maske)

# rst_cov = pg.interpolate(rst.paraDomain(), rst.rayCoverage(),
                         # mesh.cellCenters())
rst.rayCoverage().save("rst_coverage.dat")
