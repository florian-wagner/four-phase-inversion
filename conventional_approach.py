import matplotlib.pyplot as plt
import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from petro import FourPhaseModel, testFourPhaseModel
from pybert.manager import ERTManager
from pygimli.physics import Refraction
from pygimli.physics.traveltime import createRAData

# Model creation
geom = mt.createWorld([0, -15], [117, 0], layers=[-5, -10], worldMarker=False)
mesh = mt.createMesh(geom, area=1.0)

for cell in mesh.cells():
    if cell.marker() == 3:
        cell.setMarker(4)
    if cell.marker() == 2 and cell.center().x() > 50:
        cell.setMarker(3)

mesh.save("mesh.bms")

rholayers = np.array([0, 8000, 200000, 2000, 5000])  # 0 is a dummy value
vellayers = np.array([0, 1750, 3500, 2000, 4000])  # 0 is a dummy value

rhotrue = rholayers[mesh.cellMarkers()]
veltrue = vellayers[mesh.cellMarkers()]

fpm = FourPhaseModel()
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
meshERT = mt.createParaMesh(ertScheme, quality=33, paraMaxCellSize=3.0,
                            boundaryMaxCellSize=50, smooth=[1, 2], paraBoundary=15)

res = pg.RVector()
pg.interpolate(mesh, rhotrue, meshERT.cellCenters(), res)
res = mt.fillEmptyToCellArray(meshERT, res, slope=True)
# pg.show(meshERT, res)
ertData = ert.simulate(meshERT, res, ertScheme, noiseLevel=0.01, noiseAbs=0.0)
ertData.save("erttrue.dat")
ert.setData(ertData)
ert.setMesh(meshERT)
ert.inv.setData(ertData("rhoa"))
ert.fop.regionManager().setZWeight(0.5)
resinv = ert.invert(ertData, mesh=meshERT, lam=5, zWeight=0.5)
print("ERT chi:", ert.inv.chi2())
print("ERT rms:", ert.inv.relrms())
# resinv = ert.inv.runChi1()
rhoest = pg.interpolate(ert.paraDomain, resinv, mesh.cellCenters())

print("-Simulate refraction seismics" + "-" * 50)
vel = pg.RVector()
pg.interpolate(mesh, veltrue, meshERT.cellCenters(), vel)
vel = mt.fillEmptyToCellArray(meshERT, vel, slope=True)

ttScheme = createRAData(sensors)
rst = Refraction(verbose=True)
# rst.useFMM(True)
ttData = rst.simulate(meshERT, 1. / vel, ttScheme, noiseLevel=0.01, noiseAbs=0.0)
ttData.save("tttrue.dat")

rst.setData(ttData)
rst.createMesh(depth=15., paraMaxCellSize=3.)
rst.fop.regionManager().setZWeight(0.5)
rst.inv.setData(ttData("s"))
# ttData.set("err", np.ones(len(ttData("s"))) * 0.001)
rst.inv.setRelativeError(0.01)
vest = rst.invert(ttData, zWeight=0.5, lam=100, useGradient=True, vtop=1000, vbottom=4000)
# vest = rst.inv.runChi1()
print("RST chi:", rst.inv.chi2())
print("RST rms:", rst.inv.relrms())
velest = pg.interpolate(rst.mesh, vest, mesh.cellCenters())
fae, fie, fwe, maske = fpm.all(rhoest, velest)
np.savez("conventional.npz", vel=velest, rho=rhoest, fa=fae, fi=fie, fw=fwe)
