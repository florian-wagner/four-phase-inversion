import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from pybert.manager import ERTManager
from pygimli.physics import Refraction
from pygimli.physics.traveltime.ratools import createGradientModel2D

############
# Settings
maxIter = 15
zWeight = 0.25
############

ertData = pb.load("erttrue.dat")
print(ertData)
mesh = pg.load("mesh.bms")
depth = mesh.ymax() - mesh.ymin()

# %% Build inversion meshes
plc = mt.createParaMeshPLC(ertData, paraDepth=depth, paraDX=0.3, boundary=0,
                           paraBoundary=2)

for sensor in ertData.sensorPositions():
    plc.createNode([sensor.x(), sensor.y() - 0.1])

rect = mt.createRectangle([mesh.xmin(), mesh.ymin()],
                          [mesh.xmax(), mesh.ymax()])
geom = mt.mergePLC([plc, rect])

meshRST = mt.createMesh(geom, quality=34, area=0.5, smooth=[1, 2])
for cell in meshRST.cells():
    cell.setMarker(2)
for boundary in meshRST.boundaries():
    boundary.setMarker(0)

pg.show(meshRST)
print(meshRST)
# %%
meshRST.save("paraDomain.bms")

meshERT = mt.appendTriangleBoundary(meshRST, xbound=500, ybound=500,
                                    quality=34, isSubSurface=True)
meshERT.save("meshERT.bms")

# ERT inversion
ert = ERTManager()
ert.setMesh(meshERT)
ert.fop.createRefinedForwardMesh()

resinv = ert.invert(ertData, lam=100, zWeight=zWeight, maxIter=maxIter)
print("ERT chi: %.2f" % ert.inv.chi2())
print("ERT rms: %.2f" % ert.inv.relrms())
np.savetxt("res_conventional.dat", resinv)

1/0
# Seismic inversion
rst = Refraction("tttrue.dat", verbose=True)
ttData = rst.dataContainer
rst.setMesh(meshRST, secNodes=3)

veltrue = np.loadtxt("veltrue.dat")
startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue),
                                   np.max(veltrue))
np.savetxt("rst_startmodel.dat", 1 / startmodel)
vest = rst.invert(ttData, zWeight=zWeight, startModel=startmodel,
                  maxIter=maxIter, lam=180)
print("RST chi: %.2f" % rst.inv.chi2())
print("RST rms: %.2f" % rst.inv.relrms())

rst.rayCoverage().save("rst_coverage.dat")
np.savetxt("vel_conventional.dat", vest)
