import sys

import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt
from pygimli.physics import ERTManager
from pygimli.physics import TravelTimeManager
from pygimli.physics.traveltime.ratools import createGradientModel2D

############
# Settings
maxIter = 15
zWeight = 0.25
############

case = int(sys.argv[1])
ertData = pb.load("erttrue.dat")
print(ertData)
mesh = pg.load("mesh.bms")
depth = mesh.ymax() - mesh.ymin()

# %% Build inversion meshes
plc = mt.createParaMeshPLC(ertData, paraDepth=depth, paraDX=0.3, boundary=0,
                           paraBoundary=2)

if case == 1:
    for depth in (5, 15):
        start = plc.createNode(mesh.xmin(), -depth, 0.0)
        end = plc.createNode(mesh.xmax(), -depth, 0.0)
        plc.createEdge(start, end, marker=1)

for sensor in ertData.sensorPositions():
    plc.createNode([sensor.x(), sensor.y() - 0.1])

rect = mt.createRectangle([mesh.xmin(), mesh.ymin()],
                          [mesh.xmax(), mesh.ymax()], boundaryMarker=0)
geom = mt.mergePLC([plc, rect])

meshRST = mt.createMesh(geom, quality=34, area=1, smooth=[1, 2])

for cell in meshRST.cells():
    cell.setMarker(2)
for boundary in meshRST.boundaries():
    boundary.setMarker(0)

pg.show(meshRST)
print(meshRST)
# %%
meshRST.save("paraDomain_%d.bms" % case)

meshERT = mt.appendTriangleBoundary(meshRST, xbound=500, ybound=500,
                                    quality=34, isSubSurface=True)
meshERT.save("meshERT_%d.bms" % case)

# ERT inversion
ert = ERTManager()
ert.setMesh(meshERT)

resinv = ert.invert(ertData, lam=30, zWeight=zWeight, maxIter=maxIter)
print("ERT chi: %.2f" % ert.inv.chi2())
print("ERT rms: %.2f" % ert.inv.inv.relrms())
np.savetxt("res_conventional_%d.dat" % case, resinv)

# Seismic inversion
ttData = pg.DataContainer("tttrue.dat")
print(ttData)
rst = TravelTimeManager(verbose=True)
rst.setMesh(meshRST, secNodes=3)

veltrue = np.loadtxt("veltrue.dat")
startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue),
                                   np.max(veltrue))
np.savetxt("rst_startmodel_%d.dat" % case, 1 / startmodel)
vest = rst.invert(ttData, zWeight=zWeight, startModel=startmodel,
                  maxIter=maxIter, lam=220)
print("RST chi: %.2f" % rst.inv.chi2())
print("RST rms: %.2f" % rst.inv.inv.relrms())

rst.rayCoverage().save("rst_coverage_%d.dat" % case)
np.savetxt("vel_conventional_%d.dat" % case, vest)
