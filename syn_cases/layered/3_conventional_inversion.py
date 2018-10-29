

import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt

from pygimli.physics.traveltime.ratools import createGradientModel2D
from pybert.manager import ERTManager
from pygimli.physics import Refraction

ertData = pb.load("erttrue.dat")
print(ertData)
mesh = pg.load("mesh.bms")
depth = mesh.ymax() - mesh.ymin()

# Build inversion meshes
plc = mt.createParaMeshPLC(ertData, paraDepth=depth, paraDX=0.1,
                           boundary=0, paraBoundary=2)

rect = mt.createRectangle([mesh.xmin(), mesh.ymin()],
                          [mesh.xmax(), mesh.ymax()])
geom = mt.mergePLC([plc, rect])

meshRST = mt.createMesh(geom, quality=34, area=.5, smooth=[1,2])
for cell in meshRST.cells():
    cell.setMarker(2)

pg.show(meshRST)
meshRST.save("paraDomain.bms")

############
# Settings
maxIter = 15
############

meshERT = mt.appendTriangleBoundary(meshRST, xbound=500, ybound=500,
                                    quality=34, isSubSurface=True)
meshERT.save("meshERT.bms")

ert = ERTManager()

resinv = ert.invert(ertData, mesh=meshERT, lam=5, zWeight=1, maxIter=maxIter)

print("ERT chi:", ert.inv.chi2())
print("ERT rms:", ert.inv.relrms())

# Seismic Inversion
rst = Refraction("tttrue.dat", verbose=True)
ttData = rst.dataContainer
rst.setMesh(meshRST, secNodes=3)

veltrue = np.loadtxt("veltrue.dat")
startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue), np.max(veltrue))
np.savetxt("rst_startmodel.dat", 1/startmodel)
vest = rst.invert(ttData, zWeight=1, startModel=startmodel, maxIter=maxIter, lam=50)
print("RST chi:", rst.inv.chi2())
print("RST rms:", rst.inv.relrms())

rst.rayCoverage().save("rst_coverage.dat")
np.savetxt("res_conventional.dat", resinv)
np.savetxt("vel_conventional.dat", vest)
