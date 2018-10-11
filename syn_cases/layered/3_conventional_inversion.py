#############################################
# to find "invlib" in the main folder
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
#############################################


import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt

from invlib import FourPhaseModel
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

if len(sys.argv) > 1:
    scenario = "Fig2"
    phi = 0.3 # Porosity assumed to calculate fi, fa, fw with 4PM
else:
    scenario = "Fig1"
    frtrue = np.load("true_model.npz")["fr"]
    phi = 1 - pg.interpolate(mesh, frtrue, meshRST.cellCenters()).array()

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
rst.setMesh(meshRST)

veltrue = np.loadtxt("veltrue.dat")
startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue), np.max(veltrue))
np.savetxt("rst_startmodel.dat", 1/startmodel)
vest = rst.invert(ttData, zWeight=1, startModel=startmodel, maxIter=maxIter, lam=50)
print("RST chi:", rst.inv.chi2())
print("RST rms:", rst.inv.relrms())

# Save some stuff
fpm = FourPhaseModel(phi=phi)
fae, fie, fwe, maske = fpm.all(resinv.array(), vest.array())
np.savez("conventional_%s.npz" % scenario, vel=np.array(vest), rho=np.array(resinv), fa=fae, fi=fie, fw=fwe, mask=maske)
rst.rayCoverage().save("rst_coverage_%s.dat" % scenario)
