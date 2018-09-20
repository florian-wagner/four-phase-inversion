#############################################
# to find "invlib" in the main folder
import sys, os
sys.path.insert(0, os.path.abspath("../.."))
#############################################

import numpy as np

import pybert as pb
import pygimli as pg
import pygimli.meshtools as mt

from invlib import FourPhaseModel, NN_interpolate
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
maxIter = 50
# phi = 0.4 # Porosity assumed to calculate fi, fa, fw with 4PM

frtrue = np.load("true_model.npz")["fr"]
phi = 1 - pg.interpolate(mesh, frtrue, meshRST.cellCenters()).array()

############


# meshERT = mt.createParaMesh(ertScheme, quality=33.5, paraMaxCellSize=1.0,
#                             paraDX=0.2, boundary=50, smooth=[1, 10],
#                             paraBoundary=3)
meshERT = mt.appendTriangleBoundary(meshRST, xbound=500, ybound=500,
                                    quality=34, isSubSurface=True)
meshERT.save("meshERT.bms")

ert = ERTManager()
# CM = pg.utils.geostatistics.covarianceMatrix(ert.paraDomain, I=[40, 3])
# ert.fop.setConstraints(pg.matrix.Cm05Matrix(CM))

resinv = ert.invert(ertData, mesh=meshERT, lam=5, zWeight=1, maxIter=maxIter)

print("ERT chi:", ert.inv.chi2())
print("ERT rms:", ert.inv.relrms())
# resinv = ert.inv.runChi1()
# rhoest = pg.interpolate(ert.paraDomain, resinv, mesh.cellCenters())
# ert_cov = pg.interpolate(ert.paraDomain, ert.coverageDC(), mesh.cellCenters())
# ert.coverageDC().save("ert_coverage.dat")
np.savetxt("ert_coverage.dat", ert.coverageDC())

rst = Refraction("tttrue.dat", verbose=True)

# rst.useFMM(True)
# rst.fop.createRefinedForwardMesh()

ttData = rst.dataContainer

# INVERSION
rst.setMesh(meshRST)

# CM = pg.utils.geostatistics.covarianceMatrix(rst.mesh, I=[40, 3])
# rst.fop.setConstraints(pg.matrix.Cm05Matrix(CM))
from pygimli.physics.traveltime.ratools import createGradientModel2D
veltrue = np.loadtxt("veltrue.dat")
startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue), np.max(veltrue))
np.savetxt("rst_startmodel.dat", 1/startmodel)
vest = rst.invert(ttData, zWeight=1, startModel=startmodel, maxIter=maxIter, lam=50)
# vest = rst.inv.runChi1()
print("RST chi:", rst.inv.chi2())
print("RST rms:", rst.inv.relrms())

# Save some stuff
fpm = FourPhaseModel(phi=phi)
fae, fie, fwe, maske = fpm.all(resinv.array(), vest.array())
np.savez("conventional.npz", vel=np.array(vest), rho=np.array(resinv), fa=fae, fi=fie, fw=fwe, mask=maske)
rst.rayCoverage().save("rst_coverage.dat")
