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
from pybert.manager import ERTManager
from pygimli.physics import Refraction

ertData = pb.load("erttrue.dat")

# Build inversion meshes
meshRST = mt.createParaMesh(ertData, paraDepth=15, paraDX=0.2, smooth=[1, 2],
                            paraMaxCellSize=.5, quality=33.5, boundary=0,
                            paraBoundary=3)
meshRST.save("paraDomain.bms")

# meshERT = mt.createParaMesh(ertScheme, quality=33.5, paraMaxCellSize=1.0,
#                             paraDX=0.2, boundary=50, smooth=[1, 10],
#                             paraBoundary=3)
meshERT = mt.appendTriangleBoundary(meshRST, xbound=500, ybound=500,
                                    quality=33.5, isSubSurface=True)
meshERT.save("meshERT.bms")

ert = ERTManager()
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

rst = Refraction("tttrue.dat", verbose=True)

# rst.useFMM(True)
# rst.fop.createRefinedForwardMesh()

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
veltrue = np.loadtxt("veltrue.dat")
startmodel = createGradientModel2D(ttData, meshRST, np.min(veltrue), np.max(veltrue))
np.savetxt("rst_startmodel.dat", 1/startmodel)
vest = rst.invert(ttData, zWeight=1, startModel=startmodel, lam=50)
# vest = rst.inv.runChi1()
print("RST chi:", rst.inv.chi2())
print("RST rms:", rst.inv.relrms())

# Save some stuff
mesh = pg.load("mesh.bms")
velest = pg.interpolate(rst.mesh, vest, mesh.cellCenters())
fpm = FourPhaseModel()
fae, fie, fwe, maske = fpm.all(resinv, vest)
np.savez("conventional.npz", vel=vest, rho=resinv, fa=fae, fi=fie, fw=fwe, mask=maske)

# rst_cov = pg.interpolate(rst.paraDomain(), rst.rayCoverage(),
                         # mesh.cellCenters())
rst.rayCoverage().save("rst_coverage.dat")
