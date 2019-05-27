############################################
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

#need ertData, rstData, a mesh and phi to be given
ertData = pb.load("ert_filtered.data")
print(ertData)
mesh = pg.load("mesh.bms")
paraDomain = pg.load("paraDomain.bms")
depth = mesh.ymax() - mesh.ymin()

############
# Settings
maxIter = 50

ert = ERTManager()
resinv = ert.invert(ertData, mesh=mesh, lam=50, zWeight=0.5, maxIter=maxIter)
print("ERT chi:", ert.inv.chi2())
print("ERT rms:", ert.inv.relrms())

np.savetxt("res_conventional.dat", resinv)
#############
rst = Refraction("rst_filtered.data", verbose=True)
ttData = rst.dataContainer

# INVERSION
rst.setMesh(mesh, secNodes=3)
from pygimli.physics.traveltime.ratools import createGradientModel2D
minvel=700
maxvel=4000
startmodel = createGradientModel2D(ttData, paraDomain, minvel, maxvel)
np.savetxt("rst_startmodel.dat", 1/startmodel)
#vest = rst.invert(ttData, zWeight=1, startModel=startmodel, maxIter=maxIter, lam=10)
#vest = rst.invert(ttData, zWeight=1, maxIter=maxIter, lam=10)
vest=rst.invert(ttData, mesh=paraDomain, zWeight=0.5, lam=100)

# vest = rst.inv.runChi1()
print("RST chi:", rst.inv.chi2())
print("RST rms:", rst.inv.relrms())

rst.rayCoverage().save("rst_coverage.dat")
np.savetxt("vel_conventional.dat", vest)
