import numpy as np

import pybert as pb
import pygimli as pg
pg.verbose = print # Temporary fix
import pygimli.meshtools as mt

from fpinv import FourPhaseModel, NN_interpolate
from pybert.manager import ERTManager
from pygimli.physics import Refraction
from settings import *

#need ertData, rstData, a mesh and phi to be given
ertData = pb.load("ert_filtered.data")
print(ertData)
mesh = pg.load("mesh_1.bms")
paraDomain = pg.load("paraDomain_1.bms")
depth = mesh.ymax() - mesh.ymin()

ert = ERTManager()
resinv = ert.invert(ertData, mesh=mesh, lam=60, zWeight=zWeight, maxIter=maxIter)
print("ERT chi:", ert.inv.chi2())
print("ERT rms:", ert.inv.relrms())

np.savetxt("res_conventional.dat", resinv)
#############
rst = Refraction("rst_filtered.data", verbose=True)
ttData = rst.dataContainer

# INVERSION
rst.setMesh(mesh, secNodes=3)
from pygimli.physics.traveltime.ratools import createGradientModel2D
minvel = 1000
maxvel = 5000
startmodel = createGradientModel2D(ttData, paraDomain, minvel, maxvel)
np.savetxt("rst_startmodel.dat", 1 / startmodel)
vest = rst.invert(ttData, mesh=paraDomain, zWeight=zWeight, lam=250)

# vest = rst.inv.runChi1()
print("RST chi:", rst.inv.chi2())
print("RST rms:", rst.inv.relrms())

rst.rayCoverage().save("rst_coverage.dat")
np.savetxt("vel_conventional.dat", vest)
